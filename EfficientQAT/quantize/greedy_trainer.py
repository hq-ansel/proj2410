import shutil
import functools
import time
import os
import copy
import pdb
import gc
import math
from functools import wraps
from contextlib import contextmanager
from typing import List, Tuple, Dict, Union, Callable
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.amp
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
import logging

from .. import utils
from . import int_linear_fake, int_linear_real
from .utils import (
    quant_parameters,weight_parameters,trainable_parameters,
    set_quant_state,quant_inplace,set_quant_parameters,
    set_weight_parameters,trainable_parameters_num,get_named_linears,set_op_by_name,
    Catcher,StopException,MultiBlock
    )
from ..datautils_block import BlockTrainDataset,OptimBlockTrainDataset,LazyLoadDataset,generate_block_train_data,LazyLoadDatasetV2
from ..loss_utils import get_loss_func


amp_enabled = os.environ.get("AMP_ENABLED", "False").lower() == "true"
print(f"AMP enabled: {amp_enabled}")


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper


def examine_parameters_grad(model:nn.Module,logger:logging.Logger):
    for n, m in model.named_parameters():
        if m.requires_grad and m.grad is not None:
            grad_max = m.grad.abs().max().item()
            if grad_max > 1:
                logger.info(f"{n} grad_max: {grad_max:.4f}.")

class CatcherManager:
    def __init__(self, layers, indices):
        self.layers = layers
        self.indices = indices
        self.original_modules = {}

    def __enter__(self):
        # 添加 Catcher 层并保存原始模块
        for idx in self.indices:
            if isinstance(self.layers[idx], Catcher):
                continue
            self.original_modules[idx] = self.layers[idx]
            self.layers[idx] = Catcher(self.layers[idx])

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始模块
        for idx in self.indices:
            if isinstance(self.layers[idx], Catcher):
                self.layers[idx] = self.original_modules[idx]

class CommonInputDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        res = self.data[idx]
        if len(res.shape) == 2:
            res = res.squeeze(0)
        return res


@timer
def train_units_layers(model: PreTrainedModel,
                        trainable_layer_idx_list: List[int],
                        loss_func: Callable,
                        train_dataset: LazyLoadDataset,
                        val_dataset: LazyLoadDataset,
                        attention_mask: torch.Tensor,
                        position_embeddings: Tuple[torch.Tensor,torch.Tensor],
                        loss_recorder,
                        logger,
                        args):
    # 解冻当前层，并冻结其它层
    # model.to(args.dev)
    total_training_iteration = args.epochs * args.train_size / args.batch_size
    layer_idx_set = set(trainable_layer_idx_list)
    step = 0
    param_groups = []
    param_group_index = 0
    assert args.quant_lr > 0 or args.weight_lr > 0

    # 使用amp仍然需要权重为float32
    with torch.no_grad():
        if amp_enabled:
            selected_layers.to(args.dev, dtype=torch.float32)
        else:
            model.model.layers = nn.ModuleList(
                    [qlayer.to(args.dev, dtype=torch.float32)
                        if index in layer_idx_set 
                        else qlayer.half() for index,
                            qlayer in enumerate(model.model.layers)])
        
    qlayers = model.model.layers
    selected_layers = [qlayers[i] for i in trainable_layer_idx_list]
    selected_layers = nn.ModuleList(selected_layers)

    # 暂时没有更好的假设，直接使用selected_layers 中的最后一个层作为对齐层
    align_index = trainable_layer_idx_list[-1]
    last_block_idx = len(model.model.layers) - 1
    if args.loss_func == "KL-Divergence":
        qlayer_idxs = []
        fp_layer_idxs = []
    else:
        qlayer_idxs = [align_index,last_block_idx]
        fp_layer_idxs = [align_index,last_block_idx]

        for name, param in model.named_parameters():
            param.requires_grad = False
        for layer_idx in trainable_layer_idx_list:
            qlayer = model.model.layers[layer_idx]
            set_quant_state(qlayer,True)
            if args.quant_lr > 0:
                set_quant_parameters(qlayer,True)
                param_groups.append({"params":quant_parameters(qlayer),
                                    "lr":args.quant_lr})
                empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)],
                                                    lr=args.quant_lr)
                quant_scheduler = CosineAnnealingLR(empty_optimizer_1,
                                            T_max=total_training_iteration,
                                            eta_min=args.quant_lr/args.min_lr_factor)
                quant_index = param_group_index
                param_group_index += 1
            else:
                set_quant_parameters(qlayer,False)
                
            if args.weight_lr > 0:
                set_weight_parameters(qlayer,True)
                param_groups.append({"params":weight_parameters(qlayer),
                                    "lr":args.weight_lr})
                empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)],
                                                    lr=args.weight_lr)
                weight_scheduler = CosineAnnealingLR(empty_optimizer_2,
                                                T_max=total_training_iteration,
                                                eta_min=args.weight_lr/args.min_lr_factor)
                weight_index = param_group_index
                param_group_index += 1
            else:
                set_weight_parameters(qlayer,False)
        
        if args.loss_func == "AFFINE_MSE":
            print("using AFFINE_MSE loss function")
            loss_func.reinitialize_A()
            loss_func = loss_func.to(args.dev)
            param_groups.append({"params":loss_func.parameters(),"lr":args.weight_lr})
        optimizer =torch.optim.AdamW(param_groups,
                                    weight_decay=args.wd,
                                    foreach=True)
        qlayers = model.model.layers
        loss_scaler= torch.amp.GradScaler(device=args.dev)
        trainable_number = trainable_parameters_num(selected_layers)
        print(f"trainable parameter number: {trainable_number/1e6}M")
        # 参数内存（float32），每个参数 4 字节
        # 混合精度 AMP 梯度内存 (bfloat16)，每个参数 2 字节
        # AdamW 动量状态内存（float32），每个状态 4 字节，共两个状态 (m, v)
        print(f"estimated memory usage: {trainable_number*(4+2+2*4)/(1024**3)}G")
        print(f"estimated data usage: {args.batch_size*args.training_seqlen*model.config.hidden_size*4/(1024**3)}G")
        best_val_loss = 1e6
        early_stop_flag = 0
        if args.get("gradual_quant",False) or args.get("interpolate",False):
            class GradualWarmupScheduler:
                def __init__(self,
                              linear_list:List[int_linear_fake.QuantLinear],
                              total_iteration:int,):
                    self.linear_list = linear_list
                    self.total_iteration = total_iteration
                    self.iteration = 0
                    self.update()
                def update(self):
                    self.iteration += 1
                    for linear in self.linear_list:
                        if self.iteration < self.total_iteration/2:
                            ratio = self.iteration/self.total_iteration/2
                            if args.get("gradual_quant",False):
                                linear.update_position_ratio(ratio)
                            if args.get("interpolate", False):
                                linear.update_interpolate_ratio(1-ratio)
                        else:
                            linear.update_position_ratio(1.0)
                            if args.get("interpolate", False):
                                linear.update_interpolate_ratio(0)
            q_linear_list = []
            for i in trainable_layer_idx_list:
                for n,m in qlayers[i].named_modules():
                    if isinstance(m, int_linear_fake.QuantLinear):
                        q_linear_list.append(m)
            graualWarmupScheduler = GradualWarmupScheduler(
                q_linear_list,
                total_training_iteration,
            )
        # step 6.3: training loop
        position_ids = torch.arange(args.training_seqlen, dtype=torch.long, device=args.dev)
        position_ids = position_ids.unsqueeze(0).expand(args.batch_size, -1).contiguous()
        # print(f" data size {len(train_dataset)}")
        for epoch in range(args.epochs):
            loss_list = []
            norm_list = []
            start_time = time.time()
            # used for debug
            torch.autograd.set_detect_anomaly(True)
            dataloader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=8,
                                    pin_memory=True,
                                    prefetch_factor=16,  
                                    shuffle=True
                                    )
            # step 6.4: training                   
            for index, input_data in enumerate(dataloader):
                with torch.autocast(device_type=args.dev,
                                    enabled=amp_enabled,
                                    dtype=args.dtype if amp_enabled else torch.float32):
                    inp,target = input_data
                    hidden_state = inp.to(args.dev,dtype=args.dtype)
                    # assert hidden_state.requires_grad == True, "hidden_state.requires_grad is False"
                    # hidden_state.requires_grad = True
                    for layer_idx in trainable_layer_idx_list:
                        # 使用 lambda 将关键字参数改为位置参数
                        # layer_func = functools.partial(
                        #     qlayers[layer_idx].forward,  # 指定需要调用的函数
                        #     attention_mask=attention_mask,
                        #     position_embeddings=position_embeddings
                        # )
                        
                        # # 在 checkpoint 中使用 layer_func 作为函数调用
                        # layer_outputs = checkpoint(layer_func, hidden_state)
                        layer_outputs = qlayers[layer_idx](
                            hidden_states=hidden_state,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings
                        )
                        hidden_state = layer_outputs[0]
                    if index == 0:
                        tmp = {
                            "hidden_state":inp,
                            "attention_mask":attention_mask,
                            "position_embeddings":position_embeddings,
                        }
                        print(f"layers {trainable_layer_idx_list} input_data {tmp} output {hidden_state} target {target} ")
                    loss = loss_func(hidden_state, target.to(args.dev,dtype=torch.float32))
                                
                if not math.isfinite(loss.item()) or loss.item()==0:
                    logger.info("Loss is NAN, stopping training")
                    pdb.set_trace()
                if args.log_loss:
                    loss_recorder.record(f"blk{trainable_layer_idx_list}",
                                        step,
                                        loss.detach().cpu().item())
                graualWarmupScheduler.update() if args.get("gradual_quant",False) else None
                loss_list.append(loss.detach().cpu())
                # 反向传播和优化
                if not args.loss_func == "KL-Divergence":
                    optimizer.zero_grad()
                loss_scaler.scale(loss).backward() if amp_enabled else loss.backward()
                # debug 检查grad
                if amp_enabled: loss_scaler.unscale_(optimizer)
                if args.clip_grad > 0:
                        # print(f"clip grad at {args.clip_grad}")
                        norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(selected_layers), args.clip_grad).cpu()
                if not args.loss_func == "KL-Divergence":
                    if amp_enabled:
                        loss_scaler.step(optimizer)
                        loss_scaler.update()
                    else:
                        optimizer.step()
                norm_list.append(norm.data)
                
                # adjust lr
                if args.quant_lr > 0:
                    quant_scheduler.step()
                    optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]
                if args.weight_lr >0 :
                    weight_scheduler.step()
                    optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[0]
                step += 1


            # step 6.5: calculate validation loss
            val_loss_list = []
            dataloader = DataLoader(val_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=8,
                                    pin_memory=True,
                                    prefetch_factor=4,  
                                    shuffle=False
                                    )
            # if not args.loss_func == "KL-Divergence":
            #     qlayers[align_index].set_forward_state(stop_forward=False)
            for index, input_data in enumerate(dataloader):  
                # obtain output of quantization model
                with torch.no_grad():
                    with torch.autocast(device_type=args.dev,
                                    enabled=amp_enabled,
                                    dtype=args.dtype if amp_enabled else torch.float32):
                        inp,target = input_data
                        hidden_state = inp.to(args.dev,dtype=args.dtype)
                        # assert hidden_state.requires_grad == True, "hidden_state.requires_grad is False"
                        for layer_idx in trainable_layer_idx_list:
                            layer_outputs = qlayers[layer_idx](
                                hidden_states=hidden_state,
                                attention_mask=attention_mask,
                                position_embeddings=position_embeddings
                            )
                            hidden_state = layer_outputs[0]
                        loss = loss_func(hidden_state, target.to(args.dev,dtype=torch.float32))
                val_loss_list.append(loss.cpu())

                # if not args.loss_func == "KL-Divergence":
                #     final_val_list.append(final_loss.cpu())
            if not args.loss_func and  args.align_type == "tail":
                qlayers[align_index].set_forward_state(stop_forward=True)

            train_mean_num = min(len(loss_list),64) 
            # calculate the average training loss of last train_mean_num samples
            loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
            val_loss_mean = torch.stack(val_loss_list).mean()
            norm_mean = torch.stack(norm_list).mean()
            logger.info(f"blocks {trainable_layer_idx_list} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(args.dev) / 1024**2} time {time.time()-start_time} ")
            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
            else:
                early_stop_flag += 1
                if args.early_stop > 0 and early_stop_flag >=args.early_stop:
                    break
        optimizer.zero_grad()
        del optimizer
        # step 7: pack quantized weights into low-bits format, note that this process is slow on poor CPU or busy CPU
        
    torch.cuda.empty_cache()



@timer
def train_units_layers_with_catcher(model: PreTrainedModel,
                        trainable_layer_idx_list: List[int],
                        loss_func: Callable,
                        train_dataset: LazyLoadDataset,
                        val_dataset: LazyLoadDataset,
                        target_model: PreTrainedModel,
                        loss_recorder,
                        logger,
                        args):
    # 解冻当前层，并冻结其它层
    model.to(args.dev)
    total_training_iteration = args.epochs * args.train_size / args.batch_size
    layer_idx_set = set(trainable_layer_idx_list)
    step = 0
    param_groups = []
    param_group_index = 0
    assert args.quant_lr > 0 or args.weight_lr > 0

    # 使用amp仍然需要权重为float32
    with torch.no_grad():
        if amp_enabled:
            selected_layers.to(args.dev, dtype=torch.float32)
        else:
            model.model.layers = nn.ModuleList(
                    [qlayer.to(args.dev, dtype=torch.float32)
                        if index in layer_idx_set 
                        else qlayer.half() for index,
                            qlayer in enumerate(model.model.layers)])
    fp_layers = target_model.model.layers
    qlayers = model.model.layers
    selected_layers = [qlayers[i] for i in trainable_layer_idx_list]
    selected_layers = nn.ModuleList(selected_layers)

    # 暂时没有更好的假设，直接使用selected_layers 中的最后一个层作为对齐层
    align_index = trainable_layer_idx_list[-1]
    last_block_idx = len(model.model.layers) - 1
    # if args.loss_func == "KL-Divergence":
    #     qlayer_idxs = []
    #     fp_layer_idxs = []
    # else:
    #     qlayer_idxs = [align_index,last_block_idx]
    #     fp_layer_idxs = [align_index,last_block_idx]
    
    start_block = trainable_layer_idx_list[0]
    qlayer_idxs = [align_index,start_block]
    fp_layer_idxs = [align_index,start_block]

    with CatcherManager(qlayers, qlayer_idxs),CatcherManager(fp_layers, fp_layer_idxs):
        if not args.loss_func == "KL-Divergence":
            if args.align_type == "tail":
                qlayers[align_index].set_forward_state(stop_forward=True)
                qlayers[start_block].detach_input = True
                # qlayers[align_index].set_output_catch_state(state=True)
            else:
                qlayers[last_block_idx].set_forward_state(stop_forward=True)
                # qlayers[last_block_idx].set_output_catch_state(state=True)

        for name, param in model.named_parameters():
            param.requires_grad = False
        for layer_idx in trainable_layer_idx_list:
            qlayer = model.model.layers[layer_idx]
            set_quant_state(qlayer,True)
            if args.quant_lr > 0:
                set_quant_parameters(qlayer,True)
                param_groups.append({"params":quant_parameters(qlayer),
                                    "lr":args.quant_lr})
                empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)],
                                                    lr=args.quant_lr)
                quant_scheduler = CosineAnnealingLR(empty_optimizer_1,
                                            T_max=total_training_iteration,
                                            eta_min=args.quant_lr/args.min_lr_factor)
                quant_index = param_group_index
                param_group_index += 1
            else:
                set_quant_parameters(qlayer,False)
                
            if args.weight_lr > 0:
                set_weight_parameters(qlayer,True)
                param_groups.append({"params":weight_parameters(qlayer),
                                    "lr":args.weight_lr})
                empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)],
                                                    lr=args.weight_lr)
                weight_scheduler = CosineAnnealingLR(empty_optimizer_2,
                                                T_max=total_training_iteration,
                                                eta_min=args.weight_lr/args.min_lr_factor)
                weight_index = param_group_index
                param_group_index += 1
            else:
                set_weight_parameters(qlayer,False)
        
        if args.loss_func == "AFFINE_MSE":
            print("using AFFINE_MSE loss function")
            loss_func.reinitialize_A()
            loss_func = loss_func.to(args.dev)
            param_groups.append({"params":loss_func.parameters(),"lr":args.weight_lr})
        optimizer =torch.optim.AdamW(param_groups,
                                    weight_decay=args.wd,
                                    foreach=True)
        qlayers = model.model.layers
        # loss_scaler = utils.NativeScalerWithGradNormCount(use_amp=amp_enabled)
        loss_scaler= torch.amp.GradScaler(device=args.dev)
        trainable_number = trainable_parameters_num(selected_layers)
        print(f"trainable parameter number: {trainable_number/1e6}M")
        # 参数内存（float32），每个参数 4 字节
        # 混合精度 AMP 梯度内存 (bfloat16)，每个参数 2 字节
        # AdamW 动量状态内存（float32），每个状态 4 字节，共两个状态 (m, v)
        print(f"estimated memory usage: {trainable_number*(4+2+2*4)/(1024**3)}G")
        print(f"estimated data usage: {args.batch_size*args.training_seqlen*model.config.hidden_size*4/(1024**3)}G")
        best_val_loss = 1e6
        early_stop_flag = 0
        if args.get("gradual_quant",False) or args.get("interpolate",False):
            class GradualWarmupScheduler:
                def __init__(self,
                              linear_list:List[int_linear_fake.QuantLinear],
                              total_iteration:int,):
                    self.linear_list = linear_list
                    self.total_iteration = total_iteration
                    self.iteration = 0
                    self.update()
                def update(self):
                    self.iteration += 1
                    for linear in self.linear_list:
                        if self.iteration < self.total_iteration/2:
                            ratio = self.iteration/self.total_iteration/2
                            if args.get("gradual_quant",False):
                                linear.update_position_ratio(ratio)
                            if args.get("interpolate", False):
                                linear.update_interpolate_ratio(1-ratio)
                        else:
                            linear.update_position_ratio(1.0)
                            if args.get("interpolate", False):
                                linear.update_interpolate_ratio(0)
            q_linear_list = []
            for i in trainable_layer_idx_list:
                for n,m in qlayers[i].named_modules():
                    if isinstance(m, int_linear_fake.QuantLinear):
                        q_linear_list.append(m)
            graualWarmupScheduler = GradualWarmupScheduler(
                q_linear_list,
                total_training_iteration,
            )
        # step 6.3: training loop
        print(f" data size {len(train_dataset)}")
        for epoch in range(args.epochs):
            loss_list = []
            norm_list = []
            start_time = time.time()
            # used for debug
            # torch.autograd.set_detect_anomaly(True)
            dataloader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=8,
                                    pin_memory=True,
                                    prefetch_factor=16,  
                                    shuffle=True
                                    )
            # step 6.4: training                   
            for index, input_data in enumerate(dataloader):
                with torch.autocast(device_type=args.dev,
                                    enabled=amp_enabled,
                                    dtype=args.dtype if amp_enabled else torch.float32):
                    inp = input_data
                    assert inp.dim() == 2, "input dim should be 2"
                    try:
                        output = model(inp.to(args.dev))
                    except StopException as e:
                            pass
                    output = qlayers[align_index].outs[0] # outs[0] is out tensor
                    with torch.no_grad():
                        try:
                            target_output = target_model(inp.to("cuda:1"))[0]
                        except StopException as e:
                            pass
                        target_output = fp_layers[align_index].outs[0]
                    if index == 0:
                        inp = qlayers[align_index].inps
                        print(f"input {inp} output {output} target_output {target_output}")
                    loss = loss_func(output, target_output.to(args.dev,dtype=torch.float32))

                          
                if not math.isfinite(loss.item()) or loss.item()==0:
                    logger.info("Loss is NAN, stopping training")
                    pdb.set_trace()
                if args.log_loss:
                    loss_recorder.record(f"blk{trainable_layer_idx_list}",
                                        step,
                                        loss.detach().cpu().item())
                graualWarmupScheduler.update() if args.get("gradual_quant",False) else None
                loss_list.append(loss.detach().cpu())
                # 反向传播和优化
                if not args.loss_func == "KL-Divergence":
                    optimizer.zero_grad()
                loss_scaler.scale(loss).backward() if amp_enabled else loss.backward()
                # debug 检查grad
                if amp_enabled: loss_scaler.unscale_(optimizer)
                if args.clip_grad > 0:
                        # print(f"clip grad at {args.clip_grad}")
                        norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(selected_layers), args.clip_grad).cpu()
                if not args.loss_func == "KL-Divergence":
                    if amp_enabled:
                        loss_scaler.step(optimizer)
                        loss_scaler.update()
                    else:
                        optimizer.step()
                norm_list.append(norm.data)
                
                # adjust lr
                if args.quant_lr > 0:
                    quant_scheduler.step()
                    optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]
                if args.weight_lr >0 :
                    weight_scheduler.step()
                    optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[0]
                step += 1


            # step 6.5: calculate validation loss
            val_loss_list = []
            dataloader = DataLoader(val_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=8,
                                    pin_memory=True,
                                    prefetch_factor=4,  
                                    shuffle=False
                                    )
            # if not args.loss_func == "KL-Divergence":
            #     qlayers[align_index].set_forward_state(stop_forward=False)
            for index, input_data in enumerate(dataloader):  
                # obtain output of quantization model
                with torch.no_grad():
                    with torch.autocast(device_type=args.dev,
                                    enabled=amp_enabled,
                                    dtype=args.dtype if amp_enabled else torch.float32):
                        inp = input_data
                        try:
                            output = model(inp.to(args.dev))
                        except StopException:
                            pass
                        output = qlayers[align_index].outs[0] # outs[0] is out tensor
                        try:
                            target_output = target_model(inp.to("cuda:1"))[0]
                        except StopException:
                            pass
                        target_output = fp_layers[align_index].outs[0].to(torch.float32)
                        loss = loss_func(output, target_output.to(args.dev,dtype=torch.float32))
                val_loss_list.append(loss.cpu())


                # if not args.loss_func == "KL-Divergence":
                #     final_val_list.append(final_loss.cpu())
            if not args.loss_func and  args.align_type == "tail":
                qlayers[align_index].set_forward_state(stop_forward=True)

            train_mean_num = min(len(loss_list),64) 
            # calculate the average training loss of last train_mean_num samples
            loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
            val_loss_mean = torch.stack(val_loss_list).mean()
            norm_mean = torch.stack(norm_list).mean()
            logger.info(f"blocks {trainable_layer_idx_list} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(args.dev) / 1024**2} time {time.time()-start_time} ")
            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
            else:
                early_stop_flag += 1
                if args.early_stop > 0 and early_stop_flag >=args.early_stop:
                    break


        optimizer.zero_grad()
        del optimizer
        # step 7: pack quantized weights into low-bits format, note that this process is slow on poor CPU or busy CPU
        
    torch.cuda.empty_cache()




def trans_quant_block(qlayer:nn.Module,args):
    for name, module in qlayer.named_modules():
        if isinstance(module,torch.nn.Linear):
            quantlinear = int_linear_fake.QuantLinear(module,
                                args.wbits,
                                args.group_size,
                                args)
            quantlinear.set_quant_state(True)
            set_op_by_name(qlayer, name, quantlinear)  
            del module  
    return qlayer

def custom_shedule_train(model:PreTrainedModel,
                    train_dataset: LazyLoadDatasetV2,
                    val_dataset: LazyLoadDatasetV2,
                    attention_mask : torch.Tensor,
                    position_embeddings : Tuple[torch.Tensor, torch.Tensor],
                    logger: logging.Logger,
                    args):
    # model.to(args.dev)

    target_model = copy.deepcopy(model)
    target_model.to("cuda:1")

    shedule_list= []
    # 暂时调度的内容是直接平移一个
    # offset = 1
    if args.quant_shedule_type == "full":
        for i in range(len(model.model.layers)):
            model.model.layers[i] = trans_quant_block(qlayer=model.model.layers[i],
                                                      args=args)
    else: is_quant_layer = [False]*len(model.model.layers)
    if args.train_shedule_type == "start2end":
        num_layers = len(model.model.layers)
        for start in range(0,num_layers,args.slide_step):
            end = min(start + args.crossblock_window_size, num_layers)
            shedule_list.append(list(range(start, end)))
    elif args.train_shedule_type == "end2start":
        num_layers = len(model.model.layers)
        for end in range(num_layers, 0, -1*args.slide_step):
            start = max(end - args.crossblock_window_size, 0)
            shedule_list.append(list(range(start, end)))
    
    logger.info(f"use loss func {args.loss_func} ")

    if args.loss_func == "KL-Divergence":
        loss_func = None
    elif args.loss_func == "AFFINE_MSE":
        loss_func = get_loss_func(args.loss_func)(model.config.hidden_size)
    else:
        loss_func = get_loss_func(args.loss_func)
    loss_recorder = utils.BlockLossRecorder(file_path=args.log_loss,)
    
    for train_layer_window in shedule_list:
        if not args.quant_shedule_type == "full" :
            for layer_idx in train_layer_window:
                if not is_quant_layer[layer_idx]:
                    is_quant_layer[layer_idx] = True
                    model.model.layers[layer_idx] = trans_quant_block(
                                            qlayer=model.model.layers[layer_idx],
                                                                      args=args)
        if args.epochs > 0 :
            logger.info(f"train blocks {train_layer_window}")
            # assert attention_mask is not None , "attention_mask is None"
            skip_flag=True
            skip_layers = args.get("skip_layers",[])
            if len(skip_layers) > 0:
                for layer_idx in train_layer_window:
                    if layer_idx not in skip_layers:
                        skip_flag=False
                        break
            else:
                skip_flag=False
            if  not skip_flag :
                if not args.with_catcher:
                    train_units_layers(model,
                            trainable_layer_idx_list=train_layer_window,
                            loss_func=loss_func,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            attention_mask=attention_mask,
                            position_embeddings=position_embeddings,
                            loss_recorder=loss_recorder,
                            logger=logger,
                            args=args)
                else: 
                    train_units_layers_with_catcher(model,
                        trainable_layer_idx_list=train_layer_window,
                        loss_func=loss_func,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        target_model=target_model,
                        loss_recorder=loss_recorder,
                        logger=logger,
                        args=args)
            selected_layers= nn.ModuleList([model.model.layers[i] for i in train_layer_window])
            quant_inplace(selected_layers)
            for layer_idx in train_layer_window:
                qlayer = model.model.layers[layer_idx]
                set_quant_state(qlayer,False)
                # step 7: pack quantized weights into low-bits format, note that this process is slow on poor CPU or busy CPU
                if args.real_quant:
                    named_linears = get_named_linears(qlayer, int_linear_fake.QuantLinear)
                    for name, module in named_linears.items():
                        scales = module.weight_quantizer.scale.clamp(1e-4,1e4).detach()
                        zeros = module.weight_quantizer.zero_point.detach().cuda().round().cpu()
                        group_size = module.weight_quantizer.group_size
                        dim0 = module.weight.shape[0]
                        scales = scales.view(dim0,-1).transpose(0,1).contiguous()
                        zeros = zeros.view(dim0,-1).transpose(0,1).contiguous()
                        q_linear = int_linear_real.QuantLinear(args.wbits,
                                                    group_size,
                                                    module.in_features,
                                                    module.out_features,
                                                    not module.bias is None,
                                                    clamp_input= args.get("clamp_input",False))
                        q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                        set_op_by_name(qlayer, name, q_linear)       
                        logger.info(f"pack quantized {name} finished")
                        del module
                torch.cuda.empty_cache()
                gc.collect()
            if amp_enabled:
                qlayer= qlayer.to(dtype=args.dtype)
            # 更新数据
            # _dtype = attention_mask.dtype
            # attention_mask = attention_mask.to(dtype=torch.float32)
            # DEBUG
            # position_embeddings = (position_embeddings[0].to(dtype=torch.float32),position_embeddings[1].to(dtype=torch.float32))
            # DEBUG
            if not args.with_catcher:
                if train_layer_window != shedule_list[-1]:
                    with torch.no_grad():
                        with torch.autocast(device_type=args.dev,
                                            enabled=amp_enabled,
                                            dtype=args.dtype if amp_enabled else torch.float32):
                            for slide_base in train_layer_window:
                                # 更新后的input 需要经过[windows_start,windows_start+slide_step)层的输出
                                if slide_base == train_layer_window[0]+args.slide_step:
                                    break
                                layer_idx = slide_base 
                                # print(f"slide_base {slide_base} layer_idx {layer_idx}")
                                layer = model.model.layers[layer_idx].to(args.dev,dtype=args.dtype)
                                next_layer = model.model.layers[layer_idx+args.slide_step].to(args.dev,dtype=args.dtype)
                                print(f" layer {layer_idx}  update input")
                                print(f" layer {layer_idx+args.slide_step}  update output")
                                train_dataset.update_dataset(module=layer, 
                                                        next_module=next_layer,
                                                        layer_idx=layer_idx+args.slide_step,
                                                        batch_size=args.batch_size,
                                                        attention_mask=attention_mask,
                                                        position_embeddings=position_embeddings,
                                                            )
                                val_dataset.update_dataset(module=layer, 
                                                        next_module=next_layer,
                                                        layer_idx=layer_idx+args.slide_step,
                                                        batch_size=args.batch_size,
                                                        attention_mask=attention_mask,
                                                        position_embeddings=position_embeddings,
                                                            )
                                layer.cpu()
                                next_layer.cpu()
            # attention_mask = attention_mask.to(dtype=_dtype)
            # position_embeddings = (position_embeddings[0].to(dtype=_dtype),position_embeddings[1].to(dtype=_dtype))
            # 保存模型
    if args.log_loss:
        loss_recorder.save_to_file()
    torch.cuda.empty_cache()
    gc.collect()

@timer
def greedy_local_train(
    model: PreTrainedModel,
    args,
    trainloader: List[Tuple[torch.Tensor, torch.Tensor]],
    valloader: List[Tuple[torch.Tensor, torch.Tensor]],
    logger: logging.Logger=None,
):
    """
    Args:
        model: PreTrainedModel,
        args: argparse.Namespace,
        trainloader: List[Tuple[torch.Tensor, torch.Tensor]] Tensor[batch_size, seq_len, hidden_size],
        valloader: List[Tuple[torch.Tensor, torch.Tensor]],
        logger: logging.Logger=None,
    """
    logger.info("Starting ...")
    if args.off_load_to_disk:
        logger.info("offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")
    
    dev ="cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if amp_enabled else torch.float32
    args.dev = dev
    args.dtype = dtype
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # step 2: init dataset 改回原来的模式，存入当前层的输入
    # 首先准备需要位置编码和embedding的部分
    cache_dir = os.path.join(args.cache_dir, args.net)
    cache_dir = os.path.join(cache_dir, f"{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}")
    print(f"try to load traindata from cache {cache_dir}")
    train_dataset_path = os.path.join(cache_dir, "train_dataset.pt")

    model = model.cpu()
    if not args.with_catcher:
        print("using catch")
        train_dataset = LazyLoadDatasetV2(
            model=model,
            dataloader=trainloader,
            cache_path=train_dataset_path,
        )
        # 准备验证集
        val_dataset = LazyLoadDatasetV2(
            model=model,
            dataloader=valloader,
            cache_path=os.path.join(cache_dir, "val_dataset.pt"),
        )
        if train_dataset.attention_mask is None:
            attention_mask = train_dataset.attention_mask
        else:
            attention_mask = train_dataset.attention_mask.to("cuda")
        position_embeddings = (train_dataset.position_embeddings[0].to("cuda"),
                            train_dataset.position_embeddings[1].to("cuda"))

        del trainloader, valloader
        gc.collect()

        custom_shedule_train(model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                attention_mask = attention_mask,
                position_embeddings = position_embeddings,
                logger=logger,
                args=args)
    else:
        common_train_inps = CommonInputDataset(list(map(lambda x: x[0], trainloader)))
        common_val_inps = CommonInputDataset(list(map(lambda x: x[0], valloader)))
        custom_shedule_train(model,
                train_dataset=common_train_inps,
                val_dataset=common_val_inps,
                attention_mask=None,
                position_embeddings=None,
                logger=logger,
                args=args)

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

