import shutil
import time
import os
import copy
import pdb
import gc
import math
from functools import wraps
from contextlib import contextmanager
from typing import List, Tuple, Dict, Union, Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.checkpoint as checkpoint
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
    Catcher
    )
from ..datautils_block import BlockTrainDataset,OptimBlockTrainDataset
from ..loss_utils import get_loss_func


amp_enabled = os.environ.get("AMP_ENABLED", "False").lower() == "true"
print(f"AMP enabled: {amp_enabled}")

class StopException(Exception):
    pass

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
        if m.requires_grad:
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
            self.original_modules[idx] = self.layers[idx]
            self.layers[idx] = Catcher(self.layers[idx])

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始模块
        for idx in self.indices:
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
                        train_dataset: Dataset,
                        val_dataset: Dataset,
                        target_model: PreTrainedModel,
                        loss_recorder,
                        logger,
                        args):
    # 解冻当前层，并冻结其它层
    model.to(args.dev)
    target_model.cpu()
    total_training_iteration = args.epochs * args.train_size / args.batch_size
    layer_idx_set = set(trainable_layer_idx_list)
    step = 0
    param_groups = []
    param_group_index = 0
    assert args.quant_lr > 0 or args.weight_lr > 0
    # 使用amp仍然需要权重为float32
    with torch.no_grad():
        model.model.layers = nn.ModuleList(
            [qlayer.float() 
                   if index in layer_idx_set 
                   else qlayer for index,
                     qlayer in enumerate(model.model.layers)])
    fp_layers = target_model.model.layers
    qlayers = model.model.layers
    selected_layers = [qlayers[i] for i in trainable_layer_idx_list]
    selected_layers = nn.ModuleList(selected_layers)

    # 暂时没有更好的假设，直接使用selected_layers 中的最后一个层作为对齐层
    align_index = trainable_layer_idx_list[-1]
    last_block_idx = len(model.model.layers) - 1
    with CatcherManager(qlayers, [align_index,last_block_idx]),CatcherManager(fp_layers, [align_index,last_block_idx]):
        qlayers[align_index].set_forward_state(stop_forward=True)
        fp_layers[align_index].set_forward_state(stop_forward=True)
        
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
        optimizer =torch.optim.AdamW(param_groups,
                                    weight_decay=args.wd,
                                    foreach=True)
        qlayers = model.model.layers
        loss_scaler = utils.NativeScalerWithGradNormCount(use_amp=amp_enabled)
        trainable_number = trainable_parameters_num(selected_layers)
        print(f"trainable parameter number: {trainable_number/1e6}M")
        # 参数内存（float32），每个参数 4 字节
        # 混合精度 AMP 梯度内存 (bfloat16)，每个参数 2 字节
        # AdamW 动量状态内存（float32），每个状态 4 字节，共两个状态 (m, v)
        print(f"estimated memory usage: {trainable_number*(4+2+2*4)/(1024**3)}G")
        best_val_loss = 1e6
        early_stop_flag = 0
        # step 6.3: training loop
        for epoch in range(args.epochs):
            loss_list = []
            norm_list = []
            start_time = time.time()
            # used for debug
            torch.autograd.set_detect_anomaly(True)
            dataloader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True
                                    )
            # step 6.4: training                   
            for index, input_data in enumerate(dataloader):
                with torch.autocast(device_type=args.dev,
                                    enabled=amp_enabled,
                                    dtype=args.dtype):
                    
                    # debug 检查grad
                    if trainable_layer_idx_list in ([0], [1], [8],[19]):
                        examine_parameters_grad(model,logger)
                    
                    with torch.no_grad():
                        model.cpu()
                        target_model = target_model.to(args.dev)
                        try:
                            target_output = target_model(input_data.to(args.dev))
                        except StopException:
                            pass
                        except Exception as e:
                            raise e
                        target_output = fp_layers[align_index].outs[0]
                        target_model = target_model.cpu()
                        model.to(args.dev)

                    try:
                        output = model(input_data.to(args.dev))
                    except StopException:
                        pass
                    except Exception as e:
                        raise e
                    output = qlayers[align_index].outs[0] # outs[0] is out tensor

                # 获取当前层的自定义损失
                    loss = loss_func(output, target_output)
                if not math.isfinite(loss.item()) or loss.item()==0:
                    logger.info("Loss is NAN, stopping training")
                    pdb.set_trace()
                if args.log_loss:
                    loss_recorder.record(f"blk{trainable_layer_idx_list}",
                                        step,
                                        loss.detach().cpu().item())
                loss_list.append(loss.detach().cpu())
                # 反向传播和优化
                optimizer.zero_grad()
                norm = loss_scaler(loss,
                        optimizer,
                        clip_grad=args.clip_grad,
                        parameters=trainable_parameters(selected_layers)).cpu()
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
                                    shuffle=False
                                    )
            for index, input_data in enumerate(dataloader):  
                # obtain output of quantization model
                with torch.no_grad():
                    with torch.autocast(device_type=args.dev,
                                        enabled=amp_enabled,
                                        dtype=args.dtype):
                        try:
                            output = model(input_data.to(args.dev))
                        except ValueError:
                            pass
                        output = qlayers[align_index].outs[0] # outs[0] is out tensor
                        try:
                            target_output = target_model(input_data.to(args.dev))[0]
                        except ValueError:
                            pass
                        target_output = fp_layers[align_index].outs[0].to(torch.float32)
                        loss = loss_func(output, target_output)
                val_loss_list.append(loss.cpu())
                
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
        # step 7: pack quantized weights into low-bits format, note that this process is slow on poor CPU or busy CPU
        quant_inplace(selected_layers)
        for layer_idx in trainable_layer_idx_list:
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
                                                not module.bias is None)
                    q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                    set_op_by_name(qlayer, name, q_linear)       
                    logger.info(f"pack quantized {name} finished")
                    del module
    torch.cuda.empty_cache()
def trans_quant_block(qlayer:nn.Module,args):
    for name, module in qlayer.named_modules():
        if isinstance(module,torch.nn.Linear):
            quantlinear = int_linear_fake.QuantLinear(module,
                                args.wbits,
                                args.group_size,
                                args)
            set_op_by_name(qlayer, name, quantlinear)  
            del module  
    return qlayer

def custom_shedule_train(model:PreTrainedModel,
                    train_dataset: Dataset,
                    val_dataset: Dataset,
                    target_model:PreTrainedModel,
                    logger: logging.Logger,
                    args):
    target_model.to(args.dev)
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
        for start in range(num_layers):
            end = min(start + args.crossblock_window_size, num_layers)
            shedule_list.append(list(range(start, end)))
    elif args.train_shedule_type == "end2start":
        num_layers = len(model.model.layers)
        for end in range(num_layers, 0, -1):
            start = max(end - args.crossblock_window_size, 0)
            shedule_list.append(list(range(start, end)))
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
        if args.epochs > 0:
            logger.info(f"train blocks {train_layer_window}")
            train_units_layers(model,
                    trainable_layer_idx_list=train_layer_window,
                    loss_func=loss_func,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    target_model=target_model,
                    loss_recorder=loss_recorder,
                    logger=logger,
                    args=args)
    if args.log_loss:
        loss_recorder.save_to_file()
    torch.cuda.empty_cache()
    gc.collect()


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
    dtype = torch.float16
    args.dev = dev
    args.dtype = dtype
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # step 2: init dataset

    common_train_inps = CommonInputDataset(list(map(lambda x: x[0], trainloader)))
    common_val_inps = CommonInputDataset(list(map(lambda x: x[0], valloader)))

    del trainloader, valloader
    gc.collect()

    custom_shedule_train(model,
            train_dataset=common_train_inps,
            val_dataset=common_val_inps,
            target_model = copy.deepcopy(model),
            logger=logger,
            args=args)

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

