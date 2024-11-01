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
import torch.amp
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
    Catcher,StopException
    )
from ..datautils_block import BlockTrainDataset,OptimBlockTrainDataset
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
    target_model.to("cuda:1")
    total_training_iteration = args.epochs * args.train_size / args.batch_size
    layer_idx_set = set(trainable_layer_idx_list)
    step = 0
    param_groups = []
    param_group_index = 0
    assert args.quant_lr > 0 or args.weight_lr > 0
    # 使用amp仍然需要权重为float32
    if amp_enabled:
        model.to(args.dev, dtype=torch.float32)
    else:
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
    if args.loss_func == "KL-Divergence":
        qlayer_idxs = []
        fp_layer_idxs = []
    else:
        qlayer_idxs = [align_index,last_block_idx]
        fp_layer_idxs = [align_index,last_block_idx]

    with CatcherManager(qlayers, qlayer_idxs),CatcherManager(fp_layers, fp_layer_idxs):
        if not args.loss_func == "KL-Divergence":
            if args.align_type == "tail":
                qlayers[align_index].set_forward_state(stop_forward=True)
                qlayers[align_index].set_output_catch_state(state=True)
                fp_layers[align_index].set_forward_state(stop_forward=True)
                fp_layers[align_index].set_output_catch_state(state=True)
            else:
                qlayers[last_block_idx].set_forward_state(stop_forward=True)
                qlayers[last_block_idx].set_output_catch_state(state=True)
                fp_layers[last_block_idx].set_forward_state(stop_forward=True)
                fp_layers[last_block_idx].set_output_catch_state(state=True)

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
        # loss_scaler = utils.NativeScalerWithGradNormCount(use_amp=amp_enabled)
        loss_scaler= torch.amp.GradScaler(device=args.dev)
        trainable_number = trainable_parameters_num(selected_layers)
        print(f"trainable parameter number: {trainable_number/1e6}M")
        # 参数内存（float32），每个参数 4 字节
        # 混合精度 AMP 梯度内存 (bfloat16)，每个参数 2 字节
        # AdamW 动量状态内存（float32），每个状态 4 字节，共两个状态 (m, v)
        print(f"estimated memory usage: {trainable_number*(4+2+2*4)/(1024**3)}G")
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
                                    dtype=args.dtype if amp_enabled else torch.float32):
                    
                    # debug 检查grad
                    # if trainable_layer_idx_list in ([0], [1], [8],[19]):
                    #     examine_parameters_grad(model,logger)
                    
                    with torch.no_grad():
                        try:
                            target_logits = target_model(input_data.to("cuda:1")).logits
                        except StopException:
                            pass
                        except Exception as e:
                            raise e
                        if not args.loss_func == "KL-Divergence" :
                            if  args.align_type == "tail":
                                target_output = fp_layers[align_index].outs[0]
                            else:
                                target_output = fp_layers[last_block_idx].outs[0]

                    try:
                        student_logits = model(input_data.to(args.dev)).logits
                    except StopException:
                        pass
                    except Exception as e:
                        raise e
                    if not args.loss_func == "KL-Divergence" :
                        if  args.align_type == "tail":
                            output = qlayers[align_index].outs[0] # outs[0] is out tensor
                        else:
                            output = qlayers[last_block_idx].outs[0]

                # 获取当前层的自定义损失 mse
                    if args.loss_func == "KL-Divergence":
                        def kl_div(student_hiddens, teacher_hiddens):
                            C = student_hiddens.shape[-1]  # num classes
                            return F.kl_div(
                                input=F.log_softmax(student_hiddens.view(-1, C), dim=-1),
                                target=F.log_softmax(teacher_hiddens.view(-1, C), dim=-1),
                                log_target=True,
                                reduction="batchmean",
                            )
                        # loss = F.kl_div(F.log_softmax(student_logits, dim=-1),
                        #                 F.softmax(target_logits.to(student_logits.device), dim=-1),
                        #                 reduction='batchmean')
                        loss = kl_div(student_logits,
                                      target_logits.to(student_logits.device))/args.train_size
                    else:
                        loss = loss_func(output, target_output.to(args.dev,dtype=torch.float32))
                    if args.get("constrain2raw",False):
                        trainable_layer_idx_list
                        for idx in trainable_layer_idx_list:
                            name = f"model.layers.{idx}"
                            module = dict(model.named_modules()).get(name,None)
                            assert module is not None
                            target_module = dict(target_model.named_modules()).get(name,None)
                            assert target_module is not None
                            if isinstance(module, Catcher):
                                module = module.module
                                target_module = target_module.module
                            for sub_name, sub_module in module.named_modules():
                                if isinstance(sub_module, int_linear_fake.QuantLinear):
                                    w1,b1 = sub_module.get_quant_weight_bias()
                                    target_sub_module = dict(target_module.named_modules()).get(sub_name,None)
                                    assert target_sub_module is not None
                                    w2 = target_sub_module.weight.detach()
                                    if b1 is not None:
                                        b2 = target_sub_module.bias.detach().view(-1)
                                        b1 = b1.view(-1)
                                        loss += F.cosine_similarity(b1,b2.to(b1.device),dim=0)
                                    w1 = w1.view(-1)
                                    w2 = w2.view(-1)
                                    loss += F.cosine_similarity(w1,w2.to(w1.device),dim=0)

                                
                    # 看看dlc与akl
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
                # norm = loss_scaler(loss,
                #         optimizer,
                #         clip_grad=args.clip_grad,
                #         parameters=trainable_parameters(selected_layers)).cpu()
                loss_scaler.scale(loss).backward() if amp_enabled else loss.backward()
                # debug 检查grad
                if None and trainable_layer_idx_list in ([1], [8],[19]):
                    examine_parameters_grad(model,logger)
                if amp_enabled: loss_scaler.unscale_(optimizer)
                if args.clip_grad > 0:
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
            if args.loss_func == "KL-Divergence":
                if amp_enabled:
                    loss_scaler.step(optimizer)
                    loss_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # step 6.5: calculate validation loss
            val_loss_list = []
            final_val_list = []
            dataloader = DataLoader(val_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False
                                    )
            if not args.loss_func == "KL-Divergence":
                qlayers[align_index].set_forward_state(stop_forward=False)
                fp_layers[align_index].set_forward_state(stop_forward=False)
            for index, input_data in enumerate(dataloader):  
                # obtain output of quantization model
                with torch.no_grad():
                    with torch.autocast(device_type=args.dev,
                                        enabled=amp_enabled,
                                        dtype=args.dtype if amp_enabled else torch.float32):
                        try:
                            student_logits = model(input_data.to(args.dev)).logits
                        except StopException:
                            pass
                        except Exception as e:
                            raise e
                        if not args.loss_func == "KL-Divergence":
                            if args.align_type == "tail":
                                output = qlayers[align_index].outs[0].to(dtype=torch.float32,device=args.dev) # outs[0] is out tensor
                            else:
                                output = qlayers[last_block_idx].outs[0].to(dtype=torch.float32,device=args.dev)

                            # final_output = qlayers[last_block_idx].outs[0]
                        try:
                            teacher_logits = target_model(input_data.to("cuda:1")).logits
                        except StopException:
                            pass
                        except Exception as e:
                            raise e
                        if not args.loss_func == "KL-Divergence":
                            if args.align_type == "tail":
                                target_output = fp_layers[align_index].outs[0].to(dtype=torch.float32,device=args.dev)
                            else:
                                target_output = fp_layers[last_block_idx].outs[0].to(dtype=torch.float32,device=args.dev)

                            # final_target_output = fp_layers[last_block_idx].outs[0].to(dtype=torch.float32,device=args.dev)
                        if args.loss_func == "KL-Divergence":
                            loss = F.kl_div(F.log_softmax(student_logits, dim=-1),
                                            F.softmax(teacher_logits.to(student_logits.device), dim=-1),
                                            reduction='batchmean')
                        else:
                            loss = loss_func(output, target_output)
                            # final_loss = loss_func(final_output, final_target_output)
                val_loss_list.append(loss.cpu())
                # if not args.loss_func == "KL-Divergence":
                #     final_val_list.append(final_loss.cpu())
            if not args.loss_func and  args.align_type == "tail":
                qlayers[align_index].set_forward_state(stop_forward=True)
                fp_layers[align_index].set_forward_state(stop_forward=True)

            train_mean_num = min(len(loss_list),64) 
            # calculate the average training loss of last train_mean_num samples
            loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
            val_loss_mean = torch.stack(val_loss_list).mean()
            norm_mean = torch.stack(norm_list).mean()
            logger.info(f"blocks {trainable_layer_idx_list} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(args.dev) / 1024**2} time {time.time()-start_time} ")
            # if not args.loss_func == "KL-Divergence":
            #     logger.info(f"blocks {trainable_layer_idx_list} epoch {epoch} final_loss:{torch.stack(final_val_list).mean()}")
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
                                                not module.bias is None,
                                                clamp_input= args.get("clamp_input",False))
                    q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                    set_op_by_name(qlayer, name, q_linear)       
                    logger.info(f"pack quantized {name} finished")
                    del module
            if amp_enabled:
                qlayer.to(dtype=args.dtype)
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
    model.to(args.dev)
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
    loss_func = get_loss_func(args.loss_func) if not args.loss_func == "KL-Divergence" else None
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
    dtype = torch.float16 if amp_enabled else torch.float32
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

