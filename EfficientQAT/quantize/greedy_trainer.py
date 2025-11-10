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
from typing import List, Tuple, Dict, Union, Callable, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json

import numpy as np
import torch
import torch.nn as nn
import torch.amp
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from transformers.modeling_utils import PreTrainedModel
import logging
import bitsandbytes as bnb

from .. import utils
from . import int_linear_fake, int_linear_real
from EfficientQAT.core.quantization import (
    build_real_quant_linear,
    export_scale_tensor,
    export_zero_tensor,
)
from .utils import (
    quant_parameters,weight_parameters,trainable_parameters,
    set_quant_state,quant_inplace,set_quant_parameters,
    set_weight_parameters,trainable_parameters_num,get_named_linears,set_op_by_name,
    Catcher,StopException,MultiBlock,sub_space_clean
    )
from ..datautils_block import BlockTrainDataset,OptimBlockTrainDataset,LazyLoadDataset,generate_block_train_data,LazyLoadDatasetV2
from ..loss_utils import get_loss_func

# 添加对新可视化模块的导入
try:
    from .visualization import VisualizationRecorder
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VisualizationRecorder = None
    VISUALIZATION_AVAILABLE = False

# 添加对调试工具的导入
try:
    from .debug_utils import check_loss_anomaly, save_debug_info, collect_gradients, debug_model_state, print_tensor_anomalies
    DEBUG_UTILS_AVAILABLE = True
    # 检查是否启用了调试模式
    QAT_DEBUG = os.environ.get('QAT_DEBUG', '').lower() in ('1', 'true', 'yes')
except ImportError:
    DEBUG_UTILS_AVAILABLE = False
    QAT_DEBUG = False

from torch.optim.lr_scheduler import CosineAnnealingLR


amp_enabled = True
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

class CosineAnnealingScheduler:
    def __init__(self, max_value, min_value=0, total_steps=100, ascend=False):
        """
        余弦退火调度器：支持从 max_value -> min_value 或 0 -> max_value
        :param max_value: 最大值（目标值或起始值）
        :param min_value: 最小值（仅在 ascend=False 时生效）
        :param total_steps: 总步数
        :param ascend: 是否从0增加到 max_value，True 表示从0增加，否则从 max_value 减少到 min_value
        """
        self.max_value = max_value
        self.min_value = min_value
        self.total_steps = total_steps
        self.current_step = 0
        self.ascend = ascend

    def step(self):
        """
        执行一步退火，返回当前步的值
        """
        if self.current_step > self.total_steps:
            self.current_step = self.total_steps

        if self.ascend:  # 从 0 增加到 max_value
            value = self.max_value * 0.5 * (1 - math.cos(math.pi * self.current_step / self.total_steps))
        else:  # 从 max_value 减少到 min_value
            value = self.min_value + 0.5 * (self.max_value - self.min_value) * (
                1 + math.cos(math.pi * self.current_step / self.total_steps)
            )

        self.current_step += 1
        return value

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
                        vis_recorder=None,  # 添加可视化记录器参数
                        logger=None,
                        config: Dict[str, Any]=None):
    train_params = config.get('train_param_settings', {})
    hyper_params = config.get('hyperparam_settings', {})

    total_training_iteration = train_params['epochs'] * train_params['train_size'] / train_params['batch_size']
    layer_idx_set = set(trainable_layer_idx_list)
    step = 0
    assert float(train_params['quant_lr']) > 0 or float(train_params['weight_lr']) > 0

    # 使用amp仍然需要权重为float32
    with torch.no_grad():
        model.model.layers = nn.ModuleList(
                [qlayer.to(train_params['dev'], dtype=torch.float32)
                    if index in layer_idx_set 
                    else qlayer.half() for index,
                        qlayer in enumerate(model.model.layers)])
        
    qlayers = model.model.layers
    selected_layers = [qlayers[i] for i in trainable_layer_idx_list]
    selected_layers = nn.ModuleList(selected_layers)

    for name, param in model.named_parameters():
        param.requires_grad = False
    for layer_idx in trainable_layer_idx_list:
        param_groups = []
        param_group_index = 0
        qlayer = model.model.layers[layer_idx]
        set_quant_state(qlayer,True)
        with torch.no_grad():
            qlayer.float()
        if float(train_params['quant_lr']) > 0:
            set_quant_parameters(qlayer,True)
            param_groups.append({"params":quant_parameters(qlayer),
                                "lr":train_params['quant_lr']})
            empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)],
                                                lr=train_params['quant_lr'])
            quant_scheduler = CosineAnnealingLR(empty_optimizer_1,
                                        T_max=total_training_iteration,
                                        eta_min=train_params['quant_lr']/hyper_params['min_lr_factor'])
            quant_index = param_group_index
            param_group_index += 1
        else:
            set_quant_parameters(qlayer,False)
            
        if float(train_params['weight_lr']) > 0:
            set_weight_parameters(qlayer,True)
            param_groups.append({"params":weight_parameters(qlayer),
                                "lr":train_params['weight_lr']})
            empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)],
                                                lr=train_params['weight_lr'])
            weight_scheduler = CosineAnnealingLR(empty_optimizer_2,
                                            T_max=total_training_iteration,
                                            eta_min=train_params['weight_lr']/hyper_params['min_lr_factor'])
            weight_index = param_group_index
            param_group_index += 1
        else:
            set_weight_parameters(qlayer,False)
        
        optimizer =torch.optim.AdamW(param_groups,
                                    weight_decay=train_params['wd'],
                                    foreach=True)
        
        loss_scaler= torch.amp.GradScaler(device=train_params['dev'])
        trainable_number = trainable_parameters_num(selected_layers)
        print(f"trainable parameter number: {trainable_number/1e6}M")
        best_val_loss = 1e6
        early_stop_flag = 0

        if hyper_params.get("gradual_quant",False) or config.get("interpolate",False):
            gradual_factor = hyper_params.get("gradual_factor",2.0)
            class GradualWarmupScheduler:
                def __init__(self,
                              linear_list:List[int_linear_fake.QuantLinear],
                              total_iteration:int,
                              gradual_factor:float=2.0
                              ):
                    self.linear_list = linear_list
                    self.total_iteration = total_iteration
                    self.iteration = 0
                    self.gradual_factor = gradual_factor
                    self.update()
                def update(self):
                    self.iteration += 1
                    for linear in self.linear_list:
                        if self.iteration < (self.total_iteration/self.gradual_factor):
                            ratio = self.iteration/(self.total_iteration/self.gradual_factor)
                            if hyper_params.get("gradual_quant",False):
                                linear.update_position_ratio(ratio)
                            if config.get("interpolate", False):
                                linear.update_interpolate_ratio(1-ratio)
                        else:
                            linear.update_position_ratio(1.0)
                            if config.get("interpolate", False):
                                linear.update_interpolate_ratio(0)
            q_linear_list = []
            for i in trainable_layer_idx_list:
                for n,m in qlayers[i].named_modules():
                    if isinstance(m, int_linear_fake.QuantLinear):
                        q_linear_list.append(m)
            graualWarmupScheduler = GradualWarmupScheduler(
                q_linear_list,
                total_training_iteration,
                gradual_factor
            )
        if hyper_params.get("dampen_loss"):
            dampen_loss_weight = hyper_params.get("dampen_loss_weight",0.01)
            dampen_loss_weight_scheduler = CosineAnnealingScheduler(max_value=dampen_loss_weight,
                                                                    min_value=0,
                                                                     total_steps=total_training_iteration,
                                                                      ascend=True)
        # step 6.3: training loop
        position_ids = torch.arange(train_params['training_seqlen'], dtype=torch.long, device=train_params['dev'])
        position_ids = position_ids.unsqueeze(0).expand(train_params['batch_size'], -1).contiguous()
        # print(f" data size {len(train_dataset)}")

        # before training, need to prepare model?
        if hyper_params.get("swa",False):
            """
                swa
                swa_start: 开始使用swa的epoch数
                swa_factor: 平均权重系数 默认建议为0.9
                swa_lr: 学习率 我需要用到这个吗？要不还是让其自动管理吧
            """
            swa_model = torch.optim.swa_utils.AveragedModel(selected_layers,
                                                                device=train_params['dev'],
                        avg_fn=torch.optim.swa_utils.get_ema_avg_fn(hyper_params.get("swa_factor",0.9)))
            swa_start = hyper_params.get("swa_start") # 需要手动指定
            # 需要scheduler吗，暂时存疑
            # swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, 
            #                                             anneal_strategy="cos", # “cos” or “linear”
            #                                             anneal_epochs=10, # default 
            #                                             swa_lr=args.swa_lr)
        # try torch.compile
        # selected_layers = torch.compile(selected_layers,mode="reduce-overhead")
        for epoch in range(train_params['epochs']):
            loss_list = []
            norm_list = []
            start_time = time.time()
            # used for debug
            # torch.autograd.set_detect_anomaly(True)
            dataloader = DataLoader(train_dataset,
                                    batch_size=train_params['batch_size'],
                                    # num_workers=1,
                                    pin_memory=True,
                                    # prefetch_factor=32,  
                                    shuffle=False,
                                    )
            # step 6.4: training                   
            for index, input_data in enumerate(dataloader):
                optimizer.zero_grad()
                with torch.autocast(device_type="cuda",
                                    enabled=amp_enabled,
                                    ):
                    inp,target = input_data
                    # 强制要求激活值是float32
                    inp = inp.to(train_params['dev'],dtype=train_params['dtype'])
                    trg = target.to(train_params['dev'],dtype=torch.float32)
                    hidden_state = inp
                    # estimate memory usage unit GB
                    # inp_memory = inp.numel() * inp.element_size() / 1024**3
                    # hidden_state_memory = hidden_state.numel() * hidden_state.element_size() / 1024**3
                    # target_memory = trg.numel() * trg.element_size() / 1024**3
                    # logger.info(f"index {index} input_memory {inp_memory:.4f} hidden_state_memory {hidden_state_memory:.4f} target_memory {target_memory:.4f} ")
                    # import pdb;pdb.set_trace()
                    for layer_idx in trainable_layer_idx_list:
                        layer_outputs = qlayers[layer_idx](
                            hidden_states=hidden_state,
                            attention_mask=attention_mask.float(),
                            position_ids=position_ids,
                            position_embeddings=(position_embeddings[0].float(),position_embeddings[1].float())
                        )
                        assert isinstance(layer_outputs, tuple)
                        hidden_state = layer_outputs[0]
                    loss = loss_func(hidden_state, trg)
                    # insert part
                    # if index == 32:
                    #     tmp = {
                    #         "hidden_state":inp,
                    #         # "attention_mask":attention_mask,
                    #         # "position_embeddings":position_embeddings,
                    #     }
                    #     print(f"layers {trainable_layer_idx_list} input_data {tmp} output {hidden_state} target {target} ")
                    # print(f"index {index} loss {loss}")

                    # insert dampen loss
                    if hyper_params.get("dampen_loss",False):
                        dampen_loss = torch.zeros_like(loss).to(loss.device)
                        for layer_idx in trainable_layer_idx_list:
                            qlayer = qlayers[layer_idx]
                            for n,m in qlayer.named_modules():
                                if isinstance(m, int_linear_fake.QuantLinear):
                                    dampen_loss += m.get_dampen_loss()
                        # print(f"dampen_loss {dampen_loss}")
                        loss += dampen_loss * dampen_loss_weight_scheduler.step()

                if not math.isfinite(loss.item()) or loss.item()==0:
                    logger.info("Loss is NAN, stopping training")
                    # pdb.set_trace()
                loss_cpu = loss.data.cpu()
                
                # 检查loss异常，仅在QAT_DEBUG环境变量开启时启用
                if QAT_DEBUG and DEBUG_UTILS_AVAILABLE:
                    loss_threshold = float(config.get('debug_loss_threshold', 1000.0))
                    if check_loss_anomaly(loss_cpu, threshold=loss_threshold):
                        logger.warning(f"Abnormal loss detected: {loss_cpu.item()} at step {step}, epoch {epoch}")
                        logger.warning(f"Trainable layers: {trainable_layer_idx_list}")
                        
                        # 保存调试信息
                        debug_save_dir = os.path.join(config.get('output_dir', './'), 
                                                    f"debug_loss_anomaly_blk{trainable_layer_idx_list}_step{step}_epoch{epoch}")
                        
                        # 准备调试信息
                        debug_inputs = {
                            "input": inp.detach().cpu() if isinstance(inp, torch.Tensor) else inp,
                            "target": trg.detach().cpu() if isinstance(trg, torch.Tensor) else trg,
                            "attention_mask": attention_mask.detach().cpu() if isinstance(attention_mask, torch.Tensor) else attention_mask,
                            "position_ids": position_ids.detach().cpu() if isinstance(position_ids, torch.Tensor) else position_ids
                        }
                        
                        debug_outputs = {
                            "hidden_state": hidden_state.detach().cpu() if isinstance(hidden_state, torch.Tensor) else hidden_state,
                            "layer_outputs": layer_outputs
                        }
                        
                        try:
                            debug_model_state(
                                model=selected_layers,
                                inputs=debug_inputs,
                                outputs=debug_outputs,
                                loss=loss,
                                save_dir=debug_save_dir
                            )
                            logger.info(f"Debug information saved to {debug_save_dir}")
                        except Exception as e:
                            logger.error(f"Failed to save debug information: {e}")
                        
                        # 打印张量异常信息
                        print_tensor_anomalies(inp, "input", loss_threshold)
                        print_tensor_anomalies(hidden_state, "hidden_state", loss_threshold)
                        print_tensor_anomalies(trg, "target", loss_threshold)
                        
                        # 如果设置了在检测到异常时停止训练
                        if config.get('debug_stop_on_anomaly', False):
                            logger.error("Stopping training due to loss anomaly")
                            raise RuntimeError(f"Loss anomaly detected: {loss_cpu.item()}")
                    
                if config.get("log_loss"):
                    loss_recorder.record(f"blk{trainable_layer_idx_list}",
                                        step,
                                        loss_cpu.item())
                    
                # 记录训练损失到可视化工具
                if vis_recorder is not None:
                    vis_recorder.record_loss(
                        blk_id=f"blk{trainable_layer_idx_list}",
                        step=step,
                        loss=loss_cpu.item(),
                        loss_type="train_loss"
                    )
                    
                if hyper_params.get("gradual_quant",False):
                    graualWarmupScheduler.update() 
                else: 
                    pass
                loss_list.append(loss_cpu)
                if amp_enabled:
                    loss_scaler.scale(loss).backward()
                else:
                    loss.backward()
                # debug 检查grad
                # torch.cuda.synchronize()
                # torch.save(qlayers[0].mlp.gate_proj.weight.grad,
                #             f"/home/ubuntu/data/exp/proj2410/test/cache/0_weight_{index}"
                #             )
                if amp_enabled: loss_scaler.unscale_(optimizer)
                if float(train_params['clip_grad']) > 0:
                    norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(selected_layers)
                                                            , float(train_params['clip_grad'])).cpu()
                    norm_list.append(norm.data)
                # 使用子空间优化
                if hyper_params.get("sub_space_grad_clean",False):
                    sub_space_clean(selected_layers)
                
                
                # optmizer step zone
                if amp_enabled:
                    loss_scaler.step(optimizer)
                    loss_scaler.update()
                else:
                    optimizer.step()
                
                # inject swa model
                if hyper_params.get("swa",False):
                    if step> swa_start and step % hyper_params.get("swa_freq",200)==0:
                        # When update_parameters() is called for the first time 
                        # (i.e. n_averaged is 0) the parameters of model are copied to the parameters of AveragedModel.
                        #  For every subsequent call of update_parameters() the function avg_fn is used to update the parameters.
                        swa_model.update_parameters(selected_layers)
                        print("merge model")
                
                # adjust lr 这个是需要的吗？
                if float(train_params['quant_lr']) > 0:
                    quant_scheduler.step()
                    optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]
                if float(train_params['weight_lr']) >0 :
                    weight_scheduler.step()
                    optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[0]
                step += 1

            # step 6.5: calculate validation loss
            with torch.no_grad():
                val_loss_list = []
                dataloader = DataLoader(val_dataset,
                                        batch_size=train_params['batch_size'],
                                        # num_workers=0,
                                        pin_memory=True,
                                        # prefetch_factor=32,  
                                        shuffle=False,
                                        )
                for index, input_data in enumerate(dataloader):  
                    # obtain output of quantization model
                    with torch.autocast(device_type="cuda",
                                    enabled=amp_enabled,
                                    dtype=train_params['dtype'] if amp_enabled else torch.float32):
                        inp,target = input_data
                        hidden_state = inp.to(train_params['dev'],dtype=train_params['dtype'])
                        trg = target.to(train_params['dev'],dtype=torch.float32)
                        for layer_idx in trainable_layer_idx_list:
                            layer_outputs = qlayers[layer_idx](
                                hidden_states=hidden_state,
                                attention_mask=attention_mask.float(),
                                position_embeddings=(position_embeddings[0].float(),position_embeddings[1].float())
                            )
                            hidden_state = layer_outputs[0]
                        loss = loss_func(hidden_state, trg)
                    val_loss_list.append(loss.cpu())

                train_mean_num = min(len(loss_list),64) 
                # calculate the average training loss of last train_mean_num samples
                loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"blocks {trainable_layer_idx_list} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} ")
                logger.info(f"quant_lr:{quant_scheduler.get_lr()[0]} weight_lr:{weight_scheduler.get_lr()[0]} norm:{norm_mean:.8f}  ")
                logger.info(f"max memory_allocated {torch.cuda.max_memory_allocated(train_params['dev']) / 1024**2} time {time.time()-start_time} ")
                
                # 记录验证损失到可视化工具
                if vis_recorder is not None:
                    vis_recorder.record_loss(
                        blk_id=f"blk{trainable_layer_idx_list}",
                        step=epoch,
                        loss=val_loss_mean.item(),
                        loss_type="val_loss"
                    )
                    # 同时记录学习率
                    vis_recorder.record_scalar(
                        name=f"learning_rate/quant_lr_blk{trainable_layer_idx_list}",
                        value=quant_scheduler.get_lr()[0],
                        step=epoch
                    )
                    vis_recorder.record_scalar(
                        name=f"learning_rate/weight_lr_blk{trainable_layer_idx_list}",
                        value=weight_scheduler.get_lr()[0],
                        step=epoch
                    )
                    vis_recorder.record_scalar(
                        name=f"gradient_norm/blk{trainable_layer_idx_list}",
                        value=norm_mean.item(),
                        step=epoch
                    )
                
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                else:
                    early_stop_flag += 1
                    if int(config.get('early_stop', 0)) > 0 and early_stop_flag >= int(config.get('early_stop', 0)):
                        break
            
        optimizer.zero_grad()
        del optimizer
        # step 7: pack quantized weights into low-bits format, note that this process is slow on poor CPU or busy CPU
        
    torch.cuda.empty_cache()
    gc.collect()


@timer
def train_units_layers_with_catcher(model: PreTrainedModel,
                        trainable_layer_idx_list: List[int],
                        loss_func: Callable,
                        train_dataset: LazyLoadDataset,
                        val_dataset: LazyLoadDataset,
                        target_model: PreTrainedModel,
                        loss_recorder,
                        vis_recorder=None,  # 添加可视化记录器参数
                        logger=None,
                        config: Dict[str, Any]=None):
    # 解冻当前层，并冻结其它层
    train_params = config.get('train_param_settings', {})
    hyper_params = config.get('hyperparam_settings', {})

    model.to(train_params['dev'])
    total_training_iteration = train_params['epochs'] * train_params['train_size'] / train_params['batch_size']
    layer_idx_set = set(trainable_layer_idx_list)
    step = 0
    param_groups = []
    param_group_index = 0
    assert float(train_params['quant_lr']) > 0 or float(train_params['weight_lr']) > 0

    # 使用amp仍然需要权重为float32
    with torch.no_grad():
        model.model.layers = nn.ModuleList(
                [qlayer.to(train_params['dev'], dtype=torch.float32)
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
    # if config.get("loss_func") == "KL-Divergence":
    #     qlayer_idxs = []
    #     fp_layer_idxs = []
    # else:
    #     qlayer_idxs = [align_index,last_block_idx]
    #     fp_layer_idxs = [align_index,last_block_idx]
    
    start_block = trainable_layer_idx_list[0]
    qlayer_idxs = [align_index,start_block]
    fp_layer_idxs = [align_index,start_block]

    with CatcherManager(qlayers, qlayer_idxs),CatcherManager(fp_layers, fp_layer_idxs):
        if not config.get("loss_func") == "KL-Divergence":
            if config.get("align_type") == "tail":
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
            if float(train_params['quant_lr']) > 0:
                set_quant_parameters(qlayer,True)
                param_groups.append({"params":quant_parameters(qlayer),
                                    "lr":train_params['quant_lr']})
                empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)],
                                                    lr=train_params['quant_lr'])
                quant_scheduler = CosineAnnealingLR(empty_optimizer_1,
                                            T_max=total_training_iteration,
                                            eta_min=train_params['quant_lr']/train_params['min_lr_factor'])
                quant_index = param_group_index
                param_group_index += 1
            else:
                set_quant_parameters(qlayer,False)
                
            if float(train_params['weight_lr']) > 0:
                set_weight_parameters(qlayer,True)
                param_groups.append({"params":weight_parameters(qlayer),
                                    "lr":train_params['weight_lr']})
                empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)],
                                                    lr=train_params['weight_lr'])
                weight_scheduler = CosineAnnealingLR(empty_optimizer_2,
                                                T_max=total_training_iteration,
                                                eta_min=train_params['weight_lr']/train_params['min_lr_factor'])
                weight_index = param_group_index
                param_group_index += 1
            else:
                set_weight_parameters(qlayer,False)
        
        if config.get("loss_func") == "AFFINE_MSE":
            print("using AFFINE_MSE loss function")
            loss_func.reinitialize_A()
            loss_func = loss_func.to(train_params['dev'])
            param_groups.append({"params":loss_func.parameters(),"lr":train_params['weight_lr']})
        optimizer =torch.optim.AdamW(param_groups,
                                    weight_decay=train_params['wd'],
                                    foreach=True)
        qlayers = model.model.layers
        # loss_scaler = utils.NativeScalerWithGradNormCount(use_amp=amp_enabled)
        loss_scaler= torch.amp.GradScaler(device=train_params['dev'])
        trainable_number = trainable_parameters_num(selected_layers)
        print(f"trainable parameter number: {trainable_number/1e6}M")
        # 参数内存（float32），每个参数 4 字节
        # 混合精度 AMP 梯度内存 (bfloat16)，每个参数 2 字节
        # AdamW 动量状态内存（float32），每个状态 4 字节，共两个状态 (m, v)
        print(f"estimated memory usage: {trainable_number*(4+2+2*4)/(1024**3)}G")
        print(f"estimated data usage: {train_params['batch_size']*train_params['training_seqlen']*model.config.hidden_size*4/(1024**3)}G")
        best_val_loss = 1e6
        early_stop_flag = 0
        if hyper_params.get("gradual_quant",False) or config.get("interpolate",False):
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
                        if self.iteration < (self.total_iteration/2.0):
                            ratio = self.iteration/(self.total_iteration/2.0)
                            if hyper_params.get("gradual_quant",False):
                                linear.update_position_ratio(ratio)
                            if config.get("interpolate", False):
                                linear.update_interpolate_ratio(1-ratio)
                        else:
                            linear.update_position_ratio(1.0)
                            if config.get("interpolate", False):
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
        # print(f" data size {len(train_dataset)}")
        for epoch in range(train_params['epochs']):
            loss_list = []
            norm_list = []
            start_time = time.time()
            # used for debug
            # torch.autograd.set_detect_anomaly(True)
            dataloader = DataLoader(train_dataset,
                                    batch_size=train_params['batch_size'],
                                    num_workers=0,
                                    pin_memory=True,
                                    # prefetch_factor=32,  
                                    shuffle=True
                                    )
            # step 6.4: training                   
            for index, input_data in enumerate(dataloader):
                with torch.autocast(device_type="cuda",
                                    enabled=amp_enabled,
                                    dtype=train_params['dtype'] if amp_enabled else torch.float32):
                    inp = input_data
                    assert inp.dim() == 2, "input dim should be 2"
                    try:
                        output = model(inp.to(train_params['dev']))
                    except StopException as e:
                            pass
                    output = qlayers[align_index].outs[0] # outs[0] is out tensor
                    with torch.no_grad():
                        try:
                            target_output = target_model(inp.to(train_params['cuda'][-1]))[0]
                        except StopException as e:
                            pass
                        target_output = fp_layers[align_index].outs[0]
                    # if index == 0:
                    #     inp = qlayers[align_index].inps
                    #     print(f"input {inp} output {output} target_output {target_output}")
                    loss = loss_func(output, target_output.to(train_params['dev'],dtype=torch.float32))

                    if hyper_params.get("dampen_loss",False):
                        for idx in trainable_layer_idx_list:
                            for n,m in qlayers[idx].named_modules():
                                if isinstance(m, int_linear_fake.QuantLinear):
                                    loss += m.get_dampen_loss()*0.001
                          
                if not math.isfinite(loss.item()) or loss.item()==0:
                    logger.info("Loss is NAN, stopping training")
                    pdb.set_trace()
                
                # 检查loss异常，仅在QAT_DEBUG环境变量开启时启用
                if QAT_DEBUG and DEBUG_UTILS_AVAILABLE:
                    loss_threshold = float(config.get('debug_loss_threshold', 1000.0))
                    if check_loss_anomaly(loss, threshold=loss_threshold):
                        logger.warning(f"Abnormal loss detected: {loss.item()} at step {step}, epoch {epoch}")
                        logger.warning(f"Trainable layers: {trainable_layer_idx_list}")
                        
                        # 保存调试信息
                        debug_save_dir = os.path.join(config.get('output_dir', './'), 
                                                    f"debug_loss_anomaly_blk{trainable_layer_idx_list}_step{step}_epoch{epoch}")
                        
                        # 准备调试信息
                        debug_inputs = {
                            "input": inp.detach().cpu() if isinstance(inp, torch.Tensor) else inp,
                            "output": output.detach().cpu() if isinstance(output, torch.Tensor) else output,
                            "target_output": target_output.detach().cpu() if isinstance(target_output, torch.Tensor) else target_output
                        }
                        
                        debug_outputs = {
                            "loss": loss.detach().cpu() if isinstance(loss, torch.Tensor) else loss
                        }
                        
                        try:
                            debug_model_state(
                                model=selected_layers,
                                inputs=debug_inputs,
                                outputs=debug_outputs,
                                loss=loss,
                                save_dir=debug_save_dir
                            )
                            logger.info(f"Debug information saved to {debug_save_dir}")
                        except Exception as e:
                            logger.error(f"Failed to save debug information: {e}")
                        
                        # 打印张量异常信息
                        print_tensor_anomalies(inp, "input", loss_threshold)
                        print_tensor_anomalies(output, "output", loss_threshold)
                        print_tensor_anomalies(target_output, "target_output", loss_threshold)
                        
                        # 如果设置了在检测到异常时停止训练
                        if config.get('debug_stop_on_anomaly', False):
                            logger.error("Stopping training due to loss anomaly")
                            raise RuntimeError(f"Loss anomaly detected: {loss.item()}")
                
                if config.get("log_loss"):
                    loss_recorder.record(f"blk{trainable_layer_idx_list}",
                                        step,
                                        loss.data.cpu().item())
                
                # 记录训练损失到可视化工具
                if vis_recorder is not None:
                    vis_recorder.record_loss(
                        blk_id=f"blk{trainable_layer_idx_list}",
                        step=step,
                        loss=loss.data.cpu().item(),
                        loss_type="train_loss"
                    )
                    
                graualWarmupScheduler.update() if hyper_params.get("gradual_quant",False) else None
                loss_list.append(loss.data.cpu())
                # 反向传播和优化
                if not config.get("loss_func") == "KL-Divergence":
                    optimizer.zero_grad()
                loss_scaler.scale(loss).backward() if amp_enabled else loss.backward()
                # debug 检查grad
                if amp_enabled: loss_scaler.unscale_(optimizer)
                if float(train_params['clip_grad']) > 0:
                        # print(f"clip grad at {args.clip_grad}")
                        norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(selected_layers), float(train_params['clip_grad'])).cpu()
                # 临时的检查
                # with torch.no_grad():
                #     idx =torch.tensor([[0],[0]])
                #     q_proj_weight = qlayers[align_index].module.self_attn.q_proj.weight
                #     quantizer = qlayers[align_index].module.self_attn.q_proj.weight_quantizer
                #     q_proj_weight_simu = quantizer(qlayers[align_index].module.self_attn.q_proj.weight)
                #     scale = quantizer.scale
                #     # 打开文件写入
                #     with open(f"/home/ubuntu/data/exp/proj2410/test/logs","a+") as f:
                #         f.write(f"{scale}\n")
                        # f.write(f"{q_proj_weight[0,:300]},{q_proj_weight_simu[0:,300]}\n")
                # 更新
                if not config.get("loss_func") == "KL-Divergence":
                    if amp_enabled:
                        loss_scaler.step(optimizer)
                        loss_scaler.update()
                    else:
                        optimizer.step()
                norm_list.append(norm.data)
                
                # adjust lr
                if float(train_params['quant_lr']) > 0:
                    quant_scheduler.step()
                    optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]
                if float(train_params['weight_lr']) >0 :
                    weight_scheduler.step()
                    optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[0]
                step += 1


            # step 6.5: calculate validation loss
            val_loss_list = []
            dataloader = DataLoader(val_dataset,
                                    batch_size=train_params['batch_size'],
                                    num_workers=0,
                                    pin_memory=True,
                                    # prefetch_factor=32,  
                                    shuffle=True
                                    )
            # if not config.get("loss_func") == "KL-Divergence":
            #     qlayers[align_index].set_forward_state(stop_forward=False)
            for index, input_data in enumerate(dataloader):  
                # obtain output of quantization model
                with torch.no_grad():
                    with torch.autocast(device_type="cuda",
                                    enabled=amp_enabled,
                                    dtype=train_params['dtype'] if amp_enabled else torch.float32):
                        inp = input_data
                        try:
                            output = model(inp.to(train_params['dev']))
                        except StopException:
                            pass
                        output = qlayers[align_index].outs[0] # outs[0] is out tensor
                        try:
                            target_output = target_model(inp.to(train_params['cuda'][-1]))[0]
                        except StopException:
                            pass
                        target_output = fp_layers[align_index].outs[0].to(torch.float32)
                        loss = loss_func(output, target_output.to(train_params['dev'],dtype=torch.float32))
                val_loss_list.append(loss.cpu())


                # if not config.get("loss_func") == "KL-Divergence":
                #     final_val_list.append(final_loss.cpu())
            if not config.get("loss_func") and  config.get("align_type") == "tail":
                qlayers[align_index].set_forward_state(stop_forward=True)

            train_mean_num = min(len(loss_list),64) 
            # calculate the average training loss of last train_mean_num samples
            loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
            val_loss_mean = torch.stack(val_loss_list).mean()
            norm_mean = torch.stack(norm_list).mean()
            logger.info(f"blocks {trainable_layer_idx_list} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(train_params['dev']) / 1024**2} time {time.time()-start_time} ")
            
            # 记录验证损失到可视化工具
            if vis_recorder is not None:
                vis_recorder.record_loss(
                    blk_id=f"blk{trainable_layer_idx_list}",
                    step=epoch,
                    loss=val_loss_mean.item(),
                    loss_type="val_loss"
                )
                # 同时记录学习率
                vis_recorder.record_scalar(
                    name=f"learning_rate/quant_lr_blk{trainable_layer_idx_list}",
                    value=quant_scheduler.get_lr()[0],
                    step=epoch
                )
                vis_recorder.record_scalar(
                    name=f"learning_rate/weight_lr_blk{trainable_layer_idx_list}",
                    value=weight_scheduler.get_lr()[0],
                    step=epoch
                )
                vis_recorder.record_scalar(
                    name=f"gradient_norm/blk{trainable_layer_idx_list}",
                    value=norm_mean.item(),
                    step=epoch
                )
            
            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
            else:
                early_stop_flag += 1
                if int(config.get('early_stop', 0)) > 0 and early_stop_flag >= int(config.get('early_stop', 0)):
                    break


        optimizer.zero_grad()
        del optimizer
        # step 7: pack quantized weights into low-bits format, note that this process is slow on poor CPU or busy CPU
        
    torch.cuda.empty_cache()


def trans_quant_block(qlayer:nn.Module,hyper_params):
    for name, module in qlayer.named_modules():
        if isinstance(module,torch.nn.Linear):
            quantlinear = int_linear_fake.QuantLinear(module,
                                hyper_params['wbits'],
                                hyper_params['group_size'],
                                hyper_params)
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
                    config: Dict[str, Any]):

    train_params = config.get('train_param_settings', {})
    hyper_params = config.get('hyperparam_settings', {})

    if train_params.get("with_catcher"):
        target_model = copy.deepcopy(model)
        target_model.to(train_params['cuda'][-1])

    shedule_list= []
    # 暂时调度的内容是直接平移一个
    # offset = 1
    if train_params.get("quant_shedule_type") == "full":
        for i in range(len(model.model.layers)):
            model.model.layers[i] = trans_quant_block(qlayer=model.model.layers[i],
                                                      hyper_params=hyper_params)
    else: is_quant_layer = [False]*len(model.model.layers)
    if train_params.get("train_shedule_type") == "start2end":
        num_layers = len(model.model.layers)
        for start in range(0,num_layers,hyper_params['slide_step']):
            end = min(start + hyper_params['crossblock_window_size'], num_layers)
            shedule_list.append(list(range(start, end)))
    elif train_params.get("train_shedule_type") == "end2start":
        num_layers = len(model.model.layers)
        for end in range(num_layers, 0, -1*hyper_params['slide_step']):
            start = max(end - hyper_params['crossblock_window_size'], 0)
            shedule_list.append(list(range(start, end)))
    
    logger.info(f"use loss func {train_params['loss_func']}")

    loss_func = get_loss_func(train_params['loss_func'])
    loss_recorder = utils.BlockLossRecorder(file_path=config['log_loss'],)
    
    # 初始化可视化记录器
    vis_recorder = None
    if VISUALIZATION_AVAILABLE and config.get('enable_visualization', False):
        vis_type = config.get('visualization_type', 'tensorboard')  # 'tensorboard' 或 'wandb'
        if vis_type == 'tensorboard':
            log_dir = config.get('tensorboard_log_dir', './tensorboard_logs')
            vis_recorder = VisualizationRecorder(
                visualization_type='tensorboard',
                log_dir=log_dir,
                experiment_name=config.get('experiment_name', 'efficientqat')
            )
        elif vis_type == 'wandb':
            project_name = config.get('wandb_project', 'efficientqat')
            experiment_name = config.get('experiment_name', 'experiment')
            vis_recorder = VisualizationRecorder(
                visualization_type='wandb',
                project_name=project_name,
                experiment_name=experiment_name
            )
    
    for train_layer_window in shedule_list:
        if not train_params.get("quant_shedule_type") == "full" :
            for layer_idx in train_layer_window:
                if not is_quant_layer[layer_idx]:
                    is_quant_layer[layer_idx] = True
                    # if train_params.get("keep_fp_weight"):
                    #     fp_layer = copy.deepcopy(model.model.layers[layer_idx])
                    model.model.layers[layer_idx] = trans_quant_block(
                                            qlayer=model.model.layers[layer_idx],
                                            hyper_params=hyper_params)
        if train_params['epochs'] > 0 :
            logger.info(f"train blocks {train_layer_window}")
            # assert attention_mask is not None , "attention_mask is None"
            skip_flag=True
            skip_layers = train_params.get("skip_layers",[])
            if len(skip_layers) > 0:
                for layer_idx in train_layer_window:
                    if layer_idx not in skip_layers:
                        skip_flag=False
                        break
            else:
                skip_flag=False
            if  not skip_flag :
                if not train_params.get("with_catcher"):
                    train_units_layers(model,
                            trainable_layer_idx_list=train_layer_window,
                            loss_func=loss_func,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            attention_mask=attention_mask,
                            position_embeddings=position_embeddings,
                            loss_recorder=loss_recorder,
                            vis_recorder=vis_recorder,  # 传递可视化记录器
                            logger=logger,
                            config=config)
                else: 
                    train_units_layers_with_catcher(model,
                        trainable_layer_idx_list=train_layer_window,
                        loss_func=loss_func,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        target_model=target_model,
                        loss_recorder=loss_recorder,
                        vis_recorder=vis_recorder,  # 传递可视化记录器
                        logger=logger,
                        config=config)
            selected_layers= nn.ModuleList([model.model.layers[i] for i in train_layer_window])
        quant_inplace(selected_layers)
        # print(f"q_proj",selected_layers[0].self_attn.q_proj.weight)
        # print(f"q_proj scale",selected_layers[0].self_attn.q_proj.weight_quantizer.scale)

        with torch.no_grad():
            for layer_idx in train_layer_window:
                qlayer = model.model.layers[layer_idx].half()
                quant_inplace(qlayer)
                set_quant_state(qlayer,False)
                # step 7: pack quantized weights into low-bits format, note that this process is slow on poor CPU or busy CPU
                if train_params.get("real_quant"):
                    named_linears = get_named_linears(qlayer, int_linear_fake.QuantLinear)
                    for name, module in named_linears.items():
                        quantizer_version = getattr(
                            module, "quantizer_version", train_params.get("quantizer_version", "v1")
                        )
                        scales = export_scale_tensor(module.weight_quantizer)
                        zeros = export_zero_tensor(module.weight_quantizer, quantizer_version)
                        group_size = module.weight_quantizer.group_size
                        print(
                            f"pack quantized {name} with group_size {group_size} and scales max {scales.max()}"
                        )
                        dim0 = module.weight.shape[0]
                        scales = scales.view(dim0, -1).transpose(0, 1).contiguous()
                        zeros = zeros.view(dim0, -1).transpose(0, 1).contiguous()
                        q_linear = build_real_quant_linear(
                            version=quantizer_version,
                            wbits=hyper_params["wbits"],
                            group_size=group_size,
                            in_features=module.in_features,
                            out_features=module.out_features,
                            bias=module.bias is not None,
                            clamp_input=train_params.get("clamp_input", False),
                        )
                        q_linear.pack(module.cpu(), scales.half().float(), zeros.float())
                        set_op_by_name(qlayer, name, q_linear)       
                        logger.info(f"pack quantized {name} finished")
                        del module
            torch.cuda.empty_cache()
            gc.collect()
        if amp_enabled:
            qlayer= qlayer.to(dtype=train_params['dtype'])
            # 更新数据
            # _dtype = attention_mask.dtype
            # attention_mask = attention_mask.to(dtype=torch.float32)
            # DEBUG
            # position_embeddings = (position_embeddings[0].to(dtype=torch.float32),position_embeddings[1].to(dtype=torch.float32))
            # DEBUG
        if not train_params.get("with_catcher"):
            if train_layer_window != shedule_list[-1]:
                with torch.no_grad():
                    with torch.autocast(device_type="cuda",
                                        enabled=amp_enabled):
                        for slide_base in train_layer_window:
                            # 更新后的input 需要经过[windows_start,windows_start+slide_step)层的输出
                            if slide_base == train_layer_window[0]+hyper_params['slide_step']:
                                break
                            layer_idx = slide_base 
                            # print(f"slide_base {slide_base} layer_idx {layer_idx}")
                            layer = model.model.layers[layer_idx].to(train_params['dev'],dtype=train_params['dtype'])
                            next_layer = model.model.layers[layer_idx+hyper_params['slide_step']].to(train_params['dev'],dtype=train_params['dtype'])
                            print(f" layer {layer_idx}  update input")
                            print(f" layer {layer_idx+hyper_params['slide_step']}  update output")
                            # keep_fp_weight 代表用使用原权重更新作为训练输入而不是使用量化后权重进行更新
                            # if train_params.get("keep_fp_weight"):
                            #     fp_layer = fp_layer.to(train_params['dev'],dtype=train_params['dtype'])
                            #     train_dataset.update_dataset(module=fp_layer, 
                            #                         next_module=next_layer,
                            #                         layer_idx=layer_idx+hyper_params['slide_step'],
                            #                         # batch_size=args.batch_size,
                            #                         attention_mask=attention_mask,
                            #                         position_embeddings=position_embeddings,
                            #                             )
                            #     val_dataset.update_dataset(module=fp_layer, 
                            #                             next_module=next_layer,
                            #                             layer_idx=layer_idx+hyper_params['slide_step'],
                            #                             # batch_size=args.batch_size,
                            #                             attention_mask=attention_mask,
                            #                             position_embeddings=position_embeddings,
                            #                                 )
                            #     del fp_layer
                            # else:
                            train_dataset.update_dataset(module=layer, 
                                                    next_module=next_layer,
                                                    layer_idx=layer_idx+hyper_params['slide_step'],
                                                    # batch_size=args.batch_size,
                                                    attention_mask=attention_mask,
                                                    position_embeddings=position_embeddings,
                                                        )
                            val_dataset.update_dataset(module=layer, 
                                                    next_module=next_layer,
                                                    layer_idx=layer_idx+hyper_params['slide_step'],
                                                    # batch_size=args.batch_size,
                                                    attention_mask=attention_mask,
                                                    position_embeddings=position_embeddings,
                                                        )
                            layer.cpu()
                            next_layer.cpu()
            # attention_mask = attention_mask.to(dtype=_dtype)
            # position_embeddings = (position_embeddings[0].to(dtype=_dtype),position_embeddings[1].to(dtype=_dtype))
            # 保存模型
    if train_params.get("log_loss"):
        loss_recorder.save_to_file()
    
    # 关闭可视化记录器
    if vis_recorder is not None:
        vis_recorder.close()
        
    torch.cuda.empty_cache()
    gc.collect()

@timer
def greedy_local_train(
    model: PreTrainedModel,
    config: Dict[str, Any],
    trainloader: List[Tuple[torch.Tensor, torch.Tensor]],
    valloader: List[Tuple[torch.Tensor, torch.Tensor]],
    logger: logging.Logger = None,
):
    logger.info("Starting ...")
    
    train_params = config.get('train_param_settings', {})
    hyper_params = config.get('hyperparam_settings', {})
    
    off_load_to_disk = train_params.get('off_load_to_disk', False)
    if off_load_to_disk:
        logger.info("offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")
    
    dev = config["cluster_settings"]['cuda_ids'][0] if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 
    train_params['dev'] = dev
    train_params['dtype'] = dtype
    
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # step 2: init dataset 改回原来的模式，存入当前层的输入
    # 首先准备需要位置编码和embedding的部分

    model = model.cpu()
    if not train_params.get("with_catcher"):
        print("not using catch")
        train_dataset = LazyLoadDatasetV2(
            model=model,
            dataloader=trainloader,
            crossblock_window_size=hyper_params['crossblock_window_size'],
            device=config["cluster_settings"]['cuda_ids'][0],
        )
        # 准备验证集
        val_dataset = LazyLoadDatasetV2(
            model=model,
            dataloader=valloader,
            crossblock_window_size=hyper_params['crossblock_window_size'],
            device=config["cluster_settings"]['cuda_ids'][0],
        )
        if train_dataset.attention_mask is None:
            attention_mask = train_dataset.attention_mask
        else:
            attention_mask = train_dataset.attention_mask.to(config["cluster_settings"]['cuda_ids'][0])
        position_embeddings = (train_dataset.position_embeddings[0].to(config["cluster_settings"]['cuda_ids'][0]),
                            train_dataset.position_embeddings[1].to(config["cluster_settings"]['cuda_ids'][0]))

        del trainloader, valloader
        gc.collect()

        custom_shedule_train(model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                attention_mask = attention_mask,
                position_embeddings = position_embeddings,
                logger=logger,
                config=config)
    else:
        print("using catch")
        common_train_inps = CommonInputDataset(list(map(lambda x: x[0], trainloader)))
        common_val_inps = CommonInputDataset(list(map(lambda x: x[0], valloader)))
        custom_shedule_train(model,
                train_dataset=common_train_inps,
                val_dataset=common_val_inps,
                attention_mask=None,
                position_embeddings=None,
                logger=logger,
                config=config)

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model
