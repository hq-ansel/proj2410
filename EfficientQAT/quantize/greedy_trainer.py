import shutil
import functools
import time
import os
import pdb
import gc
import math
from functools import wraps
from contextlib import contextmanager
from typing import List, Tuple, Dict, Union, Callable, Any, Optional
import json

import torch
import torch.nn as nn
import torch.amp
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
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
from ..loss_utils import get_loss_func
from .greedy import GreedyBlockPipeline, trans_quant_block
from .greedy.training import (
    train_units_layers,
    train_units_layers_with_catcher,
)

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


def _greedy_stage_executor(ctx, stage):
    config = ctx.extras["config"]
    train_params = config.get("train_param_settings", {})
    hyper_params = config.get("hyperparam_settings", {})
    train_dataset = ctx.extras["train_dataset"]
    val_dataset = ctx.extras["val_dataset"]
    loss_recorder = ctx.loss_recorder
    vis_recorder = ctx.extras.get("vis_recorder")
    target_model = ctx.extras.get("target_model")
    loss_func = ctx.extras["loss_func"]
    is_quant_layer = ctx.extras["is_quant_layer"]
    schedule = ctx.extras.get("schedule", [])
    with_catcher = train_params.get("with_catcher")
    train_layer_window = stage.metadata["indices"]

    if train_params.get("quant_shedule_type") != "full":
        for layer_idx in train_layer_window:
            if not is_quant_layer[layer_idx]:
                is_quant_layer[layer_idx] = True
                ctx.layers[layer_idx] = trans_quant_block(
                    qlayer=ctx.layers[layer_idx],
                    hyper_params=hyper_params,
                )

    if train_params.get("epochs", 0) > 0:
        skip_layers = train_params.get("skip_layers", [])
        skip_flag = all(layer_idx in skip_layers for layer_idx in train_layer_window) if skip_layers else False
        if not skip_flag:
            if not with_catcher:
                train_units_layers(
                    ctx.model,
                    trainable_layer_idx_list=train_layer_window,
                    loss_func=loss_func,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    attention_mask=ctx.attention_mask,
                    position_embeddings=ctx.position_ids,
                    loss_recorder=loss_recorder,
                    vis_recorder=vis_recorder,
                    logger=ctx.logger,
                    config=config,
                    amp_enabled=amp_enabled,
                )
            else:
                train_units_layers_with_catcher(
                    ctx.model,
                    trainable_layer_idx_list=train_layer_window,
                    loss_func=loss_func,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    target_model=target_model,
                    loss_recorder=loss_recorder,
                    vis_recorder=vis_recorder,
                    logger=ctx.logger,
                    config=config,
                    amp_enabled=amp_enabled,
                )

    selected_layers = nn.ModuleList([ctx.layers[i] for i in train_layer_window])
    quant_inplace(selected_layers)

    with torch.no_grad():
        for layer_idx in train_layer_window:
            qlayer = ctx.layers[layer_idx].half()
            quant_inplace(qlayer)
            set_quant_state(qlayer, False)
            if train_params.get("real_quant"):
                named_linears = get_named_linears(qlayer, int_linear_fake.QuantLinear)
                for name, module in named_linears.items():
                    quantizer_version = getattr(
                        module,
                        "quantizer_version",
                        train_params.get("quantizer_version", "v1"),
                    )
                    scales = export_scale_tensor(module.weight_quantizer)
                    zeros = export_zero_tensor(module.weight_quantizer, quantizer_version)
                    group_size = module.weight_quantizer.group_size
                    ctx.logger.info(
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
                    ctx.logger.info(f"pack quantized {name} finished")
                    del module
        torch.cuda.empty_cache()
        gc.collect()

    if amp_enabled:
        qlayer = qlayer.to(dtype=train_params["dtype"])

    is_last_stage = bool(schedule) and stage is schedule[-1]
    if not with_catcher and not is_last_stage:
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                for slide_base in train_layer_window:
                    if slide_base == train_layer_window[0] + hyper_params["slide_step"]:
                        break
                    layer_idx = slide_base
                    layer = ctx.layers[layer_idx].to(train_params["dev"], dtype=train_params["dtype"])
                    next_layer = ctx.layers[layer_idx + hyper_params["slide_step"]].to(
                        train_params["dev"], dtype=train_params["dtype"]
                    )
                    train_dataset.update_dataset(
                        module=layer,
                        next_module=next_layer,
                        layer_idx=layer_idx + hyper_params["slide_step"],
                        attention_mask=ctx.attention_mask,
                        position_embeddings=ctx.position_ids,
                    )
                    val_dataset.update_dataset(
                        module=layer,
                        next_module=next_layer,
                        layer_idx=layer_idx + hyper_params["slide_step"],
                        attention_mask=ctx.attention_mask,
                        position_embeddings=ctx.position_ids,
                    )
                    layer.cpu()
                    next_layer.cpu()


@timer
def greedy_local_train(
    model: PreTrainedModel,
    config: Dict[str, Any],
    trainloader: List[Tuple[torch.Tensor, torch.Tensor]],
    valloader: List[Tuple[torch.Tensor, torch.Tensor]],
    logger: logging.Logger = None,
):
    pipeline = GreedyBlockPipeline(
        model,
        config,
        trainloader,
        valloader,
        executor=_greedy_stage_executor,
        logger=logger,
    )
    pipeline.run()
    return model
