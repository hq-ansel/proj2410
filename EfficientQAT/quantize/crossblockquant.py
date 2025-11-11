import copy
import logging
import math
import os
import pdb
import time
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from . import int_linear_fake
from .. import utils
from ..loss_utils import get_loss_func
from .block_pipeline import (
    BlockContext,
    BlockPipeline,
    CombinedDataset,
    build_block_schedule,
    update_dataset,
)
from .utils import (
    get_named_linears,
    quant_inplace,
    quant_parameters,
    set_op_by_name,
    set_quant_parameters,
    set_quant_state,
    set_weight_parameters,
    trainable_parameters,
    trainable_parameters_num,
    weight_parameters,
)
from EfficientQAT.core.quantization import (
    build_real_quant_linear,
    export_scale_tensor,
    export_zero_tensor,
)



amp_enabled = os.environ.get("AMP_ENABLED", "True").lower() == "true"


def _ensure_state(ctx: BlockContext) -> dict:
    return ctx.extras.setdefault(
        "crossblock_state",
        {"qlayers": [None] * len(ctx.layers)},
    )


def _train_crossblock_window(
    ctx: BlockContext,
    stage,
    *,
    loss_func: Callable,
    slide_step: int,
) -> None:
    args = ctx.args
    dev = ctx.device
    logger = ctx.logger

    indices: List[int] = stage.metadata["indices"]
    start = stage.metadata["start"]
    end = stage.metadata["end"]
    logger.info(f"=== Start quantize blocks {start} to {end} ===")

    state = _ensure_state(ctx)
    qlayers_state: List[Optional[nn.Module]] = state["qlayers"]

    window_modules: List[nn.Module] = []
    for idx in indices:
        qlayer = qlayers_state[idx]
        if qlayer is None:
            layer = ctx.layers[idx].to(dev)
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module, nn.Linear):
                    quantlinear = int_linear_fake.QuantLinear(module, args.wbits, args.group_size, args)
                    set_op_by_name(qlayer, name, quantlinear)
            qlayer.to(dev)
        else:
            qlayer = qlayer.to(dev)
        qlayers_state[idx] = qlayer
        window_modules.append(qlayer)
    module_list = nn.ModuleList(window_modules)

    fp_train = ctx.fp_train_inps
    fp_val = ctx.fp_val_inps
    quant_train = ctx.quant_train_inps
    quant_val = ctx.quant_val_inps
    if None in (fp_train, fp_val, quant_train, quant_val):
        raise RuntimeError("BlockPipeline datasets have not been initialised.")

    set_quant_state(module_list, weight_quant=False)
    if args.epochs > 0:
        update_dataset(module_list, fp_train, dev, ctx.attention_mask, ctx.position_ids)
        update_dataset(module_list, fp_val, dev, ctx.attention_mask, ctx.position_ids)
    set_quant_state(module_list, weight_quant=True)

    if args.epochs > 0:
        with torch.no_grad():
            for module in module_list:
                module.float()

        param = []
        assert args.quant_lr > 0 or args.weight_lr > 0
        total_training_iteration = args.epochs * args.train_size / args.batch_size
        if args.quant_lr > 0:
            set_quant_parameters(module_list, True)
            param.append({"params": quant_parameters(module_list), "lr": args.quant_lr})
            empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.quant_lr)
            quant_scheduler = CosineAnnealingLR(
                empty_optimizer_1,
                T_max=total_training_iteration,
                eta_min=args.quant_lr/args.min_lr_factor,
            )
            quant_index = 0
        else:
            set_quant_parameters(module_list, False)

        if args.weight_lr > 0:
            set_weight_parameters(module_list, True)
            param.append({"params": weight_parameters(module_list), "lr": args.weight_lr})
            empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
            weight_scheduler = CosineAnnealingLR(
                empty_optimizer_2,
                T_max=total_training_iteration,
                eta_min=args.weight_lr/args.min_lr_factor,
            )
            weight_index = 0 if args.quant_lr <= 0 else 1
        else:
            set_weight_parameters(module_list, False)

        optimizer = torch.optim.AdamW(param, weight_decay=args.wd, foreach=True)
        loss_scaler = utils.NativeScalerWithGradNormCount(use_amp=amp_enabled)
        trainable_number = trainable_parameters_num(list(module_list))
        logger.info(f"trainable parameter number: {trainable_number/1e6}M")
        logger.info(f"Memory profile of qlayers: {utils.profile_memory(module_list)}")
        logger.info(f"Memory profile of optimizer: {utils.profile_memory(optimizer)}")

        best_val_loss = 1e6
        early_stop_flag = 0
        dataset = CombinedDataset(quant_train, fp_train)

        for epoch in range(args.epochs):
            loss_list = []
            norm_list = []
            start_time = time.time()
            torch.autograd.set_detect_anomaly(True)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            step = 1

            for quant_inps, fp_inps in dataloader:
                with torch.autocast(device_type="cuda", enabled=amp_enabled, dtype=torch.bfloat16):
                    hidden_states = quant_inps.to(dev)
                    label = fp_inps.to(dev)
                    if len(hidden_states.shape) == 4:
                        hidden_states = hidden_states.squeeze(0)
                        label = label.squeeze(0)

                    for module in module_list:
                        if not math.isfinite(hidden_states.sum().item()):
                            logger.info("hidden_states is NAN, stopping training")
                            pdb.set_trace()
                        hidden_states = module(
                            hidden_states,
                            attention_mask=ctx.attention_mask_batch,
                            position_ids=ctx.position_ids,
                        )[0]
                    if not math.isfinite(hidden_states.sum().item()):
                        logger.info("hidden_states is NAN, stopping training")
                        pdb.set_trace()
                    quant_out = hidden_states
                    reconstruction_loss = loss_func(label, quant_out)
                    loss = reconstruction_loss

                if not math.isfinite(loss.item()):
                    logger.info("Loss is NAN, stopping training")
                    pdb.set_trace()
                if ctx.loss_recorder is not None:
                    ctx.loss_recorder.record(
                        f"blk{start}-{end}",
                        step,
                        reconstruction_loss.detach().cpu().item(),
                    )
                loss_list.append(reconstruction_loss.detach().cpu())
                optimizer.zero_grad()

                norm = loss_scaler(
                    loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    parameters=trainable_parameters(module_list),
                ).cpu()
                norm_list.append(norm.data)

                if args.quant_lr > 0:
                    quant_scheduler.step()
                    optimizer.param_groups[quant_index]["lr"] = quant_scheduler.get_lr()[0]
                if args.weight_lr > 0:
                    weight_scheduler.step()
                    optimizer.param_groups[weight_index]["lr"] = weight_scheduler.get_lr()[0]
                step += 1

            val_loss_list = []
            for quant_inps, fp_inps in zip(quant_val, fp_val):
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", enabled=amp_enabled, dtype=torch.bfloat16):
                        hidden_states = quant_inps.to(dev)
                        label = fp_inps.to(dev)
                        for module in module_list:
                            hidden_states = module(
                                hidden_states,
                                attention_mask=ctx.attention_mask_batch,
                                position_ids=ctx.position_ids,
                            )[0]
                        quant_out = hidden_states
                        reconstruction_loss = loss_func(label, quant_out)
                val_loss_list.append(reconstruction_loss.cpu())

            train_mean_num = min(len(loss_list), 64)
            loss_mean = torch.stack(loss_list)[-(train_mean_num - 1):].mean()
            val_loss_mean = torch.stack(val_loss_list).mean()
            norm_mean = torch.stack(norm_list).mean() if norm_list else torch.tensor(0.0)
            logger.info(
                f"blocks {start}-{end} epoch {epoch} recon_loss:{loss_mean} "
                f"val_loss:{val_loss_mean} quant_lr:{args.quant_lr} "
                f"norm:{norm_mean:.8f} max memory_allocated "
                f"{torch.cuda.max_memory_allocated(dev) / 1024**2} time {time.time()-start_time} "
            )
            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
            else:
                early_stop_flag += 1
                if args.early_stop > 0 and early_stop_flag >= args.early_stop:
                    break
        optimizer.zero_grad()
        del optimizer

    module_list.half()
    quant_inplace(module_list)
    set_quant_state(module_list, weight_quant=False)

    update_count = min(slide_step, len(module_list))
    newly_quantized = module_list[:update_count]
    if args.epochs > 0:
        update_dataset(newly_quantized, quant_train, dev, ctx.attention_mask, ctx.position_ids)
        update_dataset(newly_quantized, quant_val, dev, ctx.attention_mask, ctx.position_ids)

    for idx, module in zip(indices, module_list):
        cpu_module = module.to("cpu")
        qlayers_state[idx] = cpu_module
        ctx.layers[idx] = cpu_module

    if args.real_quant:
        for module in newly_quantized:
            named_linears = get_named_linears(module, int_linear_fake.QuantLinear)
            for name, sub_module in named_linears.items():
                quantizer_version = getattr(
                    sub_module,
                    "quantizer_version",
                    getattr(args, "quantizer_version", "v1"),
                )
                scales = export_scale_tensor(sub_module.weight_quantizer)
                zeros = export_zero_tensor(sub_module.weight_quantizer, quantizer_version)
                group_size = sub_module.weight_quantizer.group_size
                dim0 = sub_module.weight.shape[0]
                scales = scales.view(dim0, -1).transpose(0, 1).contiguous()
                zeros = zeros.view(dim0, -1).transpose(0, 1).contiguous()
                q_linear = build_real_quant_linear(
                    version=quantizer_version,
                    wbits=args.wbits,
                    group_size=group_size,
                    in_features=sub_module.in_features,
                    out_features=sub_module.out_features,
                    bias=sub_module.bias is not None,
                    clamp_input=getattr(args, "clamp_input", False),
                )
                q_linear.pack(sub_module.cpu(), scales.float(), zeros.float())
                set_op_by_name(module, name, q_linear)
                logger.info(f"pack quantized {name} finished")

    torch.cuda.empty_cache()


def cross_block_quantization(
    model: PreTrainedModel,
    args,
    trainloader,
    valloader,
    logger: Optional[logging.Logger] = None,
):
    logger = logger or logging.getLogger(__name__)
    slide_step = max(1, getattr(args, "crossblock_slide_step", 1))
    window_size = getattr(args, "crossblock_window_size", 1)
    loss_func = get_loss_func(args.loss_func)
    loss_factory = (
        (lambda: utils.BlockLossRecorder(file_path=args.log_loss))
        if getattr(args, "log_loss", None)
        else None
    )

    def schedule_builder(ctx: BlockContext):
        return build_block_schedule(len(ctx.layers), window_size=window_size, step=slide_step)

    executor = lambda ctx, stage: _train_crossblock_window(
        ctx,
        stage,
        loss_func=loss_func,
        slide_step=slide_step,
    )

    pipeline = BlockPipeline(
        model,
        args,
        trainloader,
        valloader,
        executor=executor,
        logger=logger,
        loss_recorder_factory=loss_factory,
        enable_loss_recorder=loss_factory is not None,
        schedule_builder=schedule_builder,
    )
    pipeline.run()
    return model
