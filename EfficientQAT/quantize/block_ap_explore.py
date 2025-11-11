import copy
import math
import pdb
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .. import utils
from EfficientQAT.core.quantization import (
    build_real_quant_linear,
    export_scale_tensor,
    export_zero_tensor,
)
from . import int_linear_fake
from .block_pipeline import BlockContext, BlockPipeline, CombinedDataset, update_dataset
from .utils import (
    quant_parameters, weight_parameters, trainable_parameters,
    set_quant_state, quant_inplace, set_quant_parameters,
    set_weight_parameters, trainable_parameters_num, get_named_linears, set_op_by_name,
)


def _train_block_explore(ctx: BlockContext, stage) -> None:
    block_index = stage.metadata["indices"][0]
    args = ctx.args
    dev = ctx.device
    logger = ctx.logger
    loss_func = nn.MSELoss()
    fp_train = ctx.fp_train_inps
    fp_val = ctx.fp_val_inps
    quant_train = ctx.quant_train_inps
    quant_val = ctx.quant_val_inps
    if fp_train is None or fp_val is None or quant_train is None or quant_val is None:
        raise RuntimeError("BlockPipeline datasets have not been initialised.")

    step = 1
    logger.info(f"=== Start quantize blocks {block_index}===")
    layer = ctx.layers[block_index].to(dev)
    qlayer = copy.deepcopy(layer)
    for name, module in qlayer.named_modules():
        if isinstance(module, nn.Linear):
            quantlinear = int_linear_fake.QuantLinear(module, args.wbits, args.group_size, args)
            set_op_by_name(qlayer, name, quantlinear)
            del module
    qlayer.to(dev)

    set_quant_state(qlayer, weight_quant=False)
    if args.epochs > 0:
        update_dataset(qlayer, fp_train, dev, ctx.attention_mask, ctx.position_ids)
        update_dataset(qlayer, fp_val, dev, ctx.attention_mask, ctx.position_ids)
    set_quant_state(qlayer, weight_quant=True)

    if args.epochs > 0:
        with torch.no_grad():
            qlayer.float()  # fp32 is required for AMP training
        param = []
        assert args.quant_lr > 0 or args.weight_lr > 0
        param_group_index = 0
        total_training_iteration = args.epochs * args.train_size / args.batch_size
        if args.quant_lr > 0:
            set_quant_parameters(qlayer, True)
            param.append({"params": quant_parameters(qlayer), "lr": args.quant_lr})
            empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.quant_lr)
            quant_scheduler = CosineAnnealingLR(
                empty_optimizer_1,
                T_max=total_training_iteration,
                eta_min=args.quant_lr/args.min_lr_factor,
            )
            quant_index = param_group_index
            param_group_index += 1
        else:
            set_quant_parameters(qlayer, False)

        if args.weight_lr > 0:
            set_weight_parameters(qlayer, True)
            param.append({"params": weight_parameters(qlayer), "lr": args.weight_lr})
            empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
            weight_scheduler = CosineAnnealingLR(
                empty_optimizer_2,
                T_max=total_training_iteration,
                eta_min=args.weight_lr/args.min_lr_factor,
            )
            weight_index = param_group_index
            param_group_index += 1
        else:
            set_weight_parameters(qlayer, False)
        optimizer = torch.optim.AdamW(param, weight_decay=args.wd, foreach=True)
        loss_scaler = utils.NativeScalerWithGradNormCount()
        trainable_number = trainable_parameters_num(qlayer)
        logger.info(f"trainable parameter number: {trainable_number/1e6}M")

        best_val_loss = 1e6
        early_stop_flag = 0
        combined_dataset = CombinedDataset(quant_train, fp_train)
        combined_loader = DataLoader(combined_dataset, batch_size=1, shuffle=True)
        for epoch in range(args.epochs):
            loss_list = []
            norm_list = []
            start_time = time.time()

            for quant_inps, fp_inps in combined_loader:
                if len(quant_inps.shape) == 4:
                    quant_inps = quant_inps.squeeze(0)
                    fp_inps = fp_inps.squeeze(0)
                with torch.cuda.amp.autocast(False):
                    inp = quant_inps.to(dev, dtype=torch.float32)
                    label = fp_inps.to(dev, dtype=torch.float32)
                    quant_out = qlayer(
                        inp,
                        attention_mask=ctx.attention_mask_batch,
                        position_ids=ctx.position_ids,
                    )[0]
                    reconstruction_loss = loss_func(label, quant_out)
                    loss = reconstruction_loss

                if not math.isfinite(loss.item()):
                    logger.info("Loss is NAN, stopping training")
                    pdb.set_trace()

                if ctx.loss_recorder is not None:
                    ctx.loss_recorder.record(
                        f"{block_index}",
                        step,
                        reconstruction_loss.detach().cpu().item(),
                    )
                loss_list.append(reconstruction_loss.detach().cpu())
                optimizer.zero_grad()
                norm = loss_scaler(loss, optimizer, parameters=trainable_parameters(qlayer)).cpu()
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
                    with torch.cuda.amp.autocast():
                        inp = quant_inps.to(dev)
                        label = fp_inps.to(dev)
                        quant_out = qlayer(
                            inp,
                            attention_mask=ctx.attention_mask_batch,
                            position_ids=ctx.position_ids,
                        )[0]
                        reconstruction_loss = loss_func(label, quant_out)
                val_loss_list.append(reconstruction_loss.cpu())

            train_mean_num = min(len(loss_list), 64)
            loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
            val_loss_mean = torch.stack(val_loss_list).mean()
            norm_mean = torch.stack(norm_list).mean()
            logger.info(
                f"blocks {block_index} epoch {epoch} recon_loss:{loss_mean} "
                f"val_loss:{val_loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} "
                f"norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} "
                f"time {time.time()-start_time} "
            )
            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
            else:
                early_stop_flag += 1
                if args.early_stop > 0 and early_stop_flag >= args.early_stop:
                    break

        optimizer.zero_grad()
        del optimizer

    qlayer.half()
    quant_inplace(qlayer)
    set_quant_state(qlayer, weight_quant=False)

    if args.epochs > 0:
        update_dataset(qlayer, quant_train, dev, ctx.attention_mask, ctx.position_ids)
        update_dataset(qlayer, quant_val, dev, ctx.attention_mask, ctx.position_ids)
    ctx.layers[block_index] = qlayer.to("cpu")

    if ctx.loss_recorder is not None:
        ctx.loss_recorder.save_to_file()

    if args.real_quant:
        named_linears = get_named_linears(qlayer, int_linear_fake.QuantLinear)
        for name, module in named_linears.items():
            quantizer_version = getattr(module, "quantizer_version", getattr(args, "quantizer_version", "v1"))
            scales = export_scale_tensor(module.weight_quantizer)
            zeros = export_zero_tensor(module.weight_quantizer, quantizer_version)
            group_size = module.weight_quantizer.group_size
            dim0 = module.weight.shape[0]
            scales = scales.view(dim0, -1).transpose(0, 1).contiguous()
            zeros = zeros.view(dim0, -1).transpose(0, 1).contiguous()
            q_linear = build_real_quant_linear(
                version=quantizer_version,
                wbits=args.wbits,
                group_size=group_size,
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                clamp_input=getattr(args, "clamp_input", False),
            )
            q_linear.pack(module.cpu(), scales.float(), zeros.float())
            set_op_by_name(qlayer, name, q_linear)
            logger.info(f"pack quantized {name} finished")
            del module
    del layer
    torch.cuda.empty_cache()


def block_ap_explore(model, args, trainloader, valloader, logger=None):
    pipeline = BlockPipeline(
        model,
        args,
        trainloader,
        valloader,
        executor=_train_block_explore,
        logger=logger,
    )
    pipeline.run()
    return model
