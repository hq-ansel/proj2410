import copy
import gc
import logging
import math
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from .. import utils
from EfficientQAT.core.pipeline import (
    PipelineConfig,
    PipelineHooks,
    PipelineRunner,
    PipelineStage,
)
from EfficientQAT.core.quantization import (
    build_real_quant_linear,
    export_scale_tensor,
    export_zero_tensor,
)
from . import int_linear_fake
from .utils import (
    quant_parameters, weight_parameters, trainable_parameters,
    set_quant_state, quant_inplace, set_quant_parameters,
    set_weight_parameters, trainable_parameters_num, get_named_linears, set_op_by_name,
)
from ..datautils_block import BlockTrainDataset


class CombinedDataset(Dataset):
    def __init__(self, quant_dataset, fp_dataset):
        assert len(quant_dataset) == len(fp_dataset), "Datasets must have the same length"
        self.quant_dataset = quant_dataset
        self.fp_dataset = fp_dataset

    def __len__(self):
        return len(self.quant_dataset)

    def __getitem__(self, idx):
        quant_data = self.quant_dataset[idx]
        fp_data = self.fp_dataset[idx]
        return quant_data, fp_data


def update_dataset(layer, dataset, device, attention_mask, position_ids):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for index, inps in enumerate(dataset):
                inps = inps.to(device)
                if len(inps.shape) == 2:
                    inps = inps.unsqueeze(0)
                new_data = layer(inps, attention_mask=attention_mask, position_ids=position_ids)[0].to("cpu")
                dataset.update_data(index, new_data)


class _BlockAPPipeline:
    def __init__(self, model, args, trainloader, valloader, logger):
        self.model = model
        self.args = args
        self.trainloader = trainloader
        self.valloader = valloader
        self.logger = logger or logging.getLogger(__name__)
        self.layers = model.model.layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        self.use_cache = model.config.use_cache
        self.loss_func = torch.nn.MSELoss()
        self.cache_paths = {}
        self.fp_train_inps = None
        self.fp_val_inps = None
        self.quant_train_inps = None
        self.quant_val_inps = None
        self.attention_mask = None
        self.position_ids = None
        self.attention_mask_batch = None
        self.loss_recorder = None

    def run(self):
        hooks = PipelineHooks(
            setup=self._setup,
            prepare_data=self._prepare_data,
            build_schedule=self._build_schedule,
            train_stage=self._train_stage,
            after_stage=self._after_stage,
            export=self._export,
            teardown=self._teardown,
        )
        PipelineRunner(PipelineConfig(enable_eval=False), hooks).run()

    # ------------------------------------------------------------------ Hooks
    def _setup(self, ctx):
        self.logger.info("Starting ...")
        if self.args.off_load_to_disk:
            self.logger.info(
                "offload the training dataset to disk, saving CPU memory, "
                "but may slowdown the training due to additional I/O..."
            )
        self.model.config.use_cache = False
        self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.device)
        self.model.model.norm = self.model.model.norm.to(self.device)
        if hasattr(self.model.model, "rotary_emb"):
            self.model.model.rotary_emb = self.model.model.rotary_emb.to(self.device)
        self.layers[0] = self.layers[0].to(self.device)

    def _prepare_data(self, ctx):
        args = self.args
        flag = time.time()
        if args.off_load_to_disk:
            self.cache_paths = {
                "fp_train": f"{args.cache_dir}/{flag}/block_training_fp_train",
                "fp_val": f"{args.cache_dir}/{flag}/block_training_fp_val",
                "quant_train": f"{args.cache_dir}/{flag}/block_training_quant_train",
                "quant_val": f"{args.cache_dir}/{flag}/block_training_quant_val",
            }
            for path in self.cache_paths.values():
                if os.path.exists(path):
                    shutil.rmtree(path)
        else:
            self.cache_paths = {key: None for key in ["fp_train", "fp_val", "quant_train", "quant_val"]}

        self.fp_train_inps = BlockTrainDataset(
            args.train_size, args.training_seqlen, self.model.config.hidden_size,
            args.batch_size, self.dtype, cache_path=self.cache_paths["fp_train"],
            off_load_to_disk=args.off_load_to_disk,
        )
        self.fp_val_inps = BlockTrainDataset(
            args.val_size, args.training_seqlen, self.model.config.hidden_size,
            args.batch_size, self.dtype, cache_path=self.cache_paths["fp_val"],
            off_load_to_disk=args.off_load_to_disk,
        )

        class Catcher(nn.Module):
            def __init__(self, module, dataset):
                super().__init__()
                self.module = module
                self.dataset = dataset
                self.index = 0
                self.attention_mask = None
                self.position_ids = None

            def forward(self, inp, **kwargs):
                self.dataset.update_data(self.index, inp.squeeze(0).to("cpu"))
                self.index += 1
                if self.attention_mask is None:
                    self.attention_mask = kwargs["attention_mask"]
                if self.position_ids is None:
                    self.position_ids = kwargs["position_ids"]
                raise ValueError

        # catch training inputs
        self.layers[0] = Catcher(self.layers[0], self.fp_train_inps)
        iters = len(self.trainloader) // self.args.batch_size
        with torch.no_grad():
            for i in range(iters):
                data = torch.cat(
                    [self.trainloader[j][0] for j in range(i*self.args.batch_size, (i+1)*self.args.batch_size)],
                    dim=0,
                )
                try:
                    self.model(data.to(self.device))
                except ValueError:
                    pass
        self.layers[0] = self.layers[0].module

        # catch validation inputs
        self.layers[0] = Catcher(self.layers[0], self.fp_val_inps)
        iters = len(self.valloader) // self.args.batch_size
        with torch.no_grad():
            for i in range(iters):
                data = torch.cat(
                    [self.valloader[j][0] for j in range(i*self.args.batch_size, (i+1)*self.args.batch_size)],
                    dim=0,
                )
                try:
                    self.model(data.to(self.device))
                except ValueError:
                    pass
        self.attention_mask = self.layers[0].attention_mask
        self.position_ids = self.layers[0].position_ids
        self.layers[0] = self.layers[0].module

        if self.attention_mask is not None:
            self.attention_mask_batch = self.attention_mask.repeat(self.args.batch_size, 1, 1, 1).float()
        else:
            self.logger.info(
                "No attention mask caught from the first layer. "
                "Seems that model's attention works without a mask."
            )
            self.attention_mask_batch = None

        # move embeddings back to CPU
        self.layers[0] = self.layers[0].cpu()
        self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()
        self.model.model.norm = self.model.model.norm.cpu()
        if hasattr(self.model.model, "rotary_emb"):
            self.model.model.rotary_emb = self.model.model.rotary_emb.cpu()
        torch.cuda.empty_cache()

        if args.off_load_to_disk:
            shutil.copytree(self.cache_paths["fp_train"], self.cache_paths["quant_train"])
            shutil.copytree(self.cache_paths["fp_val"], self.cache_paths["quant_val"])

        self.quant_train_inps = BlockTrainDataset(
            args.train_size, args.training_seqlen, self.model.config.hidden_size,
            args.batch_size, self.dtype, cache_path=self.cache_paths["quant_train"],
            off_load_to_disk=args.off_load_to_disk,
        )
        self.quant_val_inps = BlockTrainDataset(
            args.val_size, args.training_seqlen, self.model.config.hidden_size,
            args.batch_size, self.dtype, cache_path=self.cache_paths["quant_val"],
            off_load_to_disk=args.off_load_to_disk,
        )

        if not args.off_load_to_disk:
            for idx, data in enumerate(self.fp_train_inps):
                self.quant_train_inps.update_data(idx, data)
            for idx, data in enumerate(self.fp_val_inps):
                self.quant_val_inps.update_data(idx, data)

        loss_dir = "/home/ubuntu/data/exp/proj2410/logs"
        self.loss_recorder = utils.BlockLossRecorder(
            file_path=os.path.join(loss_dir, "Llama2-7b-block-ap-loss.csv")
        )

    def _build_schedule(self, ctx):
        return [
            PipelineStage(name=f"block-{idx}", metadata={"index": idx})
            for idx in range(len(self.layers))
        ]

    def _train_stage(self, ctx, stage):
        self._train_block(stage.metadata["index"])

    def _after_stage(self, ctx, stage):
        if self.loss_recorder is not None:
            self.loss_recorder.save_to_file()

    def _export(self, ctx):
        if self.args.off_load_to_disk:
            for path in self.cache_paths.values():
                if path and os.path.exists(path):
                    shutil.rmtree(path)
        torch.cuda.empty_cache()
        gc.collect()

    def _teardown(self, ctx):
        self.model.config.use_cache = self.use_cache

    # ------------------------------------------------------------------ Impl
    def _train_block(self, block_index):
        args = self.args
        dev = self.device
        logger = self.logger
        step = 1
        logger.info(f"=== Start quantize blocks {block_index}===")
        layer = self.layers[block_index].to(dev)
        qlayer = copy.deepcopy(layer)
        for name, module in qlayer.named_modules():
            if isinstance(module, torch.nn.Linear):
                quantlinear = int_linear_fake.QuantLinear(module, args.wbits, args.group_size, args)
                set_op_by_name(qlayer, name, quantlinear)
                del module
        qlayer.to(dev)

        set_quant_state(qlayer, weight_quant=False)
        if args.epochs > 0:
            update_dataset(qlayer, self.fp_train_inps, dev, self.attention_mask, self.position_ids)
            update_dataset(qlayer, self.fp_val_inps, dev, self.attention_mask, self.position_ids)
        set_quant_state(qlayer, weight_quant=True)

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()
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
            for epoch in range(args.epochs):
                loss_list = []
                norm_list = []
                start_time = time.time()
                combined_dataset = CombinedDataset(self.quant_train_inps, self.fp_train_inps)
                combined_loader = DataLoader(combined_dataset, batch_size=1, shuffle=True)

                for quant_inps, fp_inps in combined_loader:
                    if len(quant_inps.shape) == 4:
                        quant_inps = quant_inps.squeeze(0)
                        fp_inps = fp_inps.squeeze(0)
                    with torch.cuda.amp.autocast(False):
                        inp = quant_inps.to(dev, dtype=torch.float32)
                        label = fp_inps.to(dev, dtype=torch.float32)
                        quant_out = qlayer(inp, attention_mask=self.attention_mask_batch, position_ids=self.position_ids)[0]
                        reconstruction_loss = self.loss_func(label, quant_out)
                        loss = reconstruction_loss

                    self.loss_recorder.record(f"{block_index}", step, reconstruction_loss.detach().cpu().item())
                    step += 1
                    loss_list.append(loss.detach().cpu())

                    optimizer.zero_grad()
                    loss_scaler(loss, optimizer, clip_grad=max(1.0, args.clip_grad))
                    if args.quant_lr > 0:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]["lr"] = quant_scheduler.get_lr()[0]

                    if args.weight_lr > 0:
                        norm_list.append(
                            utils.dispatch_clip_grad(
                                loss_scaler,
                                optimizer.param_groups[weight_index]["params"],
                                max_norm=max(1.0, args.clip_grad),
                            )
                        )
                        weight_scheduler.step()
                        optimizer.param_groups[weight_index]["lr"] = weight_scheduler.get_lr()[0]
                    step += 1

                val_loss_list = []
                for quant_inps, fp_inps in zip(self.quant_val_inps, self.fp_val_inps):
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            inp = quant_inps.to(dev)
                            label = fp_inps.to(dev)
                            quant_out = qlayer(inp, attention_mask=self.attention_mask_batch, position_ids=self.position_ids)[0]
                            reconstruction_loss = self.loss_func(label, quant_out)
                    val_loss_list.append(reconstruction_loss.cpu())

                train_mean_num = min(len(loss_list), 64)
                loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean() if norm_list else torch.tensor(0.0)
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
            update_dataset(qlayer, self.quant_train_inps, dev, self.attention_mask, self.position_ids)
            update_dataset(qlayer, self.quant_val_inps, dev, self.attention_mask, self.position_ids)
        self.layers[block_index] = qlayer.to("cpu")

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


def block_ap(model, args, trainloader, valloader, logger=None):
    pipeline = _BlockAPPipeline(model, args, trainloader, valloader, logger)
    pipeline.run()
    return model
