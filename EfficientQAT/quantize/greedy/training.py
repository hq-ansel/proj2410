from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers.modeling_utils import PreTrainedModel

from .. import int_linear_fake
from ..utils import (
    quant_parameters,
    weight_parameters,
    trainable_parameters,
    set_quant_state,
    set_quant_parameters,
    set_weight_parameters,
    trainable_parameters_num,
    StopException,
)
from .common import CatcherManager, CosineAnnealingScheduler




def train_units_layers(
    model: PreTrainedModel,
    trainable_layer_idx_list: List[int],
    loss_func,
    train_dataset,
    val_dataset,
    attention_mask: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    loss_recorder,
    vis_recorder=None,
    logger=None,
    config: Dict[str, Any] = None,
    *,
    amp_enabled: bool = True,
) -> None:
    train_params = config.get("train_param_settings", {})
    hyper_params = config.get("hyperparam_settings", {})

    total_training_iteration = (
        train_params["epochs"] * train_params["train_size"] / train_params["batch_size"]
    )
    layer_idx_set = set(trainable_layer_idx_list)
    step = 0
    assert float(train_params["quant_lr"]) > 0 or float(train_params["weight_lr"]) > 0

    with torch.no_grad():
        model.model.layers = nn.ModuleList(
            [
                qlayer.to(train_params["dev"], dtype=torch.float32)
                if index in layer_idx_set
                else qlayer.half()
                for index, qlayer in enumerate(model.model.layers)
            ]
        )

    qlayers = model.model.layers
    selected_layers = nn.ModuleList([qlayers[i] for i in trainable_layer_idx_list])

    for param in model.parameters():
        param.requires_grad = False

    param_groups = []
    param_group_index = 0
    quant_scheduler = None
    weight_scheduler = None
    quant_index = None
    weight_index = None

    quant_scheduler = None
    weight_scheduler = None
    quant_index = None
    weight_index = None

    for layer_idx in trainable_layer_idx_list:
        qlayer = model.model.layers[layer_idx]
        set_quant_state(qlayer, True)
        with torch.no_grad():
            qlayer.float()
        if float(train_params["quant_lr"]) > 0:
            set_quant_parameters(qlayer, True)
            param_groups.append(
                {"params": quant_parameters(qlayer), "lr": train_params["quant_lr"]}
            )
            empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=train_params["quant_lr"])
            quant_scheduler = CosineAnnealingLR(
                empty_optimizer_1,
                T_max=total_training_iteration,
                eta_min=train_params["quant_lr"] / train_params["min_lr_factor"],
            )
            quant_index = param_group_index
            param_group_index += 1
        else:
            set_quant_parameters(qlayer, False)

        if float(train_params["weight_lr"]) > 0:
            set_weight_parameters(qlayer, True)
            param_groups.append(
                {"params": weight_parameters(qlayer), "lr": train_params["weight_lr"]}
            )
            empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=train_params["weight_lr"])
            weight_scheduler = CosineAnnealingLR(
                empty_optimizer_2,
                T_max=total_training_iteration,
                eta_min=train_params["weight_lr"] / train_params["min_lr_factor"],
            )
            weight_index = param_group_index
            param_group_index += 1
        else:
            set_weight_parameters(qlayer, False)

    if config.get("loss_func") == "AFFINE_MSE":
        loss_func.reinitialize_A()
        loss_func = loss_func.to(train_params["dev"])
        param_groups.append({"params": loss_func.parameters(), "lr": train_params["weight_lr"]})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_params["wd"], foreach=True)
    loss_scaler = torch.amp.GradScaler(device=train_params["dev"])
    trainable_number = trainable_parameters_num(selected_layers)
    logger.info(f"trainable parameter number: {trainable_number/1e6}M")

    best_val_loss = 1e6
    early_stop_flag = 0

    graualWarmupScheduler = None
    if hyper_params.get("gradual_quant", False) or config.get("interpolate", False):
        gradual_factor = hyper_params.get("gradual_factor", 2.0)

        class GradualWarmupScheduler:
            def __init__(
                self,
                linear_list: List[int_linear_fake.QuantLinear],
                total_iteration: int,
                gradual_factor: float = 2.0,
            ):
                self.linear_list = linear_list
                self.total_iteration = total_iteration
                self.iteration = 0
                self.gradual_factor = gradual_factor
                self.update()

            def update(self):
                self.iteration += 1
                threshold = self.total_iteration / self.gradual_factor
                for linear in self.linear_list:
                    if self.iteration < threshold:
                        ratio = self.iteration / threshold
                        if hyper_params.get("gradual_quant", False):
                            linear.update_position_ratio(ratio)
                        if config.get("interpolate", False):
                            linear.update_interpolate_ratio(1 - ratio)
                    else:
                        linear.update_position_ratio(1.0)
                        if config.get("interpolate", False):
                            linear.update_interpolate_ratio(0)

        q_linear_list = []
        for i in trainable_layer_idx_list:
            for _, module in qlayers[i].named_modules():
                if isinstance(module, int_linear_fake.QuantLinear):
                    q_linear_list.append(module)
        graualWarmupScheduler = GradualWarmupScheduler(
            q_linear_list,
            total_training_iteration,
            gradual_factor,
        )

    dampen_loss_weight_scheduler = None
    if hyper_params.get("dampen_loss"):
        dampen_loss_weight = hyper_params.get("dampen_loss_weight", 0.01)
        dampen_loss_weight_scheduler = CosineAnnealingScheduler(
            max_value=dampen_loss_weight,
            min_value=0,
            total_steps=total_training_iteration,
            ascend=True,
        )

    position_ids = torch.arange(
        train_params["training_seqlen"], dtype=torch.long, device=train_params["dev"]
    ).unsqueeze(0).expand(train_params["batch_size"], -1).contiguous()

    if hyper_params.get("swa", False):
        swa_model = torch.optim.swa_utils.AveragedModel(
            selected_layers,
            device=train_params["dev"],
            avg_fn=torch.optim.swa_utils.get_ema_avg_fn(hyper_params.get("swa_factor", 0.9)),
        )
        swa_start = hyper_params.get("swa_start")

    for epoch in range(train_params["epochs"]):
        loss_list = []
        norm_list = []
        start_time = time.time()
        dataloader = DataLoader(
            train_dataset,
            batch_size=train_params["batch_size"],
            pin_memory=True,
            shuffle=False,
        )

        for index, input_data in enumerate(dataloader):
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                inp, target = input_data
                inp = inp.to(train_params["dev"], dtype=train_params["dtype"])
                trg = target.to(train_params["dev"], dtype=torch.float32)
                hidden_state = inp
                for layer_idx in trainable_layer_idx_list:
                    layer_outputs = qlayers[layer_idx](
                        hidden_states=hidden_state,
                        attention_mask=attention_mask.float(),
                        position_ids=position_ids,
                        output_attentions=False,
                        use_cache=False,
                    )
                    hidden_state = layer_outputs[0]
                loss = loss_func(hidden_state, trg)

                if hyper_params.get("dampen_loss") and dampen_loss_weight_scheduler is not None:
                    dampen_loss_weight = dampen_loss_weight_scheduler.step()
                    loss = loss * dampen_loss_weight

            if graualWarmupScheduler is not None:
                graualWarmupScheduler.update()

            if not torch.isfinite(loss):
                raise StopException("Loss explode")

            loss_list.append(loss.detach().cpu())
            loss_scaler.scale(loss).backward() if amp_enabled else loss.backward()

            if amp_enabled:
                loss_scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(selected_layers),
                    max_norm=train_params["clip_grad"],
                )
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(selected_layers),
                    max_norm=train_params["clip_grad"],
                )
                optimizer.step()

            norm_list.append(grad_norm.detach().cpu())

            if quant_scheduler is not None and quant_index is not None:
                quant_scheduler.step()
                optimizer.param_groups[quant_index]["lr"] = quant_scheduler.get_last_lr()[0]
            if weight_scheduler is not None and weight_index is not None:
                weight_scheduler.step()
                optimizer.param_groups[weight_index]["lr"] = weight_scheduler.get_last_lr()[0]

            if amp_enabled:
                torch.cuda.synchronize()

            if loss_recorder is not None:
                loss_recorder.record(
                    f"{trainable_layer_idx_list}",
                    step,
                    loss.detach().cpu().item(),
                )
            step += 1

        val_loss_list = []
        with torch.no_grad():
            for inp, target in DataLoader(
                val_dataset,
                batch_size=train_params["batch_size"],
                pin_memory=True,
                shuffle=False,
            ):
                with torch.autocast(device_type="cuda", enabled=amp_enabled):
                    inp = inp.to(train_params["dev"], dtype=train_params["dtype"])
                    trg = target.to(train_params["dev"], dtype=torch.float32)
                    hidden_state = inp
                    for layer_idx in trainable_layer_idx_list:
                        layer_outputs = qlayers[layer_idx](
                            hidden_states=hidden_state,
                            attention_mask=attention_mask.float(),
                            position_ids=position_ids,
                            output_attentions=False,
                            use_cache=False,
                        )
                        hidden_state = layer_outputs[0]
                    val_loss = loss_func(hidden_state, trg)
                    val_loss_list.append(val_loss.detach().cpu())

        loss_mean = torch.stack(loss_list).mean()
        val_loss_mean = torch.stack(val_loss_list).mean()
        norm_mean = torch.stack(norm_list).mean()
        logger.info(
            f"trainable blocks {trainable_layer_idx_list} epoch {epoch} "
            f"recon_loss:{loss_mean} val_loss:{val_loss_mean} grad_norm:{norm_mean}"
        )

        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
        else:
            early_stop_flag += 1
            if int(config.get("early_stop", 0)) > 0 and early_stop_flag >= int(config.get("early_stop", 0)):
                break

    optimizer.zero_grad()
    del optimizer
    torch.cuda.empty_cache()


def train_units_layers_with_catcher(
    model: PreTrainedModel,
    trainable_layer_idx_list: List[int],
    loss_func,
    train_dataset,
    val_dataset,
    target_model: PreTrainedModel,
    loss_recorder,
    vis_recorder=None,
    logger=None,
    config: Dict[str, Any] = None,
    *,
    amp_enabled: bool = True,
) -> None:
    train_params = config.get("train_param_settings", {})
    hyper_params = config.get("hyperparam_settings", {})

    total_training_iteration = (
        train_params["epochs"] * train_params["train_size"] / train_params["batch_size"]
    )
    layer_idx_set = set(trainable_layer_idx_list)
    qlayers = model.model.layers
    fp_layers = copy.deepcopy(model.model.layers)

    with torch.no_grad():
        for index, qlayer in enumerate(model.model.layers):
            if index in layer_idx_set:
                qlayer.to(train_params["dev"], dtype=torch.float32)
            else:
                qlayer.half()

    selected_layers = nn.ModuleList([model.model.layers[i] for i in trainable_layer_idx_list])
    for param in model.parameters():
        param.requires_grad = False

    param_groups = []
    param_group_index = 0
    for layer_idx in trainable_layer_idx_list:
        qlayer = model.model.layers[layer_idx]
        set_quant_state(qlayer, True)
        with torch.no_grad():
            qlayer.float()
        if float(train_params["quant_lr"]) > 0:
            set_quant_parameters(qlayer, True)
            param_groups.append(
                {"params": quant_parameters(qlayer), "lr": train_params["quant_lr"]}
            )
            empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=train_params["quant_lr"])
            quant_scheduler = CosineAnnealingLR(
                empty_optimizer_1,
                T_max=total_training_iteration,
                eta_min=train_params["quant_lr"] / hyper_params["min_lr_factor"],
            )
            quant_index = param_group_index
            param_group_index += 1
        else:
            set_quant_parameters(qlayer, False)

        if float(train_params["weight_lr"]) > 0:
            set_weight_parameters(qlayer, True)
            param_groups.append(
                {"params": weight_parameters(qlayer), "lr": train_params["weight_lr"]}
            )
            empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=train_params["weight_lr"])
            weight_scheduler = CosineAnnealingLR(
                empty_optimizer_2,
                T_max=total_training_iteration,
                eta_min=train_params["weight_lr"] / hyper_params["min_lr_factor"],
            )
            weight_index = param_group_index
            param_group_index += 1
        else:
            set_weight_parameters(qlayer, False)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_params["wd"], foreach=True)
    loss_scaler = torch.amp.GradScaler(device=train_params["dev"])
    trainable_number = trainable_parameters_num(selected_layers)
    logger.info(f"trainable parameter number: {trainable_number/1e6}M")

    best_val_loss = 1e6
    early_stop_flag = 0

    for epoch in range(train_params["epochs"]):
        loss_list = []
        norm_list = []
        start_time = time.time()

        qlayer_idxs = trainable_layer_idx_list
        fp_layer_idxs = trainable_layer_idx_list
        with CatcherManager(qlayers, qlayer_idxs), CatcherManager(fp_layers, fp_layer_idxs):
            for batch_idx, inp in enumerate(train_dataset):
                with torch.autocast(device_type="cuda", enabled=amp_enabled):
                    inp = inp.to(train_params["dev"], dtype=train_params["dtype"])
                    hidden_state = inp
                    for layer_idx in trainable_layer_idx_list:
                        layer_outputs = qlayers[layer_idx](
                            hidden_states=hidden_state,
                            attention_mask=None,
                            output_attentions=False,
                            use_cache=False,
                        )
                        hidden_state = layer_outputs[0]
                    target_output = target_model(inp.to(train_params["dev"]))[0]
                    loss = loss_func(hidden_state, target_output)

                optimizer.zero_grad()
                loss_scaler.scale(loss).backward()
                loss_scaler.step(optimizer)
                loss_scaler.update()
                if quant_scheduler is not None and quant_index is not None:
                    quant_scheduler.step()
                    optimizer.param_groups[quant_index]["lr"] = quant_scheduler.get_last_lr()[0]
                if weight_scheduler is not None and weight_index is not None:
                    weight_scheduler.step()
                    optimizer.param_groups[weight_index]["lr"] = weight_scheduler.get_last_lr()[0]
                loss_list.append(loss.detach().cpu())

        val_loss_list = []
        with torch.no_grad():
            for inp in val_dataset:
                with torch.autocast(device_type="cuda", enabled=amp_enabled):
                    inp = inp.to(train_params["dev"], dtype=train_params["dtype"])
                    hidden_state = inp
                    for layer_idx in trainable_layer_idx_list:
                        layer_outputs = qlayers[layer_idx](
                            hidden_states=hidden_state,
                            attention_mask=None,
                            output_attentions=False,
                            use_cache=False,
                        )
                        hidden_state = layer_outputs[0]
                    target_output = target_model(inp.to(train_params["dev"]))[0]
                    val_loss = loss_func(hidden_state, target_output)
                    val_loss_list.append(val_loss.detach().cpu())

        loss_mean = torch.stack(loss_list).mean()
        val_loss_mean = torch.stack(val_loss_list).mean()
        norm_mean = torch.stack(norm_list).mean() if norm_list else torch.tensor(0.0)
        logger.info(
            f"blocks {trainable_layer_idx_list} epoch {epoch} recon_loss:{loss_mean} "
            f"val_loss:{val_loss_mean} grad_norm:{norm_mean:.8f} time {time.time()-start_time}"
        )

        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
        else:
            early_stop_flag += 1
            if int(config.get("early_stop", 0)) > 0 and early_stop_flag >= int(config.get("early_stop", 0)):
                break

    optimizer.zero_grad()
    del optimizer
    torch.cuda.empty_cache()
