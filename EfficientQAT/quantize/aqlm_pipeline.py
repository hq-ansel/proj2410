import shutil
import functools
import time
import os
import copy
import pdb
import gc
import math
from functools import wraps
from itertools import chain
from collections import defaultdict
from contextlib import contextmanager
from typing import List, Tuple, Dict, Union, Callable,Sequence,Any
from concurrent.futures import ThreadPoolExecutor
from regex import E, T
from tqdm import tqdm
from tqdm import trange
from tqdm.auto import trange

import logging
import json
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
from transformers import PreTrainedModel,AutoTokenizer,AutoModelForCausalLM,AutoConfig
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from easydict import EasyDict as edict
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from accelerate.utils.modeling import get_balanced_memory

from llm_aqlm.aq_engine import AQEngine
from llm_aqlm.src.aq import QuantizedLinear
from llm_aqlm.src.utils import using_tf32
from llm_aqlm.src.finetune import finetune_groupwise

MODEL_ERROR_MSG = "Unsupported model type {} - only 'llama', 'Yi', 'opt', 'falcon', 'phi3' are supported"
FALCON_TYPES = ("falcon", "refinedweb", "refinedwebmodel")
LLAMA_LIKE = ("llama", "Yi", "mistral", "mixtral", "gemma", "cohere", "qwen2")

def load_awq_model_tokenizer(model_path,
                             ):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def get_sequential_groups(model):
    if model.config.model_type in LLAMA_LIKE:
        assert "mixtral" not in model.config.model_type.lower()  # check that this is not mixtral
        return [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]
    elif model.config.model_type.lower() in FALCON_TYPES:
        return [
            ["self_attention.query_key_value"],
            ["self_attention.dense"],
            ["mlp.dense_h_to_4h"],
            ["mlp.dense_4h_to_h"],
        ]
    elif model.config.model_type == "opt":
        return [
            ["self_attn.q_proj"],
            ["self_attn.k_proj"],
            ["self_attn.v_proj"],
            ["self_attn.out_proj"],
            ["fc1"],
            ["fc2"],
        ]
    elif model.config.model_type == "phi3":
        return [["self_attn.qkv_proj"], ["self_attn.o_proj"], ["mlp.gate_up_proj"], ["mlp.down_proj"]]
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))

def find_sublayers(module, layers=(nn.Conv2d, nn.Linear)):
    res = {}
    for name, layer in module.named_modules():
        if isinstance(layer, layers):
            res[name] = layer
    return res


def get_layers(model):
    if model.config.model_type in (*LLAMA_LIKE, "phi3"):
        return model.model.layers
    elif model.config.model_type.lower() in FALCON_TYPES:
        return model.transformer.h
    elif model.config.model_type == "opt":
        return model.model.decoder.layers
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))

@torch.no_grad()
def get_inps(
    model: PreTrainedModel,
    data: Sequence,
    model_seqlen: int,
    devices: Sequence[torch.device],
    offload_activations: bool,
) -> Tuple[Sequence[torch.Tensor], Dict]:
    """
    mocks model launch to collect inputs to the first model layer
    :returns: a list of torch tensors with activations for each device in args.devices.
    Each tensor has shape [nsample_per_device, seq_len, hid_size]
    """
    print("catching layer inputs from data", flush=True)
    layers = get_layers(model)
    device = devices[0] if not offload_activations else torch.device("cpu")


    emb = model.get_input_embeddings()
    emb_device = emb.weight.device
    if emb_device.type != "cuda":
        emb = emb.to(device)
        # opt has other embeddings
        if model.config.model_type == "opt":
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
            if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(device)
    device = emb.weight.device  # now default device is the one where the embeddings are.
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layer_device = next(layers[0].parameters()).device
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    nsamples_per_device = (len(data) - 1) // len(devices) + 1
    inps = [
        torch.zeros(
            (min(nsamples_per_device, len(data) - i * nsamples_per_device), model_seqlen, model.config.hidden_size),
            dtype=dtype,
            device=devices[i] if not offload_activations else "cpu",
            pin_memory=offload_activations,
        )
        for i in range(len(devices))
    ]
    forward_arg_names = ["attention_mask", "position_ids", "position_embeddings"]
    if model.config.model_type.lower() in FALCON_TYPES:
        forward_arg_names.append("alibi")

    cache = {"i": 0, "alibi": None}

    class CatcherExit(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"] // nsamples_per_device][cache["i"] % nsamples_per_device] = inp
            cache["i"] += 1
            for forward_arg_name in forward_arg_names:
                cache[forward_arg_name] = kwargs.get(forward_arg_name)
            raise CatcherExit()

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch_inps in data:
        try:
            if isinstance(batch_inps, (list, tuple)):
                batch_inps, *_ = batch_inps
            batch_inps = batch_inps.to(device)
            # call model.forward to trigger the Catcher
            model(batch_inps, attention_mask=torch.ones_like(batch_inps,
                                                                 dtype=torch.long, device=device))
        except CatcherExit:
            pass  # exit after catcher finished without running the rest of the model layers

    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].to(layer_device)
    model.get_input_embeddings().to(emb_device)
    if model.config.model_type == "opt":
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(emb_device)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(emb_device)
    torch.cuda.empty_cache()

    forward_args = {k: cache[k] for k in forward_arg_names}
    assert cache["i"] == sum(len(inp_tensor) for inp_tensor in inps), "internal error: found empty rows in inps"
    return inps, forward_args

@torch.no_grad()
def update_outs(
    layer: nn.Module, inps_tensor: torch.Tensor, outs_tensor: torch.Tensor, compute_mse: bool, **forward_args
) -> Sequence[float]:
    """
    Update outs_tensor with new activations and optionally compute sample-wise mse loss with previous activations
    :param layer: transformer layer with one or more linear layer to be quantized
    :param inps_tensor: a tensor of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs_tensor: a tensor to write output activations into, [nsamples_per_device, seq_len, hidden_size]
    :note: outs_tensor must contain previous activations with which to compute MSE loss
    :param compute_mse: if True, return a list of sample-wise mse losses; if False, return an empty sequence
    :param forward_args: additional keyword arguments, e.g. attention mask
    :returns: a list of mean squared errors for each sequence
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    out_losses = []
    for j in trange(len(inps_tensor), desc="calc outs after quantization", leave=False):
        outs_batch = layer(inps_tensor[j].to(device).unsqueeze(0), **forward_args)[0]
        if compute_mse:
            batch_size = outs_batch.shape[0]
            outs_batch_loss = (
                (outs_batch - outs_tensor[j].to(device)).float().square().view(batch_size, -1).mean(dim=-1)
            )
            outs_batch_loss /= outs_batch.float().square().view(batch_size, -1).mean(dim=-1).clamp(min=1e-6)
            outs_batch_loss = outs_batch_loss.mean()
            out_losses.append(outs_batch_loss.item())
        outs_tensor[j].copy_(outs_batch.reshape_as(outs_tensor[j]), non_blocking=True)
    return out_losses

def move_to_device(data, device):
    """
    递归地将 `torch.Tensor` 或包含张量的结构迁移到目标设备。
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    elif isinstance(data, (tuple, list)):
        # 如果是元组或列表，递归处理其每个元素
        return type(data)(move_to_device(item, device) for item in data)
    elif isinstance(data, dict):
        # 如果是字典，递归处理其值
        return {key: move_to_device(value, device) for key, value in data.items()}
    else:
        # 非张量类型保持不变
        return data

@torch.no_grad()
def update_outs_parallel(
    devices: Sequence[torch.device],
    layer: nn.Module,
    inps: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
    compute_mse: bool,
    **forward_args,
) -> Sequence[float]:
    """Parallel version of update_outs_and_compute_losses; works on lists of input/output tensors"""
    layer_replicas = torch.nn.parallel.replicate(layer, devices=devices, detach=True)
    funcs_by_device = [update_outs for _ in devices]
    inputs_by_device = []
    kwargs_by_device = []
    for i in range(len(devices)):
        inputs_by_device.append((layer_replicas[i], inps[i], outs[i], compute_mse))
        kwargs_by_device.append(
            {
                k: move_to_device(v, devices[i])
                for k, v in forward_args.items()
            }
        )
    out_losses_by_device: Sequence[Sequence[float]] = torch.nn.parallel.parallel_apply(
        funcs_by_device, inputs_by_device, kwargs_by_device, devices=devices
    )
    return list(chain(*out_losses_by_device))

class _LayerWrapperThatAccumulatesXTX(nn.Module):
    def __init__(self, layer: nn.Module, aq_handler: AQEngine):
        super().__init__()
        self.wrapped_layer, self.aq_handler = layer, aq_handler

    def forward(self, input, *args, **kwargs):
        self.aq_handler.add_batch(input)
        return self.wrapped_layer(input, *args, **kwargs)

@torch.no_grad()
def init_aq_engines(
    layer: nn.Module,
    names: Sequence[str],
    inps_tensor: torch.Tensor,
    outs_tensor: torch.Tensor,
    **forward_args: Dict[str, Any],
) -> Dict[str, AQEngine]:
    """
    Create a dictionary of AQUtil instances for each quantized layer;
    Run forward pass on each sample in inps_tensor; write output activations to outs_tensor (in-plance)
    Accumulate XTX to each one of aq_handlers
    :param layer: transformer layer with one or more linear layer to be quantized
    :param names: a list/tuple of string names for linear sub-layers inside :layer: that shall be quantized
    :param inps_tensor: a tensor of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs_tensor: a tensor to write output activations into, [nsamples_per_device, seq_len, hidden_size]
    :param forward_args: additional keyword arguments, e.g. attention mask
    :returns: a dictionary where keys are full layer names and values are AQUtil instances ready to run .quantize
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    all_sublayers = find_sublayers(layer)
    subset = {name: all_sublayers[name] for name in names}
    assert len(subset) > 0
    aq_handlers = {}
    for sublayer_name in subset:
        aq_handlers[sublayer_name] = AQEngine(subset[sublayer_name])

    # wrap all quantized sub-layers with a wrapper that accumulates inputs on forward
    # note: the code below uses wrappers instead of hooks because hooks cause bugs in multi-gpu code
    wrapped_layer_to_hander = {aq_handler.layer: aq_handler for aq_handler in aq_handlers.values()}
    for module in list(layer.modules()):
        for child_name, child in list(module.named_children()):
            if child in wrapped_layer_to_hander:
                setattr(module, child_name, _LayerWrapperThatAccumulatesXTX(child, wrapped_layer_to_hander[child]))

    # compute output activations and accumulate XTX
    for j in trange(len(inps_tensor), desc="calc outs before quantization", leave=False):
        outs_tensor[j].copy_(
            layer(inps_tensor[j].to(device).unsqueeze(0), **forward_args)[0].view_as(outs_tensor[j]), non_blocking=True
        )

    # remove wrappers
    for module in list(layer.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, _LayerWrapperThatAccumulatesXTX):
                setattr(module, child_name, child.wrapped_layer)
    return aq_handlers

@torch.no_grad()
def init_aq_engines_parallel(
    devices: Sequence[torch.device],
    layer: nn.Module,
    names: Sequence[str],
    inps: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
    **forward_args,
):
    """Parallel version of init_aq_engines; works on lists of input/output tensors"""
    layer_replicas = torch.nn.parallel.replicate(layer, devices=devices, detach=True)
    layer_replicas[0] = layer  # this ensures that aq_handlers returned by 0-th replica operate on the main layer
    funcs_by_device = [init_aq_engines for _ in devices]
    inputs_by_device = []
    kwargs_by_device = []
    for i in range(len(devices)):
        inputs_by_device.append((layer_replicas[i], names, inps[i], outs[i]))
        kwargs_by_device.append(
            {
                k: move_to_device(v, devices[i])
                for k, v in forward_args.items()
            }
        )
    aq_handles_by_device: Sequence[Dict[str, AQEngine]] = torch.nn.parallel.parallel_apply(
        funcs_by_device, inputs_by_device, kwargs_by_device, devices=devices
    )
    aq_handlers = aq_handles_by_device[0]
    for key, aq_handler in aq_handlers.items():
        replica_handlers = [device_aq_handlers[key] for device_aq_handlers in aq_handles_by_device]
        replica_nsamples = [replica_handler.nsamples for replica_handler in replica_handlers]
        total_nsamples = sum(replica_nsamples)
        aq_handler.XTX = sum(
            (replica_handlers[i].XTX * (replica_nsamples[i] / total_nsamples)).to(devices[0], non_blocking=True)
            for i in range(len(devices))
        )
        aq_handler.nsamples = total_nsamples
    return aq_handlers



def aqlm_pipeline(
    model: PreTrainedModel,
    train_dataset: List[Tuple[torch.Tensor, torch.Tensor]],
    vali_dataset: List[Tuple[torch.Tensor, torch.Tensor]],
    args: edict,
):
    train_dataset = [x[0] for x in train_dataset]
    vali_dataset = [x[0] for x in vali_dataset]
    
    args.num_codebooks = 2
    args.nbits_per_codebook = 8
    args.in_group_size = 8
    args.out_group_size = 1
    args.codebook_value_nbits = 16
    args.codebook_value_num_groups = 1
    args.init_max_iter = 100
    args.init_max_points_per_centroid = None

    args.scale_nbits = 0
    args.lr = 1e-4
    args.beam_size = 1
    args.max_epochs = 1000
    args.relative_mse_tolerance = None
    args.steps_per_epoch = 100
    args.logger.info_frequency = 40
    args.finetune_max_epochs = 5
    args.finetune_early_stop = 3
    args.finetune_lr = 1e-5
    args.finetune_batch_size = 16
    args.finetune_adam_beta1 = 0.9
    args.finetune_adam_beta2 = 0.95
    args.use_checkpointing = False
    args.local_batch_size = None
    args.offload_activations = False
    args.on_save = args.save_quant_dir

    args.model_seqlen= args.training_seqlen
    args.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    # TODO:  这个是一个很神秘的变量，我没有找到它应该表示什么
    args.true_sequential =True
    args.resume = None
    args.skip_out_loss = None


    inps, forward_args = get_inps(model, train_dataset, args.model_seqlen, args.devices, False)
    outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in inps]
    run_validation = True
    val_inps, _ = get_inps(model, vali_dataset, args.model_seqlen, args.devices, False)
    val_outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in val_inps]
    
    use_cache = model.config.use_cache
    num_codebooks = args.num_codebooks

    model.config.use_cache = False

    quantizers = {}
    overall_bits = 0
    number_of_quantized_params = 0
    layers = get_layers(model)

    for layer_index in range(len(layers)):
        args.logger.info(f"\n---------------- Layer {layer_index} of {len(layers)} ----------------")
        stats_payload = {}
        start_time = time.time()

        # quantized layer will return there
        layer_device_original = next(layers[layer_index].parameters()).device
        # backup layer dtype
        layer_dtype_original = next(layers[layer_index].parameters()).dtype
        args.logger.info(f"{layer_device_original=}")
        layer = layers[layer_index].to(args.devices[0])
        for k, v in forward_args.items():
            forward_args[k] = v.to(args.devices[0]) if isinstance(v, torch.Tensor) else v

        if args.true_sequential:
            sequential = get_sequential_groups(model)
        else:
            sequential = [list(find_sublayers(layer).keys())]

        loaded_layer = False
        if args.resume:
            assert args.save_quant_dir is not None, "using --resume requires a --save path to resume from"
            layer_save_path = os.path.join(args.save_quant_dir, f"{layer_index}.pth")
            if os.path.exists(layer_save_path):
                args.logger.info(f"Loading layer {layer_index} from {layer_save_path}")
                layer = torch.load(layer_save_path, map_location=args.devices[0])
                loaded_layer = True

        # prepare validation  outputs
        if run_validation and not loaded_layer:  # note: if we skip validation, val_outs will still be updated later
            if len(args.devices) == 1:
                assert len(val_inps) == len(val_outs) == 1
                update_outs(layer, val_inps[0], val_outs[0], compute_mse=not args.skip_out_loss, **forward_args)
            else:
                update_outs_parallel(
                    args.devices, layer, val_inps, val_outs, compute_mse=not args.skip_out_loss, **forward_args
                )

        for names in sequential:
            if loaded_layer:
                args.logger.info("Skipping quantization: loaded a previously quantized layer")
                break
            if len(args.devices) == 1:
                assert len(inps) == len(outs) == 1  # number of per-device inputs/outputs
                aq_handlers = init_aq_engines(
                    layer,
                    [
                        name
                        for name in names
                        if ((".gate" not in name.lower()) or ("mixtral" not in model.config.model_type.lower()))
                    ],
                    inps[0],
                    outs[0],
                    **forward_args,
                )
            else:
                aq_handlers = init_aq_engines_parallel(
                    args.devices,
                    layer,
                    [
                        name
                        for name in names
                        if ((".gate" not in name.lower()) or ("mixtral" not in model.config.model_type.lower()))
                    ],
                    inps,
                    outs,
                    **forward_args,
                )
            for sublayer_name in aq_handlers.keys():
                args.logger.info(f"Quantizing module {sublayer_name} of layer {layer_index} shape is {aq_handlers[sublayer_name].layer.weight.shape}")
                if "mixtral" in model.config.model_type.lower() and args.mix_compression:
                    assert "mixtral" in model.config.model_type.lower()
                    if "self_attn" in sublayer_name.lower():
                        args.num_codebooks = 2 * num_codebooks
                    else:
                        args.num_codebooks = num_codebooks
                    args.logger.info(sublayer_name.lower(), " mixtral num codebooks", args.num_codebooks)
                quantized_weight = aq_handlers[sublayer_name].quantize(args=args, verbose=False)

                with torch.no_grad():
                    assert aq_handlers[sublayer_name].layer.weight in set(
                        layer.parameters()
                    )  # test that this is not a replica

                    new_linear = QuantizedLinear(quantized_weight, aq_handlers[sublayer_name].layer.bias)
                    if args.use_checkpointing:
                        new_linear.use_checkpoint = True
                        args.logger.info("ENABLED CHECKPOINTING FOR", sublayer_name)
                    found_original = False
                    for submodule in layer.modules():
                        for child_name, child_module in submodule.named_children():
                            if child_module is aq_handlers[sublayer_name].layer:
                                setattr(submodule, child_name, new_linear)
                                found_original = True  # note: do not break to handle tied layers

                    assert found_original, f"could not find {sublayer_name}"

                weight_avg_bits = quantized_weight.estimate_nbits_per_parameter()
                overall_bits += int(weight_avg_bits * torch.numel(aq_handlers[sublayer_name].layer.weight.data))
                number_of_quantized_params += torch.numel(aq_handlers[sublayer_name].layer.weight.data)
                args.logger.info("curent_avg_bits", overall_bits / number_of_quantized_params)
                quantizers["model.layers.%d.%s" % (layer_index, sublayer_name)] = ()  # to be updated

            del aq_handlers
            assert not loaded_layer

            args.logger.info("PREPARING TO FINETUNE")
            args.logger.info(layer)
            layer = layer.to(dtype=torch.float32)
            with using_tf32(enabled=True):
                layer = finetune_groupwise(
                    layer=layer,
                    train_inps=inps,
                    train_outs=outs,
                    args=args,
                    valid_inps=val_inps,
                    valid_outs=val_outs,
                    **forward_args,
                )
            layer = layer.to(dtype=layer_dtype_original)
            args.logger.info("FINISHED FINETUNING")

        if args.save_quant_dir and not loaded_layer:
            os.makedirs(args.save_quant_dir, exist_ok=True)
            layer_save_path = os.path.join(args.save_quant_dir, f"{layer_index}.pth")
            args.logger.info(f"Saving layer {layer_index}... to {layer_save_path}")
            torch.save(layer, layer_save_path)
            # if args.on_save:
            #     exec(args.on_save)  # a callback e.g. to save progress in slurm or similar distributed infrastructure

        should_compute_mse = not (args.skip_out_loss or loaded_layer)
        if len(args.devices) == 1:
            assert len(inps) == len(outs) == 1
            out_losses = update_outs(layer, inps[0], outs[0], compute_mse=should_compute_mse, **forward_args)
        else:
            out_losses = update_outs_parallel(
                args.devices, layer, inps, outs, compute_mse=should_compute_mse, **forward_args
            )
        stats_payload["out_loss"] = torch.mean(torch.Tensor(out_losses)).item()

        if run_validation:
            if len(args.devices) == 1:
                assert len(val_inps) == len(val_outs) == 1
                out_val_losses = update_outs(
                    layer, val_inps[0], val_outs[0], compute_mse=should_compute_mse, **forward_args
                )
            else:
                out_val_losses = update_outs_parallel(
                    args.devices, layer, val_inps, val_outs, compute_mse=should_compute_mse, **forward_args
                )
            stats_payload["out_val_loss"] = torch.mean(torch.Tensor(out_val_losses)).item()

        layers[layer_index] = layer.to(layer_device_original)
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        if run_validation:
            val_inps, val_outs = val_outs, val_inps

        # Logging
        stats_payload["layer_time"] = time.time() - start_time
        stats_payload["Step"] = layer_index
        args.logger.info(f"Layer {layer_index} of {len(layers)} done in {stats_payload['layer_time']:.2f}s")
        if not loaded_layer:
            args.logger.info(stats_payload)

    args.logger.info("=====================\nFinal stats:")
    if args.save_quant_dir:
        torch.save(vars(args), os.path.join(args.save_quant_dir, "args.pt"))
        from llm_aqlm.src.modelutils import save_not_quantized_weights
        save_not_quantized_weights(model, args.save_quant_dir)

    # if args.wandb:
    # save config
    old_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    metadata = {
        "nbits_per_codebook": args.nbits_per_codebook,
        "num_codebooks": args.num_codebooks,
        "out_group_size": args.out_group_size,
        "in_group_size": args.in_group_size,
    }
    from llm_aqlm.convert_to_hf import get_converted_state_dict,update_config

    state_dict, linear_weights_not_to_quantize = get_converted_state_dict(
        old_config, metadata["nbits_per_codebook"], args.save_quant_dir
    )
    torch.save(state_dict, os.path.join(args.save_quant_dir, "pytorch_model.bin"))
    
    new_config_dict = update_config(old_config.to_diff_dict(), metadata, linear_weights_not_to_quantize)
    with open(os.path.join(args.save_quant_dir, "config.json"), "w") as config_file:
        json.dump(new_config_dict, config_file, indent=4)
    
    model = AutoModelForCausalLM.from_pretrained(args.save_quant_dir, trust_remote_code=True, torch_dtype=torch.float16)
    shutil.rmtree(args.save_quant_dir)

    args.logger.info(f"max_cuda_mem_quantize: {round(torch.cuda.max_memory_allocated() / 1e9, 2)}")
    if number_of_quantized_params > 0:  # do not report avg bits if we load all pre-quantized layers via --resume
        args.logger.info(f"Avg_bits: {overall_bits / number_of_quantized_params}")
    model.config.use_cache = use_cache
    args.logger.info(f"quantize: {torch.cuda.max_memory_allocated()=:,}")
    return quantizers,model