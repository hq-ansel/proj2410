import shutil
import functools
import time
import os
import copy
import pdb
import gc
import math
from functools import wraps
from collections import defaultdict
from contextlib import contextmanager
from typing import List, Tuple, Dict, Union, Callable
from concurrent.futures import ThreadPoolExecutor
from regex import T
from tqdm import tqdm

import logging
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


from llm_awq.quantize.quantizer import (real_quantize_model_weight,
                                    pseudo_quantize_model_weight)
from llm_awq.utils.utils import simple_dispatch_model
from llm_awq.quantize.auto_scale import auto_scale_block,apply_scale
from llm_awq.quantize.auto_clip import auto_clip_block,apply_clip

def load_awq_model_tokenizer(model_path,
                             w_bit,
                             q_config,
                             max_memory="24GB",
                             ):
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=torch.float16, trust_remote_code=True
            )
    real_quantize_model_weight(
        model, w_bit=w_bit, q_config=q_config, init_only=True
    )

    model.tie_weights()

    # Infer device map
    kwargs = {"max_memory": max_memory} if len(max_memory) else {}
    device_map = infer_auto_device_map(
        model,
        no_split_module_classes=[
            "OPTDecoderLayer",
            "LlamaDecoderLayer",
            "BloomBlock",
            "MPTBlock",
            "Qwen2DecoderLayer",
            "DecoderLayer",
        ],
        **kwargs,
    )
    # Load checkpoint in the model
    # load_quant = os.path.join(model_path, "awq_model.pt")
    load_checkpoint_in_model(
        model,
        checkpoint=model_path,
        device_map=device_map,
        offload_state_dict=True,
    )
    # Dispatch model
    model = simple_dispatch_model(model, device_map=device_map)

    model.eval()


# run_awq from https://github.com/mit-han-lab/llm-awq/blob/5330d6d49be6ed6a141f1fe64c93dab99232cbfd/awq/quantize/pre_quant.py
def get_blocks(model):
    # add qwen2.5
    if model.__class__.__name__ == "LlamaForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif isinstance(model, Qwen2ForCausalLM):
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    else:
        raise NotImplementedError(type(model))
    return layers

def move_embed(model, device):
    # add qwen2.5
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
    elif isinstance(model, Qwen2ForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    else:
        raise NotImplementedError(type(model))

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}



@torch.no_grad()
def run_awq(
    model,
    samples: List[torch.Tensor],
    w_bit: int,
    q_config: Dict,
):
    from llm_awq.utils.module import append_str_prefix, get_op_name

    if "bigcode" in str(model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")

    layers = get_blocks(model)
    # 一次性输入果然炸完了
    # samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp.detach().cpu())
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    for sample in samples:
        try:
            model(sample.to(next(model.parameters()).device))
        except ValueError:  # work with early exit
            pass
    del samples
    layers[0] = layers[0].module  # restore
    # inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    awq_results = {
        "scale": [],
        "clip": [],
    }
    model = model.to("cpu")
    # solve layer by layer
    for i in tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.to("cpu")
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        estimate_memory = sum([
            p.numel() * p.element_size() for p in layer.parameters()
        ])
        print(f"Estimated layer memory usage: {estimate_memory / 1024 ** 3:.2f} GB")
        estimate_memory = sum(
            [p.numel()*2  for p in inps ]
        )
        print(f"Estimated input memory usage: {estimate_memory / 1024 ** 3:.2f} GB")
        # inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        for idx in range(len(inps)):
            inp = inps[idx].half().to(next(layer.parameters()).device)
            inps[idx] = layer(inp, **layer_kwargs)[0].to("cpu")
            del inp
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        # Clear GPU memory
        torch.cuda.empty_cache()
        print(f"Finish solving layer {i}")
        if (
            # auto_scale
            True
        ):  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block(
                layer,
                layer_kwargs,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model, layer) + "."
            )

        # Clear GPU memory
        torch.cuda.empty_cache()

        # if mse_range:
        if True:
            clip_list = auto_clip_block(
                layer,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
            )
            apply_clip(layer, clip_list)
            # append prefix to make names global
            awq_results["clip"] += append_str_prefix(
                clip_list, get_op_name(model, layer) + "."
            )

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    return awq_results


def apply_awq(model, awq_results):
    apply_scale(model, awq_results["scale"])
    apply_clip(model, awq_results["clip"])



def awq_pipline(
        model: PreTrainedModel,
        train_dataset: List[Tuple[torch.Tensor, torch.Tensor]],
        args: edict,
):
    model.to(dtype = torch.float16)
    w_bit = args.wbits
    q_config = {
        "zero_point": True,  # by default True
        "q_group_size": args.group_size,  # whether to use group quantization
    }
    cali_model = copy.deepcopy(model)
    train_dataset = [x[0] for x in train_dataset]
    args.logger.info("Running AWQ...")
    awq_results = run_awq(
        cali_model,
        train_dataset,
        w_bit=w_bit,
        q_config=q_config,)
    del cali_model
    del train_dataset
    gc.collect()
    args.logger.info("Applying AWQ...")
    apply_awq(model, awq_results)
    args.logger.info("Real-quantizing model weight...")
    real_quantize_model_weight(model, w_bit=w_bit, q_config=q_config)
    