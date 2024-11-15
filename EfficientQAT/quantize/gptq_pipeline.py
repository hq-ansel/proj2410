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

from gptqmodel import GPTQModel, QuantizeConfig

def gptq_pipeline(
        model: PreTrainedModel,
        train_dataset: List[Tuple[torch.Tensor, torch.Tensor]],
        args: edict,
)->GPTQModel:
    del model
    w_bit = args.wbits
    quant_config = QuantizeConfig(bits=w_bit, group_size=args.group_size)
    gptq_model = GPTQModel.load(
        args.model,
        quant_config
    )
    train_dataset = [
        {
            "input_ids": x[0].squeeze(0),
            "attention_mask": torch.ones_like(x[0].squeeze(0)),
            }
                      for x in train_dataset]
    gptq_model.quantize(train_dataset)
    return gptq_model
    pass