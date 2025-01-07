import random
from typing import Callable, Dict, List


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from EfficientQAT.datautils_block import get_wikitext2


# 
def stat_weight_rank(weight:torch.Tensor):
    """
    weight应该至少是float32
    计算奇异值分布，给出奇异值的最大值与最小值
    且给出大于最大值*0.01的个数与比例
    """
    res_dict={
        "sigma_max":0,
        "sigma_min":0,
        "sigma_1percent_num":0,
        "sigma_1percent_ratio":0,
    }
    weight = weight
    s = torch.linalg.svdvals(weight)
    res_dict["sigma_max"] = s.max().item()
    res_dict["sigma_min"] = s.min().item()
    res_dict["sigma_1percent_num"] = (s>s.max()*0.01).sum().item()
    res_dict["sigma_1percent_ratio"] = res_dict["sigma_1percent_num"]/s.shape[0]
    return res_dict


# 

def block_inp_out_hook(metric_fn:Callable,result:Dict,layer_name:str):
    def get_block_import(module,inp, out):
        hidden_states = inp[0]
        hidden_states_out = out[0]
        block_import = metric_fn(hidden_states,hidden_states_out)
        result[layer_name] = block_import
    return get_block_import

        
def hidden_state_similarity(hidden_states:torch.Tensor,hidden_states_out:torch.Tensor)->torch.Tensor:
    """
    Calculate the similarity between the input and output hidden states of a transformer layer.
    """
    output = F.cosine_similarity(hidden_states,hidden_states_out,dim=-1)
    return output.mean().item()
    
@torch.no_grad()
def get_block_importance(model: AutoModelForCausalLM,
                        tokenizer: AutoTokenizer,
                        input_text:List[str],):
    input_ids = tokenizer(input_text, return_tensors='pt',
                          padding=True, truncation=True, max_length=2048)
    hook_list = []
    metric_dict = {}

    device = next(model.parameters()).device
    for n,m in model.named_modules():
        if isinstance(m,Qwen2DecoderLayer):
            hook =  m.register_forward_hook(block_inp_out_hook(
                metric_fn=hidden_state_similarity,
                result=metric_dict,
                layer_name=n
            ))
            hook_list.append(hook)
    input_ids = {key: value.to(device) for key, value in input_ids.items()}
    model(**input_ids)

    for hook in hook_list:
        hook.remove()
    return metric_dict


def linear_min_max_hook(metric_fn:Callable,result:Dict,layer_name:str):
    def get_linear_info(module,inp, out):
        weight = module.weight.data
        _min,_max = weight.min().item(),weight.max().item()
        result[layer_name] = (_min,_max,stat_weight_rank(weight.float()))
    return get_linear_info

def linear_input_min_max_hook(metric_fn:Callable,result:Dict,layer_name:str):
    def get_linear_info(module,inp, out):
        shape = inp[0].shape
        weight = inp[0].reshape(-1,shape[-1])
        _min,_max = weight.min().item(),weight.max().item()
        result[layer_name] = (_min,_max,stat_weight_rank(weight.float()))
    return get_linear_info


@torch.no_grad()
def test_linear_min_max(model: AutoModelForCausalLM,
                        tokenizer: AutoTokenizer,
                        input_text:List[str],):
    input_ids = tokenizer(input_text, return_tensors='pt',
                          padding=True, truncation=True, max_length=2048)
    hook_list = []
    metric_dict = {}
    device = next(model.parameters()).device
    for n,m in model.named_modules():
        if isinstance(m,nn.Linear):
            # hook =  m.register_forward_hook(linear_min_max_hook(
            hook =  m.register_forward_hook(linear_input_min_max_hook(
                metric_fn=lambda x:x,
                result=metric_dict,
                layer_name=n
            ))
            hook_list.append(hook)
    input_ids = {key: value.to(device) for key, value in input_ids.items()}
    model(**input_ids)

    for hook in hook_list:
        hook.remove()
    return metric_dict

if __name__ =="__main__":
    # CUDA_VISIBLE_DEVICES=3 python -m test.metric.block_importance
    model_path = "/home/ubuntu/data/exp/proj2410/model/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16)
    
    model.to("cuda:0")

    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    text = "\n\n".join(traindata['text'])
    text_list = []
    text_len= 2048
    for i in range(1):
        i = random.randint(0,len(text)-1-text_len)
        text_list.append(text[i:i+text_len])
    text = text_list
    print(f"text length: {len(text)} text len {len(text[0])}")
    # block_importance = get_block_importance(model,tokenizer,text)
    # print(block_importance)
    linear_min_max = test_linear_min_max(model,tokenizer,text)
    for k,v in linear_min_max.items():
        _min, _max, _rank_stat = v
        print(f"{k}: {_min}, {_max} {_rank_stat}")
    pass