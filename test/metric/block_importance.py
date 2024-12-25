import random
from typing import Callable, Dict, List


import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from EfficientQAT.datautils_block import get_wikitext2

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
    for i in range(32):
        i = random.randint(0,len(text)-1-text_len)
        text_list.append(text[i:i+text_len])
    text = text_list
    print(f"text length: {len(text)} text len {len(text[0])}")
    block_importance = get_block_importance(model,tokenizer,text)
    print(block_importance)
    pass