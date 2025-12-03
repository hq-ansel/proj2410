import os
import sys
import random
import yaml
os.environ["HF_Home"] = "/home/ubuntu/data/exp/proj2410/hf_home"
# os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import accelerate
from EfficientQAT.quantize.int_linear_real import load_quantized_model
from EfficientQAT.main_block_ap import evaluate
from EfficientQAT.datautils_block import BlockTrainDataset, get_loaders
from EfficientQAT.quantize.crossblockquant import update_dataset
# from template.datautils import *
from easydict import EasyDict
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
import torch
from EfficientQAT.main_block_ap import evaluate

# curl测试是否可以链接到huggingface



def qtip_model_from_hf_path(path, max_mem_ratio=0.7, device_map=None):

    # AutoConfig fails to read name_or_path correctly
    bad_config = AutoConfig.from_pretrained(path)
    is_quantized = hasattr(bad_config, 'quip_params')
    model_type = bad_config.model_type
    if device_map is None:
        mmap = {
            i: f"{torch.cuda.mem_get_info(i)[1]*max_mem_ratio/(1 << 30)}GiB"
            for i in range(torch.cuda.device_count())
        }
        from opponent.qtip.model.llama import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(path,
                                          torch_dtype='auto',
                                          low_cpu_mem_usage=True,
                                          attn_implementation='sdpa')
        device_map = accelerate.infer_auto_device_map(
            model,
            no_split_module_classes=['LlamaDecoderLayer'],
            max_memory=mmap)
    model = LlamaForCausalLM.from_pretrained(path,
                                      torch_dtype='auto',
                                      low_cpu_mem_usage=True,
                                      attn_implementation='sdpa',
                                      device_map=device_map)

    return model


def gptq_model_from_path(path,wbit=2):
    from gptqmodel import GPTQModel, QuantizeConfig 
    model = GPTQModel.load(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model,tokenizer


quant_path_list = [
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/w2g128",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/w2g128dampenloss",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/w2g128gradual",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/w2g128gradual_factor3",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/w2g128gradual_fator1",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/w2g128interativeFreezing",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/w4g128dampenloss",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/w4g128gradual",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama3-8B/EfficientQAT/w2g128",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama3-8B/EfficientQAT/w2g128dampenloss",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama3-8B/EfficientQAT/w2g128gradual",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama3-8B/EfficientQAT/w4g128dampenloss",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama3-8B/EfficientQAT/w4g128gradual",
    # "/home/ubuntu/data/exp/proj2410/model/Llama2-7b",
    # "/home/ubuntu/data/exp/proj2410/model/Llama3-8B",
    # "/home/ubuntu/data/exp/proj2410/model/Llama3.2-1B",
    # "/home/ubuntu/data/exp/proj2410/model/Llama3.2-3B",
    # "/home/ubuntu/data/exp/proj2410/model/Qwen2.5-0.5B",
    # "/home/ubuntu/data/exp/proj2410/model/Qwen2.5-1.5B",
    # "/home/ubuntu/data/exp/proj2410/model/Qwen2.5-3B",
    # "/home/ubuntu/data/exp/proj2410/model/Qwen2.5-7B",
    # "/home/ubuntu/data/exp/proj2410/model/Qwen2.5-14B",
    # "/home/ubuntu/data/exp/proj2410/model/Qwen3-8B",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/w4g128",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/w4g128gradual",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama3-8B/EfficientQAT/w4g128",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama3-8B/EfficientQAT/w2g128interativeFreezing",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama3-8B/EfficientQAT/w4g128gradual",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/progq2bit_cali_w2",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/w2g128gradualfactor2",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/w2g128interativeFreezing"
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/iterative_freezingw4bit",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/progq-train512w2",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/progq-train2048w2",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/progqw2bit_cali_c4",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/EfficientQAT/w2g128"
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-7B/EfficientQAT/w2gs128-gradual-quant"
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/qtip/w2g128",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/qtip/w2g128_tuned"
    "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/gptq/w2g128",
]
# CUDA_VISIBLE_DEVICES=0,1,2 python -m test.eval_batch
# CUDA_VISIBLE_DEVICES=0,1,2,3  python -m test.eval_batch > exp.logs
# CUDA_VISIBLE_DEVICES=0  python -m test.eval_batch > expfp16.logs
# CUDA_VISIBLE_DEVICES=0 python -m test.eval_batch >> exp_res.logs
import re
patterns = r"w(\d+)"
for quant_path in quant_path_list:
    match = re.search(patterns, quant_path).group(1)
    if match is None:
        wbit = 32
    else:
        wbit = int(match)
    if "EfficientQAT" in quant_path:
        quant_model,tokenizer = load_quantized_model(quant_path,wbit,128)
    elif "qtip" in quant_path:
        quant_model = qtip_model_from_hf_path(quant_path)
        tokenizer = AutoTokenizer.from_pretrained(quant_path)
    elif 'gptq' in quant_path:
        quant_model,tokenizer = gptq_model_from_path(quant_path)
    else:
        quant_model = AutoModelForCausalLM.from_pretrained(quant_path,
                                                           torch_dtype=torch.float16,
                                                           device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(quant_path)
        print("Loading model from",quant_path)
    args = EasyDict()
    args["train_param_settings"]={}
    args["train_param_settings"]["eval_ppl"] = False
    # args["train_param_settings"]["eval_tasks"] = ""
    args["train_param_settings"]["max_memory"] = "24GB"
    args["train_param_settings"]["ppl_seqlen"] = 2048
    args["train_param_settings"]["batch_size"] = 1
    args["train_param_settings"]["calib_dataset"] = "redpajama"
    args["train_param_settings"]["train_size"] = 1
    args["train_param_settings"]["val_size"] = 1
    args["train_param_settings"]["seed"] = 42
    # args["train_param_settings"]["eval_tasks"]="piqa,arc_easy,arc_challenge,hellaswag,winogrande"
    args["train_param_settings"]["eval_tasks"]="mmlu"
    args["train_param_settings"]["num_fewshot"]=5
    # args.eval_tasks="xsum,cnn_dailymail,openbookqa,copa,mathqa,rte"
    args.eval_batch_size=8
    args.training_seqlen = 2048
    evaluate(quant_model,tokenizer,args)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
