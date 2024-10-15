# Import necessary libraries
import os
import time
from functools import wraps
from pathlib import Path
import json

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from easydict import EasyDict
import numpy as np

from EfficientQAT.quantize.int_linear_real import load_quantized_model
from EfficientQAT.main_block_ap import evaluate
from EfficientQAT.datautils_block import BlockTrainDataset, get_loaders
from EfficientQAT.quantize.crossblockquant import update_dataset
from template.datautils import *

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 开始计时
        result = func(*args, **kwargs)
        end_time = time.time()  # 结束计时
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to execute.")
        return result
    return wrapper

# Main function for parameter input and initialization
@timer
def main():
    # Environment setup
    os.environ["HF_HOME"] = "/home/ubuntu/data/exp/proj2410/hf_home"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # Model paths and loading
    model_path = "/home/ubuntu/data/exp/proj2410/model/Llama2-7b"
    llm_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    abs_test_dict = []

    # Argument setup
    args = EasyDict()
    args.eval_ppl = True
    args.eval_tasks = ""
    args.max_memory = "24GB"
    args.ppl_seqlen = 2048
    args.batch_size = 1
    args.calib_dataset = "redpajama"
    args.train_size = 1
    args.val_size = 1
    args.seed = 42
    args.training_seqlen = 4096

    # Load and compare multiple quantized models
    quant_model_paths = [
        "/home/ubuntu/data/exp/proj2410/quant_model/EfficientQAT/w4gs128/Llama2-7b",
        "/home/ubuntu/data/exp/proj2410/quant_model/EfficientQAT/w2gs128/Llama2-7b",
        "/home/ubuntu/data/exp/proj2410/quant_model/GPTQ/w4gs128/Llama2-7b",
        "/home/ubuntu/data/exp/proj2410/quant_model/GPTQ/w2gs128/Llama2-7b",
    ]
    
    for quant_model_path in quant_model_paths:
        wbits = int(quant_model_path.split("/")[-2].split("w")[1].split("gs")[0])
        group_size = int(quant_model_path.split("/")[-2].split("gs")[1])
        method = quant_model_path.split("/")[-3]
        if method == "EfficientQAT":
            quant_model, tokenizer = load_quantized_model(quant_model_path, wbits, group_size)
        elif method == "GPTQ":
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig 
            if not os.path.exists(quant_model_path):
                Path(quant_model_path).mkdir(parents=True, exist_ok=True)
                quant_model = AutoGPTQForCausalLM.from_pretrained(model_path,quantize_config=BaseQuantizeConfig(bits=wbits, group_size=group_size))
                tokenizer  = AutoTokenizer.from_pretrained(model_path)
                examples = [
                    tokenizer(
                        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
                    )
                ]
                quant_model.quantize(examples)
                quant_model.save_pretrained(quant_model_path)
            else:
                quant_model = AutoGPTQForCausalLM.from_quantized(quant_model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            quant_model=quant_model.model
        # Call block test
        attention_mask, position_ids, fp_train_inps, quant_train_inps = block_test(llm_model, quant_model, tokenizer, args)
        
        # Model output comparison
        mean_abs_error_list, max_abs_error_list, num_outlier_list = model_output_comparison(llm_model, quant_model, fp_train_inps, quant_train_inps, attention_mask, position_ids)
        
        # Print formatted output
        print(f"Quant Model: {quant_model_path}")
        print(f"Mean Absolute Error: {mean_abs_error_list}")
        print(f"Max Absolute Error: {max_abs_error_list}")
        print(f"Number of Outliers: {num_outlier_list/(4096**2)}")
        abs_test_dict.append(
            {
                "quant_model_path": quant_model_path,
                "wbits": wbits,
                "group_size": group_size,
                "method": method,
                "mean_abs_error": mean_abs_error_list.tolist(),
                "max_abs_error": max_abs_error_list.tolist(),
                "num_outlier": (num_outlier_list/(4096**2)).tolist(),
            }
        )
        evaluate(quant_model, tokenizer, args)
    out_dir = "/home/ubuntu/data/exp/proj2410/test/"
    with open(os.path.join(out_dir, "abs_test.json"), "w") as f:
        json.dump(abs_test_dict, f, indent=4)

# Block test function
def block_test(llm_model, quant_model, tokenizer, args):
    trainloader, valloader = get_loaders(
        args.calib_dataset,
        tokenizer,
        args.train_size,
        args.val_size,
        seed=args.seed,
        seqlen=args.training_seqlen,
    )

    fp_train_inps = BlockTrainDataset(1, 4096, 4096, 1, torch.float16)
    quant_train_inps = BlockTrainDataset(1, 4096, 4096, 1, torch.float16)

    layer_idx = 0
    dev = "cuda:0"

    # Catcher class for capturing intermediate outputs
    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None

        def forward(self, inp, **kwargs):
            self.dataset.update_data(self.index, inp.squeeze(0).to('cpu'))
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs["attention_mask"]
            if self.position_ids is None:
                self.position_ids = kwargs["position_ids"]
            raise ValueError

    # Catcher for the original model
    fp_layers = llm_model.model.layers
    quant_layers = quant_model.model.layers
    catcher = Catcher(fp_layers[layer_idx], fp_train_inps)
    iters = len(valloader) // args.batch_size
    llm_model = llm_model.to(dev)
    model = llm_model.model
    model.layers[layer_idx] = catcher
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(i * args.batch_size, (i + 1) * args.batch_size)], dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    attention_mask = model.layers[layer_idx].attention_mask
    position_ids = model.layers[layer_idx].position_ids
    model.layers[layer_idx] = model.layers[layer_idx].module

    # Catcher for the quantized model
    catcher = Catcher(quant_layers[layer_idx], quant_train_inps)
    quant_model = quant_model.to(dev)
    model = quant_model.model
    model.layers[layer_idx] = catcher
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(i * args.batch_size, (i + 1) * args.batch_size)], dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    model.layers[layer_idx] = model.layers[layer_idx].module

    return attention_mask, position_ids, fp_train_inps, quant_train_inps


# Model output comparison function
def model_output_comparison(llm_model, quant_model, fp_train_inps, quant_train_inps, attention_mask, position_ids):
    mean_abs_error_list = np.array([])
    max_abs_error_list = np.array([])
    num_outlier_list = np.array([])
    dev = "cuda:0"

    with torch.no_grad():
        for idx in range(32):
            quant_layer = quant_model.model.layers[idx]
            fp_layer = llm_model.model.layers[idx]
            for quant_data, fp_data in zip(quant_train_inps, fp_train_inps):
                quant_out = quant_layer(quant_data.to(dev), attention_mask=attention_mask, position_ids=position_ids)[0]
                fp_out = fp_layer(fp_data.to(dev), attention_mask=attention_mask, position_ids=position_ids)[0]
                abs_error = torch.abs(quant_out - fp_out)
                mean_abs_error = torch.mean(abs_error)
                max_abs_error = torch.max(abs_error)
                num_outlier = torch.sum(abs_error > 1)
                mean_abs_error_list = np.append(mean_abs_error_list, mean_abs_error.cpu().item())
                max_abs_error_list = np.append(max_abs_error_list, max_abs_error.cpu().item())
                num_outlier_list = np.append(num_outlier_list, num_outlier.cpu().item())
            update_dataset(quant_layer, quant_train_inps, dev, attention_mask, position_ids)
            update_dataset(fp_layer, fp_train_inps, dev, attention_mask, position_ids)

    return mean_abs_error_list, max_abs_error_list, num_outlier_list


if __name__ == "__main__":
    main()