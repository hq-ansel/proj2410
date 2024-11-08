from typing import List, Tuple

import torch
import torch.nn as nn
import argparse
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM,AutoTokenizer


from EfficientQAT.utils import BlockLossRecorder
from EfficientQAT.quantize.int_linear_real import load_quantized_model,QuantLinear

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/ubuntu/data/exp/proj2410/model/Qwen2.5-0.5B",
        type=str,
        help="path to model"
    )
    parser.add_argument(
        "--out_dir",
        default="/home/ubuntu/data/exp/proj2410/test/plots",
        type=str,
        help="path to output directory"
    )
    parser.add_argument(
        "--max_memory",
        default="24GB",
        type=str,
        help="maximum memory to use"
    )
    parser.add_argument(
        "--ppl_seqlen",
        default=2048,
        type=int,
        help="sequence length for perplexity evaluation"
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="batch size for training"
    )
    parser.add_argument(
        "--plot_type",
        default="loss",
        type=str,
        help="type of plot to generate"
    )
    parser.add_argument(
        "--loss_path",
        default="/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast/loss.csv",
        type=str,
        help="path to loss file"
    )
    parser.add_argument(
        "--loss_contrast_path",
        default="/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-gradual-quant/loss.csv",
        type=str,
        help="path to contrast loss file"
    )
    parser.add_argument(
        "--base_model",
        default="/home/ubuntu/data/exp/proj2410/model/Qwen2.5-0.5B",
        type=str,
        help="path to model"
    )
    parser.add_argument(
        "--compare_model",
        default="/home/ubuntu/data/exp/proj2410/model/Qwen2.5-0.5B",
        type=str,
        help="path to model"
    )
    args = parser.parse_args()
    return args

def plot_loss(loss_path,loss_contrast_path,args):
    recorder = BlockLossRecorder(loss_path)
    def drill_data(_recorder):
        data = []
        for i in _recorder.loss_data.keys():
            loss_data =_recorder.get_loss_data(i)
            _,loss_data = zip(*loss_data)
            data.append(loss_data)
        return data
    pivot_data = drill_data(recorder)
    recorder_contrast = BlockLossRecorder(loss_contrast_path)
    contrast_data = drill_data(recorder_contrast)
    out_dir = args.out_dir
    loss = np.array(pivot_data)
    contrast_loss = np.array(contrast_data)
    # Number of blocks to group in each plot
    blocks_per_plot = 2
    total_blocks = loss.shape[0]
    steps = loss.shape[1]

    # cols = total_blocks // blocks_per_plot // 2
    rows = total_blocks // blocks_per_plot // 2
    fig, axes = plt.subplots(rows, 2, figsize=(10, 3*rows))
    for i in range(total_blocks//blocks_per_plot):
        ax = axes[i//2,i%2]
        for blk in range(i*blocks_per_plot,(i+1)*blocks_per_plot):
            ax.plot(range(steps),loss[blk], label=f"block {blk}")
            ax.plot(range(steps),contrast_loss[blk], label=f"contrast block {blk}")
        ax.set_title(f'Loss for Blocks {i*blocks_per_plot } to {(i+1)*blocks_per_plot-1}')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/loss_plot_interploate.pdf")


def get_model_and_tokenizer(model_path:str) -> Tuple[AutoModelForCausalLM,AutoTokenizer]:
    quantized = "bit" in model_path
    if "bit" in model_path:
        wbits =2
        group_size = 128
        model,tokenizer = load_quantized_model(model_path,
                                                    wbits=wbits,
                                                    group_size=group_size)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model,tokenizer,quantized

def get_raw_layer(model:AutoModelForCausalLM,quantized:bool,layer_idx:int):
    layer = model.model.layers[layer_idx]
    if quantized:
        for n, m in layer.named_modules():
            if isinstance(m, QuantLinear):
                m.use_fake_quantization(del_quant=True,transpose=True)
    return layer

def generate_markdown_table(result):
    """
    将包含层和模块信息的结果字典转换为 Markdown 表格格式。

    Args:
        result (dict): 包含层、模块名以及 diff_max 和 norm_diff 的字典。

    Returns:
        str: 生成的 Markdown 表格字符串。
    """
    md_table = "| Layer | Module | Diff Max | Norm Diff |\n|-------|--------|----------|-----------|\n"

    for idx, layer_data in result.items():
        for module_name, metrics in layer_data.items():
            diff_max = metrics["diff_max"]
            norm_diff = metrics["norm_diff"]
            md_table += f"| {idx} | {module_name} | {diff_max:.4f} | {norm_diff:.4f} |\n"

    return md_table

def main(args):
    base_model,base_tokenizer,base_quantized = get_model_and_tokenizer(args.base_model)
    compare_model,compare_tokenizer,compare_quantized = get_model_and_tokenizer(args.compare_model)
    result= {}
    for idx in range(len(base_model.model.layers)):
        base_layer = get_raw_layer(base_model,base_quantized,idx)
        compare_layer = get_raw_layer(compare_model,compare_quantized,idx)
        result[idx] = {}
        for n,m in   base_layer.named_modules():
            if base_quantized and isinstance(m, QuantLinear):
                result[idx][n] = {}
                base_weight = m.weight.data
                compare_weight = dict(compare_layer.named_parameters())[n].data
                diff = base_weight - compare_weight
                norm_diff = torch.norm(diff)
                result[idx][n]["diff_max"] = torch.max(diff).item()
                result[idx][n]["norm_diff"] = norm_diff.item()
    print(generate_markdown_table(result))

if __name__ == '__main__':
    args = args_parser()
    main(args)