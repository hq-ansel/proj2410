import os
import sys
from pathlib import Path
# 将VeOmni的读取代码加入路径
REPO_ROOT = Path(__file__).resolve().parents[1]
QUANT_TASKS_PATH = REPO_ROOT / "VeOmni" / "tasks" / "quantize"
if str(QUANT_TASKS_PATH) not in sys.path:
    sys.path.append(str(QUANT_TASKS_PATH))
HF_HOME = REPO_ROOT / "hf_home"
os.environ.setdefault("HF_HOME", str(HF_HOME))

import accelerate
import torch
from easydict import EasyDict
from transformers import AutoModelForCausalLM, AutoTokenizer

from EfficientQAT.main_block_ap import evaluate

import load_tritonv2_quant  # noqa: E402


print("HF_HOME:", HF_HOME)

def load_quantized_model(path: str):
    model, tokenizer = load_tritonv2_quant.load_tritonv2_quantized_model(
        model_dir=path,
        device="cuda",
        dtype="float16",
    )
    return model, tokenizer


def qtip_model_from_hf_path(path, max_mem_ratio=0.7, device_map=None):
    from opponent.qtip.model.llama import LlamaForCausalLM

    if device_map is None:
        mmap = {
            i: f"{torch.cuda.mem_get_info(i)[1] * max_mem_ratio / (1 << 30)}GiB"
            for i in range(torch.cuda.device_count())
        }
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        device_map = accelerate.infer_auto_device_map(
            model,
            no_split_module_classes=["LlamaDecoderLayer"],
            max_memory=mmap,
        )
    model = LlamaForCausalLM.from_pretrained(
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        device_map=device_map,
    )
    return model


def gptq_model_from_path(path, _wbit=2):
    from gptqmodel import GPTQModel
    model = GPTQModel.load(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


QUANT_PATHS = [
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
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/gptq/w2g128",
    # "/home/ubuntu/data/exp/proj2410/model/Qwen2.5-0.5B",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/w8g128-int8/checkpoints/out",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/w4g128-int4/checkpoints/out",
    "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g128-int2/checkpoints/out",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/w8g128-int8/checkpoints/global_step_614/hf_ckpt",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/w8g128-int8/checkpoints/out",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g128/checkpoints/out",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g128-gradual/checkpoints/out",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g128-gradual-magnitude/checkpoints/out"
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama3-8B/EfficientQAT/w2g128-gradual-magnitude/checkpoints/out",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama3-8B/EfficientQAT/w2g128-gradual/checkpoints/out",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama3-8B/EfficientQAT/w2g128/checkpoints/out",
]
# CUDA_VISIBLE_DEVICES=0,1,2 python -m test.eval_batch
# CUDA_VISIBLE_DEVICES=0,1,2,3  python -m test.eval_batch > exp.logs
# CUDA_VISIBLE_DEVICES=0  python -m test.eval_batch > expfp16.logs
# CUDA_VISIBLE_DEVICES=0 python -m test.eval_batch >> exp_res.logs


def build_eval_args():
    args = EasyDict()
    args["train_param_settings"] = {}
    train_params = args["train_param_settings"]
    train_params["eval_ppl"] = True
    train_params["max_memory"] = "24GB"
    train_params["ppl_seqlen"] = 2048
    train_params["batch_size"] = 1
    train_params["calib_dataset"] = "redpajama"
    train_params["train_size"] = 1
    train_params["val_size"] = 1
    train_params["seed"] = 42
    train_params["eval_tasks"] = "mmlu"
    train_params["num_fewshot"] = 5
    args.eval_batch_size = 8
    args.training_seqlen = 2048
    return args


def load_model_and_tokenizer(quant_path):
    quant_path = str(quant_path)
    if os.path.isfile(os.path.join(quant_path, "quantize_config.json")):
        return load_quantized_model(quant_path)
    if "qtip" in quant_path:
        model = qtip_model_from_hf_path(quant_path)
        tokenizer = AutoTokenizer.from_pretrained(quant_path)
        return model, tokenizer
    if "gptq" in quant_path:
        return gptq_model_from_path(quant_path)
    print("Loading model from", quant_path)
    model = AutoModelForCausalLM.from_pretrained(
        quant_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(quant_path)
    return model, tokenizer


def main():
    for quant_path in QUANT_PATHS:
        quant_model, tokenizer = load_model_and_tokenizer(quant_path)
        args = build_eval_args()
        evaluate(quant_model, tokenizer, args)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
