# -*- coding: utf-8 -*-
"""
本脚本用于批量下载主流大模型（如 Qwen, LLaMA, Baichuan, ChatGLM 等），支持 transformers 和 modelscope 两种方式。
模型会自动保存到指定目录，已存在则跳过。

依赖：
    pip install transformers modelscope torch
"""
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
import os

# 可选：如需用 modelscope 下载 LLaMA 系列
try:
    from modelscope import snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False

# ===================== transformers 下载 =====================
TRANSFORMERS_MODELS = [
    # Qwen 系列
    'Qwen/Qwen2.5-7B',
    'Qwen/Qwen2.5-14B',
    # 'Qwen/Qwen3-8B',
    # LLaMA 系列
    # 'meta-llama/Llama-2-7b-hf',
    # 'meta-llama/Llama-2-13b-hf',
    # 'meta-llama/Llama-2-70b-hf',
    # 'meta-llama/Meta-Llama-3-8B',
]

SAVE_DIR = "/home/ubuntu/data/exp/proj2410/model"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_transformers_models(model_list, save_dir):
    for path in model_list:
        save_path = os.path.join(save_dir, path.split("/")[-1])
        if os.path.exists(save_path):
            print(f"[Transformers] {save_path} 已存在，跳过。")
            continue
        print(f"[Transformers] 正在下载 {path} ...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForCausalLM.from_pretrained(path)
            print(f"[Transformers] 保存到 {save_path}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        except Exception as e:
            print(f"[Transformers] 下载 {path} 失败: {e}")

# ===================== modelscope 下载 =====================
MODELSCOPE_MODELS = [
    # LLaMA2/3 系列
    # 'llama2-13b',
    # 'llama2-70b',
    # 'llama3-8b',
    # Qwen 系列
    'qwen3-8b',
]

MODELSCOPE_MODEL_IDS = {
    'llama2-13b': 'modelscope/Llama-2-13b-ms',
    'llama2-70b': 'AI-ModelScope/Llama-2-70b-hf',
    'llama3-8b': 'LLM-Research/Meta-Llama-3-8B',
    'qwen3-8b': 'Qwen/Qwen3-8B',
}

MODELSCOPE_SAVE_DIR = os.path.join(SAVE_DIR, "modelscope")
os.makedirs(MODELSCOPE_SAVE_DIR, exist_ok=True)

def download_modelscope_models(model_list, save_dir):
    if not MODELSCOPE_AVAILABLE:
        print("未安装 modelscope，跳过 modelscope 下载。")
        return
    for key in model_list:
        model_id = MODELSCOPE_MODEL_IDS.get(key, key)
        if model_id is None:
            print(f"[ModelScope] 未找到模型ID: {key}，跳过。")
            continue
        save_path = os.path.join(save_dir, key)
        if os.path.exists(save_path):
            print(f"[ModelScope] {save_path} 已存在，跳过。")
            continue
        print(f"[ModelScope] 正在下载 {model_id} ...")
        try:
            snapshot_download(
                model_id, 
                cache_dir=save_path,
                ignore_patterns=[
                    "*.bin",
                    "*.pth",
                    "*.pt",
                ], 
                revision=None)
            print(f"[ModelScope] 保存到 {save_path}")
        except Exception as e:
            print(f"[ModelScope] 下载 {model_id} 失败: {e}")

if __name__ == "__main__":
    print("==== Transformers 主流模型批量下载 ====")
    download_transformers_models(TRANSFORMERS_MODELS, SAVE_DIR)
    print("==== ModelScope LLaMA 系列批量下载 ====")
    download_modelscope_models(MODELSCOPE_MODELS, MODELSCOPE_SAVE_DIR) 