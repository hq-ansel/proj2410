#!/usr/bin/env bash
# GPTQ PTQ 评测脚本
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

MODEL_PATH="${1:-/home/ubuntu/data/exp/proj2410/model/Qwen2.5-7B}"
OUTPUT_DIR="${2:-/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-7B/GPTQ/w2g64}"

echo "=========================================="
echo "GPTQ PTQ 评测"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

# 使用项目自带的 GPTQ 实现
python3 - "$MODEL_PATH" "$OUTPUT_DIR" << 'PYEOF'
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, TensorDataset

# 添加项目路径
sys.path.insert(0, '/home/ubuntu/data/exp/proj2410')
from EfficientQAT.ptq.gptq import GPTQConfig, apply_gptq_to_model

model_path = sys.argv[1]
output_dir = sys.argv[2]

print(f"Loading model from {model_path}...")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型 (使用 device_map 避免 OOM)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Model loaded on: {next(model.parameters()).device}")

# 创建校准数据 - 使用真实的 token IDs
print("Creating calibration data...")
calib_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "In recent years, deep learning has revolutionized many fields.",
    "Natural language processing is a subfield of linguistics.",
    "Machine learning algorithms can identify complex patterns.",
    "Artificial intelligence is transforming the way we live and work.",
    "Quantum computing promises to solve problems beyond classical computers.",
    "The transformer architecture has become ubiquitous in modern AI.",
    "Large language models have shown remarkable capabilities.",
]

# Tokenize 校准数据
calib_inputs = tokenizer(calib_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
calib_input_ids = calib_inputs.input_ids  # [batch, seq_len]

print(f"Calibration data shape: {calib_input_ids.shape}")

# 创建 DataLoader
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

calib_dataset = SimpleDataset(calib_input_ids, calib_inputs.attention_mask)
calib_loader = DataLoader(calib_dataset, batch_size=1)

# GPTQ 量化配置
config = GPTQConfig(
    n_bits=2,
    group_size=64,
    damp=0.01,
    block_size=128,
    act_order=False
)

print("Applying GPTQ quantization (W2G64)...")
quantizers = apply_gptq_to_model(model, config, calib_loader, verbose=True)

# 保存量化模型
os.makedirs(output_dir, exist_ok=True)
print(f"Saving quantized model to {output_dir}...")

# 保存模型权重和配置
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("GPTQ quantization complete!")
print(f"Quantized {len(quantizers)} layers")
PYEOF

# 评测量化后的模型
echo ""
echo "Evaluating quantized model..."
EVAL_QUANT_PATHS="$OUTPUT_DIR" python -m test.eval_batch

echo ""
echo "GPTQ evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"