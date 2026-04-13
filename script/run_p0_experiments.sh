#!/usr/bin/env bash
# 实验缺口补齐计划 - P0 优先级
# 基于 /home/ubuntu/data/exp/proj2410/docs/实验缺口.md
# 入口: script/train.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "=========================================="
echo "实验缺口补齐计划 - P0 优先级"
echo "=========================================="
echo ""
echo "P0-1: FP16 基线评测 (无需训练)"
echo "P0-2: 7B QAT 训练 + PPL 评测"
echo "P0-3: GPTQ PTQ baseline 对比"
echo ""

# ============================================
# P0-1: FP16 基线评测
# ============================================
echo "=========================================="
echo "P0-1: FP16 基线评测"
echo "=========================================="

# FP16 模型路径
FP16_MODELS=(
  "/home/ubuntu/data/exp/proj2410/model/Qwen2.5-3B"
  "/home/ubuntu/data/exp/proj2410/model/Qwen2.5-7B"
)

for model_path in "${FP16_MODELS[@]}"; do
  model_name=$(basename "$model_path")
  echo "正在评测 FP16 $model_name ..."
  # 直接用 eval.sh 评测 FP16 模型
  EVAL_QUANT_PATHS="$model_path" python -m test.eval_batch
done

echo "FP16 基线评测完成"

# ============================================
# P0-2: 7B QAT 训练 (通过 script/train.sh 入口)
# ============================================
echo ""
echo "=========================================="
echo "P0-2: 7B QAT 训练 (使用 script/train.sh 入口)"
echo "=========================================="

# 7B 训练配置列表 (用于 script/train.sh)
train_cmds_p0=(
  # Uniform INT2 基线
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-7B/qwen2-7B-uniform-int2-3gpus.yaml"
  # Uniform INT2 + KD
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-7B/qwen2-7B-uniform-int2-kd-3gpus.yaml"
  # Seq2Bit INT2 + KD (Gradual)
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-7B/qwen2-7B-seq2bit-kd-3gpus.yaml"
  # Seq2Bit INT2 + KD + GJSD
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-7B/qwen2-7B-seq2bit-kd-gjsd-3gpus.yaml"
)

# 从配置文件提取 output_dir 用于后续评测
get_output_dir() {
    local yaml_file=$1
    python3 -c "import yaml; print(yaml.safe_load(open('$yaml_file'))['train']['output_dir'])" 2>/dev/null
}

resolve_eval_path() {
    local out_dir="$1"
    local checkpoints_dir="$out_dir/checkpoints"
    local quant_cfg="$checkpoints_dir/out/quantize_config.json"
    local quant_type=""

    if [[ -f "$quant_cfg" ]]; then
        quant_type=$(python3 - <<'PY' "$quant_cfg"
import json, sys
try:
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        print((json.load(f) or {}).get('quant_type', ''))
except Exception:
    print('')
PY
)
    fi

    if [[ "$quant_type" == "mixed" && -d "$checkpoints_dir/out_dequant" ]]; then
        echo "$checkpoints_dir/out_dequant"
        return 0
    fi

    if [[ -d "$checkpoints_dir/out" ]]; then
        echo "$checkpoints_dir/out"
        return 0
    fi
    if [[ -d "$checkpoints_dir/out_dequant" ]]; then
        echo "$checkpoints_dir/out_dequant"
        return 0
    fi

    local latest_hf=""
    local latest_step=-1
    if [[ -d "$checkpoints_dir" ]]; then
        for d in "$checkpoints_dir"/global_step_*; do
            [[ -d "$d" ]] || continue
            local step="${d##*_}"
            [[ "$step" =~ ^[0-9]+$ ]] || continue
            if [[ -d "$d/hf_ckpt" ]] && (( step > latest_step )); then
                latest_step=$step
                latest_hf="$d/hf_ckpt"
            fi
        done
    fi
    if [[ -n "$latest_hf" ]]; then
        echo "$latest_hf"
        return 0
    fi

    return 1
}

eval_paths=()

# 提取 eval 路径
for cmd in "${train_cmds_p0[@]}"; do
    read -ra ADDR <<< "$cmd"
    config_file="${ADDR[-1]}"
    if [[ -f "$config_file" ]]; then
        out_dir=$(get_output_dir "$config_file")
        if [[ -n "$out_dir" ]]; then
            eval_paths+=("$out_dir")
        fi
    fi
done

# 执行训练
for cmd in "${train_cmds_p0[@]}"; do
  echo "Running: $cmd"
  bash -c "$cmd"
done

echo "7B QAT 训练完成"

# ============================================
# P0-3: GPTQ PTQ baseline
# ============================================
echo ""
echo "=========================================="
echo "P0-3: GPTQ PTQ baseline 对比"
echo "=========================================="

# 创建 GPTQ 评测脚本
cat > "$REPO_ROOT/script/run_gptq_eval.sh" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MODEL_PATH="${1:-/home/ubuntu/data/exp/proj2410/model/Qwen2.5-7B}"
OUTPUT_DIR="${2:-/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-7B/GPTQ/w2g64}"

echo "Running GPTQ quantization on $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"

# 使用 GPTQModel 进行 PTQ 量化
python - << 'PY'
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gptqmodel import GPTQModel, QuantizationConfig

model_path = sys.argv[1]
output_dir = sys.argv[2]

print(f"Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("Quantizing with GPTQ (W2G64)...")
quant_config = QuantizationConfig(
    bits=2,
    group_size=64,
    desc=False,
    true_sequential=False,
)

quantized_model = GPTQModel.quantize(model, quant_config)

os.makedirs(output_dir, exist_ok=True)
print(f"Saving quantized model to {output_dir}...")
quantized_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("GPTQ quantization complete!")
PY

# 评测 GPTQ 量化后的模型
echo "Evaluating GPTQ quantized model..."
EVAL_QUANT_PATHS="$OUTPUT_DIR" python -m test.eval_batch

echo "GPTQ evaluation complete!"
EOF

chmod +x "$REPO_ROOT/script/run_gptq_eval.sh"

# 运行 GPTQ 评测
bash "$REPO_ROOT/script/run_gptq_eval.sh"

echo ""
echo "=========================================="
echo "所有 P0 优先级实验完成!"
echo "=========================================="