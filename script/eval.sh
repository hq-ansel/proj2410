#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -z "${EVAL_QUANT_PATHS:-}" ]]; then
  DEFAULT_QUANT_PATHS=(
    # "$REPO_ROOT/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g128-int2-kd/checkpoints/out"
    "$REPO_ROOT/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g128-gradual-kd/checkpoints/out"
    "$REPO_ROOT/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g128-gradual/checkpoints/out"
    # "$REPO_ROOT/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g128-gradual-end025/checkpoints/out"
    # "$REPO_ROOT/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g128-gradual-end050/checkpoints/out"
    # "$REPO_ROOT/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g128-gradual-end075/checkpoints/out"
  )
  IFS=',' EVAL_QUANT_PATHS="${DEFAULT_QUANT_PATHS[*]}"
fi
export EVAL_QUANT_PATHS
export HF_HOME="${HF_HOME:-$REPO_ROOT/hf_home}"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m test.eval_batch
