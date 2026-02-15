#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Usage:
#   bash script/eval_seq2bit.sh [QUANT_OUT_DIR] [CUDA_VISIBLE_DEVICES] [LOG_FILE]
# Example:
#   bash script/eval_seq2bit.sh \
#     /home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/seq2bit-g64/checkpoints/out \
#     0,1,2,3 \
#     logs/eval_seq2bit.log

QUANT_OUT_DIR="${1:-/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/seq2bit-g64/checkpoints/out}"
CUDA_IDS="${2:-0,1,2,3,4,5,6,7}"
LOG_FILE="${3:-log_eval_seq2bit_$(date +%Y%m%d_%H%M%S).txt}"

if [[ ! -d "$QUANT_OUT_DIR" ]]; then
  echo "Error: quant out dir not found: $QUANT_OUT_DIR" >&2
  exit 1
fi

source .env
export CUDA_VISIBLE_DEVICES="$CUDA_IDS"

echo "Eval target: $QUANT_OUT_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Log file: $LOG_FILE"

bash script/eval.sh "$QUANT_OUT_DIR" 2>&1 | tee "$LOG_FILE"
