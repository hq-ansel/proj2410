#!/usr/bin/env bash
set -euo pipefail

# Get the repository root directory
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Set environment variables
export HF_HOME="${HF_HOME:-$REPO_ROOT/hf_home}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# Parse arguments for evaluation paths
if [ "$#" -gt 0 ]; then
    # Join arguments with commas
    EVAL_QUANT_PATHS=$(IFS=,; echo "$*")
    export EVAL_QUANT_PATHS
elif [[ -n "${EVAL_QUANT_PATHS:-}" ]]; then
    echo "Using EVAL_QUANT_PATHS from environment: $EVAL_QUANT_PATHS"
else
    echo "Error: No model paths provided."
    echo "Usage: $0 <path1> [path2 ...]"
    exit 1
fi

echo "=================================================="
echo "Starting Evaluation"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Evaluating Paths: $EVAL_QUANT_PATHS"
echo "=================================================="

# Ensure we are in the repo root
cd "$REPO_ROOT"

# Run the evaluation script
python -m test.eval_batch
