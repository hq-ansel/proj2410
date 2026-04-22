#!/bin/bash

set -euo pipefail
set -x

# Activate virtual environment
# source /home/ubuntu/data/exp/proj2410/.venv/bin/activate

export TOKENIZERS_PARALLELISM=false
# export PYTHONPATH="/home/ubuntu/data/exp/proj2410:/home/ubuntu/data/exp/proj2410/EfficientQAT:${PYTHONPATH:-}"
# export PYTHONPATH="/home/ubuntu/Projects/Quantization/proj2410:/home/ubuntu/Projects/Quantization/proj2410/EfficientQAT:${PYTHONPATH:-}"
export PYTHONPATH="/home/ubuntu/Projects/Quantization/proj2410/VeOmni:/home/ubuntu/Projects/Quantization/proj2410/EfficientQAT:${PYTHONPATH:-}"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Pre-compile CUDA extensions to avoid multi-process lock contention
python -c "from EfficientQAT.core.quantizer.kernel.fake_quant import fake_quant_ste" 2>/dev/null || true

NNODES=${NNODES:=1}
NPROC_PER_NODE=${NPROC_PER_NODE:=$(nvidia-smi --list-gpus | wc -l)}
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=127.0.0.1}
MASTER_PORT=${MASTER_PORT:=12345}
additional_args=${additional_args:-}

if [[ "$NNODES" == "1" ]]; then
  additional_args="$additional_args --standalone"
else
  additional_args="--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
fi

torchrun \
  --nnodes=$NNODES \
  --nproc-per-node=$NPROC_PER_NODE \
  --node-rank=$NODE_RANK \
  $additional_args "$@" 2>&1 | tee log.txt
exit ${PIPESTATUS[0]}
