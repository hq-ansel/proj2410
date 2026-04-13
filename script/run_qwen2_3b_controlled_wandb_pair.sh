#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SH="$ROOT_DIR/VeOmni/train.sh"
TRAIN_PY="VeOmni/tasks/quantize/train.py"
CONFIG_DIR="$ROOT_DIR/VeOmni/tasks/quantize/configs/qwen2-3B"
LOG_DIR="$ROOT_DIR/exp/manual_logs/qwen2_3b_controlled_wandb_20260322"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
FREE_MEM_THRESHOLD="${FREE_MEM_THRESHOLD:-90}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

gpu_ready() {
  local gpu_ids="$1"
  local threshold="$2"
  IFS=',' read -ra gpu_array <<< "$gpu_ids"
  for gpu_id in "${gpu_array[@]}"; do
    local ratio
    ratio=$(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | awk -F',' '{print ($1/$2)*100}')
    if [[ -z "$ratio" ]]; then
      return 1
    fi
    if (( $(echo "$ratio < $threshold" | bc -l) )); then
      return 1
    fi
  done
  return 0
}

wait_for_all_gpus() {
  local gpu_ids="$1"
  local threshold="$2"
  local interval="$3"
  until gpu_ready "$gpu_ids" "$threshold"; do
    log "Waiting for GPUs $gpu_ids to reach free_mem>${threshold}%"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    sleep "$interval"
  done
}

run_job() {
  local config_name="$1"
  local run_tag="$2"
  local master_port="$3"
  local log_path="$LOG_DIR/${run_tag}.log"
  local config_path="$CONFIG_DIR/$config_name"

  wait_for_all_gpus "$CUDA_VISIBLE_DEVICES" "$FREE_MEM_THRESHOLD" "$POLL_INTERVAL"

  log "Starting $run_tag with config=$config_path"
  (
    cd "$ROOT_DIR"
    export CUDA_VISIBLE_DEVICES
    export NPROC_PER_NODE
    export MASTER_PORT="$master_port"
    export WANDB_MODE=online
    bash "$TRAIN_SH" "$TRAIN_PY" "$config_path"
  ) 2>&1 | tee "$log_path"
  local exit_code=${PIPESTATUS[0]}
  log "Finished $run_tag with exit_code=$exit_code log=$log_path"
  return "$exit_code"
}

run_job "qwen2-3B-int2-uniform-controlled-wandb.yaml" "uniform" 12345
run_job "qwen2-3B-int2-gradual-controlled-wandb.yaml" "gradual" 12355
