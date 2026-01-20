#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1,2,3,4
export NPROC_PER_NODE=4

parallel=false
for arg in "$@"; do
  case "$arg" in
    --parallel)
      parallel=true
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

train_cmds=(
#     "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-gradual.yaml"
#     "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-gradual-end025.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-gradual-end050.yaml"
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-gradual-end075.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int4.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int8.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2-gradual-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int4-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int8-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int4-gradual.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int8-gradual.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2-metrics.yaml"
)

if "$parallel"; then
  pids=()
  for cmd in "${train_cmds[@]}"; do
    echo "Starting: $cmd"
    bash -c "$cmd" &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  exit "$status"
fi

for cmd in "${train_cmds[@]}"; do
  echo "Running: $cmd"
  bash -c "$cmd"
done
