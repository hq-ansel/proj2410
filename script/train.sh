#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
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
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int4.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int8.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2-gradual.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-gradual-end025.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-gradual-end050.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-gradual-end075.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2-gradual-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2-kd.yaml"
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-int2-gradual.yaml"
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-int2.yaml"
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-int2-gradual-kd.yaml"
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-int2-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int4-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int8-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int4-gradual.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int8-gradual.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2-metrics.yaml"
)

get_output_dir() {
    local yaml_file=$1
    python3 -c "import yaml; print(yaml.safe_load(open('$yaml_file'))['train']['output_dir'])" 2>/dev/null
}

eval_paths=()

# Extract paths first
for cmd in "${train_cmds[@]}"; do
    # Assume the last argument is the config file
    read -ra ADDR <<< "$cmd"
    config_file="${ADDR[-1]}"
    
    if [[ -f "$config_file" ]]; then
        out_dir=$(get_output_dir "$config_file")
        if [[ -n "$out_dir" ]]; then
            eval_paths+=("$out_dir/checkpoints/out")
        else
             echo "Warning: Could not extract output_dir from $config_file"
        fi
    else
        echo "Warning: Config file $config_file not found in command: $cmd"
    fi
done


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
  
  if [ "$status" -ne 0 ]; then
      echo "Training failed with status $status"
      exit "$status"
  fi
else
  for cmd in "${train_cmds[@]}"; do
    echo "Running: $cmd"
    bash -c "$cmd"
  done
fi

echo "Training complete. Starting evaluation..."

if [ ${#eval_paths[@]} -gt 0 ]; then
    echo "Evaluation paths: ${eval_paths[*]}"
    if [ -f "script/eval.sh" ]; then
        bash script/eval.sh "${eval_paths[@]}"
    else
        echo "Error: script/eval.sh not found."
        exit 1
    fi
else
    echo "No evaluation paths found."
fi
