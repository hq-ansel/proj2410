#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
export NPROC_PER_NODE=4

parallel=false
hold=false
for arg in "$@"; do
  case "$arg" in
    --parallel)
      parallel=true
      ;;
    --hold)
      hold=true
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

# 检查GPU显存是否充足的函数（要求所有GPU空闲显存 > 90%）
check_gpu_memory() {
    local threshold=90
    local gpu_ids="$1"

    IFS=',' read -ra GPU_ARRAY <<< "$gpu_ids"
    for gpu_id in "${GPU_ARRAY[@]}"; do
        # 获取空闲显存比例 (使用nvidia-smi)
        local free_mem_ratio
        free_mem_ratio=$(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | awk -F',' '{print ($1/$2)*100}')
        if [ -z "$free_mem_ratio" ]; then
            return 1
        fi

        # 如果有任何一个GPU空闲比例 < 阈值，返回失败
        if (( $(echo "$free_mem_ratio < $threshold" | bc -l) )); then
            return 1
        fi
    done

    # 所有GPU都满足条件
    return 0
}

# 显示60秒进度条（固定区域刷新）
show_progress_bar() {
    local duration=600
    local cols=$(tput cols 2>/dev/null || echo 80)
    local bar_width=$((cols - 25))
    
    for ((i=0; i<=duration; i++)); do
        local percent=$((i * 100 / duration))
        local filled=$((i * bar_width / duration))
        local empty=$((bar_width - filled))
        
        printf "\r\033[K["
        printf "%${filled}s" | tr ' ' '█'
        printf "%${empty}s" | tr ' ' '░'
        printf "] %3d%% | Next check in %2ds" "$percent" "$((duration - i))"
        sleep 1
    done
    printf "\r\033[K"
}

# 如果开启 --hold，等待GPU显存充足
if "$hold"; then
    echo "Hold mode enabled. Waiting for all GPUs ($CUDA_VISIBLE_DEVICES) to have >90% free memory..."
    
    while true; do
        if check_gpu_memory "$CUDA_VISIBLE_DEVICES"; then
            echo "✓ GPU memory check passed. All GPUs ($CUDA_VISIBLE_DEVICES) have sufficient memory."
            break
        else
            show_progress_bar
        fi
    done
fi

train_cmds=(
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int4.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int4-gradual.yaml"
#   "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2.yaml"
#   "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2-gradual.yaml"
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2-gradual-kd.yaml"
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2-kd.yaml"
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int3-kd.yaml"
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int3-gradual-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int4.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int8.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-gradual-end025.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-gradual-end050.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-gradual-end075.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-int2-gradual.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-int2.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-int2-gradual-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-int2-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int4-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int8-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int4-gradual.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int8-gradual.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/llama2-7B/llama2-7B-int2-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/llama3-8B/llama3-8B-int2-kd.yaml"
#   "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-7B/qwen2-7B-int2-kd.yaml"
  
  # INT3-KD training - ordered by model size (small to large)
#   "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2-kd-multistep.yaml"
#   "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int3-kd-multistep.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-int3-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-7B/qwen2-7B-int3-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/llama2-7B/llama2-7B-int3-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/llama3-8B/llama3-8B-int3-kd.yaml"

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
