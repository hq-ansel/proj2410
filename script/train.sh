#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE=8

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
    # Keep bar width conservative to avoid line wrapping on narrow terminals.
    local bar_width=$((cols - 36))
    if (( bar_width < 10 )); then
        bar_width=10
    elif (( bar_width > 60 )); then
        bar_width=60
    fi

    for ((i=0; i<=duration; i++)); do
        local percent=$((i * 100 / duration))
        local filled=$((i * bar_width / duration))
        local empty=$((bar_width - filled))

        # Single-line refresh: CR + clear line + redraw.
        printf "\r\033[2K["
        printf "%${filled}s" "" | tr ' ' '#'
        printf "%${empty}s" "" | tr ' ' '-'
        printf "] %3d%% | Next check in %3ds" "$percent" "$((duration - i))"
        sleep 1
    done
    printf "\r\033[2K"
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
  # ===== Active jobs =====
  # Seq2Bit-KD (LLaMA)
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/llama2-7B/llama2-7B-seq2bit-kd.yaml"
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/llama3-8B/llama3-8B-seq2bit-kd.yaml"

  # ===== Presets (commented) =====
  # Baseline / non-KD (Qwen2-0.5B)
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int4.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int8.yaml"

  # Gradual schedules (Qwen2-0.5B / Qwen2-3B)
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2-gradual.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int4-gradual.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int8-gradual.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-gradual-end025.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-gradual-end050.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-gradual-end075.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-int2-gradual.yaml"

  # Seq2Bit / Seq2Bit-KD
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-seq2bit.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-seq2bit-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-seq2bit-kd.yaml"
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-7B/qwen2-7B-seq2bit-kd.yaml"

  # INT2-KD / INT4-KD / INT8-KD
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2-gradual-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int4-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int8-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-int2.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-int2-gradual-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-int2-kd.yaml"
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/llama2-7B/llama2-7B-int2-kd.yaml"
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/llama3-8B/llama3-8B-int2-kd.yaml"
  "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-7B/qwen2-7B-int2-kd.yaml"
  
  # INT3-KD variants
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int3-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int3-gradual-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int2-kd-multistep.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-0.5B/qwen2-05B-int3-kd-multistep.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-int3-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-7B/qwen2-7B-int3-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/llama2-7B/llama2-7B-int3-kd.yaml"
  # "bash ./VeOmni/train.sh ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/llama3-8B/llama3-8B-int3-kd.yaml"

)

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

    # Seq2Bit torch simulated export writes quant_type=mixed.
    # For evaluation, prefer dequantized HF-style checkpoint in this case.
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

    # Fallback to latest HF checkpoint if packed export is not present.
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

# Extract paths first
for cmd in "${train_cmds[@]}"; do
    # Assume the last argument is the config file
    read -ra ADDR <<< "$cmd"
    config_file="${ADDR[-1]}"
    
    if [[ -f "$config_file" ]]; then
        out_dir=$(get_output_dir "$config_file")
        if [[ -n "$out_dir" ]]; then
            eval_paths+=("$out_dir")
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

resolved_eval_paths=()
for out_dir in "${eval_paths[@]}"; do
    if path=$(resolve_eval_path "$out_dir"); then
        resolved_eval_paths+=("$path")
    else
        echo "Warning: No evaluable checkpoint found under $out_dir (expected checkpoints/out or global_step_*/hf_ckpt)."
    fi
done

if [ ${#resolved_eval_paths[@]} -gt 0 ]; then
    echo "Evaluation paths: ${resolved_eval_paths[*]}"
    if [ -f "script/eval.sh" ]; then
        bash script/eval.sh "${resolved_eval_paths[@]}"
    else
        echo "Error: script/eval.sh not found."
        exit 1
    fi
else
    echo "No evaluation paths found."
fi
