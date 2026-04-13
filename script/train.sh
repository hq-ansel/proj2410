#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Default settings
backbone="veomni"
parallel=false
hold=false
run_p0=false
tp_size=1
pp_size=1
cp_size=1
config_file=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backbone)
      backbone="$2"
      shift 2
      ;;
    --backbone=*)
      backbone="${1#*=}"
      shift
      ;;
    --tp-size)
      tp_size="$2"
      shift 2
      ;;
    --tp-size=*)
      tp_size="${1#*=}"
      shift
      ;;
    --pp-size)
      pp_size="$2"
      shift 2
      ;;
    --pp-size=*)
      pp_size="${1#*=}"
      shift
      ;;
    --cp-size)
      cp_size="$2"
      shift 2
      ;;
    --cp-size=*)
      cp_size="${1#*=}"
      shift
      ;;
    --config)
      config_file="$2"
      shift 2
      ;;
    --config=*)
      config_file="${1#*=}"
      shift
      ;;
    --parallel)
      parallel=true
      shift
      ;;
    --hold)
      hold=true
      shift
      ;;
    --p0)
      run_p0=true
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --backbone {veomni|megatron-lm}  Select training backbone (default: veomni)"
      echo "  --tp-size N                      Tensor parallelism size (default: 1)"
      echo "  --pp-size N                      Pipeline parallelism size (default: 1)"
      echo "  --cp-size N                      Context parallelism size (default: 1)"
      echo "  --config FILE                    Config file (.sh for megatron-lm, .yaml for veomni)"
      echo "  --parallel                       Run training tasks in parallel"
      echo "  --hold                         Wait for GPU memory > 90% before training"
      echo "  --p0                           Run P0 priority experiments"
      echo "  --help, -h                     Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0 --backbone veomni --parallel"
      echo "  $0 --backbone megatron-lm --tp-size 2 --config script/configs/qwen2.5_0.5b.sh"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Use --help for usage information" >&2
      exit 1
      ;;
  esac
done

# Validate backbone
if [[ "$backbone" != "veomni" && "$backbone" != "megatron-lm" ]]; then
  echo "Error: Invalid backbone '$backbone'. Must be 'veomni' or 'megatron-lm'." >&2
  exit 1
fi

echo "Using backbone: $backbone"
if [[ "$backbone" == "megatron-lm" ]]; then
  echo "TP=$tp_size PP=$pp_size CP=$cp_size"
fi

# Check GPU memory
check_gpu_memory() {
    local threshold=90
    local gpu_ids="$1"
    IFS=',' read -ra GPU_ARRAY <<< "$gpu_ids"
    for gpu_id in "${GPU_ARRAY[@]}"; do
        local free_mem_ratio
        free_mem_ratio=$(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | awk -F',' '{print ($1/$2)*100}')
        if [ -z "$free_mem_ratio" ]; then return 1; fi
        if (( $(echo "$free_mem_ratio < $threshold" | bc -l) )); then return 1; fi
    done
    return 0
}

show_progress_bar() {
    local duration=60
    local cols=$(tput cols 2>/dev/null || echo 80)
    local bar_width=$((cols - 30))
    for ((i=0; i<=duration; i++)); do
        local percent=$((i * 100 / duration))
        local filled=$((i * bar_width / duration))
        local empty=$((bar_width - filled))
        printf "\r\033[K"
        printf "%${filled}s" | tr ' ' '█'
        printf "%${empty}s" | tr ' ' '░'
        printf "] %3d%% | Next: %2ds" "$percent" "$((duration - i))"
        sleep 1
    done
    printf "\r\033[K"
}

if "$hold"; then
    echo "Hold: Waiting for GPUs $CUDA_VISIBLE_DEVICES -gt 90% free..."
    while true; do
        if check_gpu_memory "$CUDA_VISIBLE_DEVICES"; then break; else show_progress_bar; fi
    done
    echo "GPU memory OK"
fi

# Build VeOmni command
build_veomni_cmd() {
    echo "bash ./VeOmni/train.sh $1 $2"
}

# Build Megatron-LM command with TP/SP support
build_megatron_cmd() {
    local cfg="${1:-}"
    local sp_flag=""
    
    # Auto-enable SP when TP gt 1
    if [[ $tp_size -gt 1 ]]; then
        sp_flag="--sequence-parallel"
        echo "Auto-enabled --sequence-parallel TP=$tp_size" >&2
    fi
    
    # Load config if provided
    if [[ -n "$cfg" && -f "$cfg" ]]; then
        echo "Loading config: $cfg" >&2
        source "$cfg" 2>/dev/null || true
    fi
    
    # Default paths
    local ckpt="${CHECKPOINT_PATH:-checkpoints/qwen2.5-0.5b-megatron}"
    local data="${DATA_PATH:-$REPO_ROOT/data/mock}"
    local tok="${TOKENIZER_PATH:-$REPO_ROOT/model/Qwen2.5-0.5B}"
    
    # Build command - use printf to avoid escaping issues
    printf "cd %s/Megatron-LM && torchrun --nproc_per_node=%s --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29500 pretrain_gpt.py --tensor-model-parallel-size %s --pipeline-model-parallel-size %s --context-parallel-size %s %s --num-layers 24 --hidden-size 896 --ffn-hidden-size 4864 --num-attention-heads 14 --group-query-attention --num-query-groups 2 --seq-length 2048 --max-position-embeddings 32768 --position-embedding-type rope --rotary-base 1000000 --swiglu --normalization RMSNorm --norm-epsilon 1e-6 --disable-bias-linear --micro-batch-size 1 --global-batch-size 8 --train-iters 1000 --lr 0.0001 --lr-decay-style cosine --min-lr 0.00001 --clip-grad 1.0 --weight-decay 0.1 --bf16 --data-path %s --split 949,50,1 --tokenizer-type HuggingFaceTokenizer --tokenizer-model %s --save %s --save-interval 100 --log-interval 10" \
        "$REPO_ROOT" "$NPROC_PER_NODE" "$tp_size" "$pp_size" "$cp_size" "$sp_flag" "$data" "$tok" "$ckpt"
}

# Define training commands
train_cmds=()

if [[ "$backbone" == "veomni" ]]; then
    if "$run_p0"; then
        echo "P0 experiments VeOmni..."
        train_cmds+=(
            "$(build_veomni_cmd ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-7B/qwen2-7B-uniform-int2-3gpus.yaml)"
            "$(build_veomni_cmd ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-7B/qwen2-7B-uniform-int2-kd-3gpus.yaml)"
        )
    else
        echo "Default experiments VeOmni..."
        train_cmds+=(
            "$(build_veomni_cmd ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-seq2bit-kd-forwardkl-alpha05-3gpus.yaml)"
            "$(build_veomni_cmd ./VeOmni/tasks/quantize/train.py VeOmni/tasks/quantize/configs/qwen2-3B/qwen2-3B-seq2bit-g128-3gpus.yaml)"
        )
    fi
elif [[ "$backbone" == "megatron-lm" ]]; then
    if "$run_p0"; then
        echo "P0 experiments Megatron-LM TP=$tp_size..."
        train_cmds+=("$(build_megatron_cmd "$config_file")")
    else
        echo "Default experiments Megatron-LM Qwen2.5-0.5B TP=$tp_size..."
        train_cmds+=("$(build_megatron_cmd "$config_file")")
    fi
fi

# Execute training
if "$parallel"; then
    pids=()
    for cmd in "${train_cmds[@]}"; do
        echo "Starting: $cmd"
        bash -c "$cmd" &
        pids+=($!)
    done
    status=0
    for pid in "${pids[@]}"; do
        wait $pid || status=1
    done
    [[ $status -ne 0 ]] && exit $status
else
    for cmd in "${train_cmds[@]}"; do
        echo "Running: $cmd"
        bash -c "$cmd"
    done
fi

echo "Training complete!"
