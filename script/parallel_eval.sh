#!/bin/bash
# 并行评估脚本 - 在不同GPU上评估多个模型

cd /home/ubuntu/data/exp/proj2410
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

# 3B 模型列表 (每个模型单卡评估)
declare -a MODELS=(
    quant_model/Qwen2.5-3B/EfficientQAT/w2g128-gradual/checkpoints/out
    quant_model/Qwen2.5-3B/EfficientQAT/w2g64-gradual-kd/checkpoints/out
    quant_model/Qwen2.5-3B/EfficientQAT/w2g64-int2-kd/checkpoints/out
)

# 启动并行评估
for i in ${!MODELS[@]}; do
    GPU_ID=$i
    MODEL_PATH=${MODELS[$i]}
    
    if [ -d "$MODEL_PATH" ]; then
        echo "Starting eval on GPU $GPU_ID: $MODEL_PATH"
        CUDA_VISIBLE_DEVICES=$GPU_ID EVAL_QUANT_PATHS="$MODEL_PATH" \
            nohup python -m test.eval_batch > "logs/eval_gpu${GPU_ID}_$(basename $(dirname $(dirname $MODEL_PATH))).log" 2>&1 &
        sleep 2
    else
        echo "Skipping: $MODEL_PATH (not found)"
    fi
done

echo "Parallel eval jobs started. Check logs/ for output."
echo "Running processes:"
ps aux | grep 'python.*eval_batch' | grep -v grep
