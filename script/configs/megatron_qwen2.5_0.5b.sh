#!/bin/bash
# Megatron-LM Qwen2.5-0.5B Configuration with TP/SP Support
# Usage: source this file to set MODEL_ARGS and TRAINING_ARGS

# Model Architecture (Qwen2.5-0.5B)
export NUM_LAYERS=24
export HIDDEN_SIZE=896
export FFN_HIDDEN_SIZE=4864
export NUM_ATTENTION_HEADS=14
export NUM_QUERY_GROUPS=2  # GQA
export HEAD_DIM=64
export VOCAB_SIZE=151936
export MAX_POSITION_EMBEDDINGS=32768
export SEQ_LENGTH=${SEQ_LENGTH:-2048}

# Parallelism Settings
export TP_SIZE=${TP_SIZE:-1}        # Tensor Parallelism (1, 2, 4, 8)
export PP_SIZE=${PP_SIZE:-1}        # Pipeline Parallelism
export CP_SIZE=${CP_SIZE:-1}        # Context Parallelism
export SP=${SP:-false}              # Sequence Parallelism (auto-enabled when TP>1)

# Auto-enable SP when TP > 1
if [[ $TP_SIZE -gt 1 ]]; then
    export SP=true
fi

# Training Settings
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
export TRAIN_ITERS=${TRAIN_ITERS:-1000}
export LR=${LR:-0.0001}
export MIN_LR=0.00001
export LR_WARMUP_ITERS=100
export WEIGHT_DECAY=0.1
export GRAD_CLIP=1.0

# Checkpoint Settings
export CHECKPOINT_PATH=${CHECKPOINT_PATH:-"checkpoints/qwen2.5-0.5b-megatron"}
export SAVE_INTERVAL=${SAVE_INTERVAL:-100}

# Data Settings
export DATA_PATH=${DATA_PATH:-"${REPO_ROOT}/data/mock"}
export TOKENIZER_PATH=${TOKENIZER_PATH:-"${REPO_ROOT}/model/Qwen2.5-0.5B"}

# Model Arguments
MODEL_ARGS=(
    --use-mcore-models
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
    --group-query-attention
    --num-query-groups $NUM_QUERY_GROUPS
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --swiglu
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --disable-bias-linear
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --vocab-size $VOCAB_SIZE
    --make-vocab-size-divisible-by 64
    # Qwen2.5 specific
    --untie-embeddings-and-output-weights
    --apply-layernorm-1p
)

# Add TP/SP arguments
PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
    --context-parallel-size $CP_SIZE
)

if [[ "$SP" == "true" ]]; then
    PARALLEL_ARGS+=(--sequence-parallel)
fi

# Training Arguments
TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITERS
    --lr $LR
    --min-lr $MIN_LR
    --lr-warmup-iters $LR_WARMUP_ITERS
    --lr-decay-style cosine
    --clip-grad $GRAD_CLIP
    --weight-decay $WEIGHT_DECAY
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --use-flash-attn
    --distributed-timeout-minutes 60
)

# Data Arguments
DATA_ARGS=(
    --data-path $DATA_PATH
    --split 949,50,1
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_PATH
)

# Checkpoint Arguments
CHECKPOINT_ARGS=(
    --save $CHECKPOINT_PATH
    --save-interval $SAVE_INTERVAL
    --log-interval 10
)

# Validation Arguments
VALIDATION_ARGS=(
    --eval-iters 10
    --eval-interval 100
)

# Combine all arguments
ALL_ARGS=("${MODEL_ARGS[@]}" "${PARALLEL_ARGS[@]}" "${TRAINING_ARGS[@]}" "${DATA_ARGS[@]}" "${CHECKPOINT_ARGS[@]}" "${VALIDATION_ARGS[@]}")

export MEGATRON_ARGS="${ALL_ARGS[*]}"
