#!/bin/bash
(
    cd /home/ubuntu/data/exp/aqlm
    # 指定数据集位置
    export HF_HOME="/home/ubuntu/data/exp/hf_home"

    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # or e.g. 0,1,2,3
    export MODEL_PATH=/home/ubuntu/data/exp/model/llama-7b
    export DATASET_PATH=pajama
    export SAVE_PATH=/home/ubuntu/data/exp/quant_model/aqlm/n1024cb1ing8/llama-7b

    python main.py $MODEL_PATH $DATASET_PATH \
    --nsamples=1024 \
    --val_size=128 \
    --num_codebooks=1 \
    --nbits_per_codebook=16 \
    --in_group_size=8 \
    --relative_mse_tolerance=0.01 \
    --finetune_batch_size=32 \
    --finetune_max_epochs=10 \
    --finetune_early_stop=3 \
    --finetune_keep_best \
    --local_batch_size=1 \
    --offload_activations \
    --resume \
    --save $SAVE_PATH
)