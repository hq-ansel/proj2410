#!/bin/bash
(
    cd /home/ubuntu/data/exp/proj2410/BitDistiller/quantization
    # 指定数据集位置
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    export HF_HOME="/home/ubuntu/data/exp/proj2410/hf_home"
    export MODEL_PATH=/home/ubuntu/data/exp/proj2410/model/Llama2-7b
    OUTPATH=/home/ubuntu/data/exp/proj2410/BitDistiller/cache

    CUDA_VISIBLE_DEVICES=4,5,6,7 python autoclip.py \
    --model_path $MODEL_PATH \
    --calib_dataset pile \
    --quant_type int \
    --w_bit 2 \
    --q_group_size 128 \
    --run_clip \
    --dump_clip $"${OUTPATH}/hf-llama2-7b/int2-g128.pt"
)