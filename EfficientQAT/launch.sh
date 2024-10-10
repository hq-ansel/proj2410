#!/bin/bash
(
    cd /home/ubuntu/data/exp/proj2410/EfficientQAT
    # 指定数据集位置
    export HF_HOME="/home/ubuntu/data/exp/proj2410/hf_home"

    export CUDA_VISIBLE_DEVICES=3  # or e.g. 0,1,2,3
    export MODEL_PATH=/home/ubuntu/data/exp/proj2410/model/Llama2-7b
    export DATASET_PATH=pajama
    export SAVE_PATH=/home/ubuntu/data/exp/proj2410/quant_model/EfficientQAT/w4gs128/Llama2-7b

    python main_block_ap.py  \
    --model $MODEL_PATH \
    --output_dir /home/ubuntu/data/exp/proj2410/EfficientQAT/output/block_ap_log/Llama-2-7b-w4g128 \
    --net Llama-2 \
    --wbits 4 \
    --group_size 128 \
    --quant_lr 1e-5 \
    --weight_lr 1e-5 \
    --real_quant \
    --eval_ppl \
    --epochs 5 \
    --save_quant_dir $SAVE_PATH 
)