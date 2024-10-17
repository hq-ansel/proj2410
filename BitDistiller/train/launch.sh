#!/bin/bash
(
    cd /home/ubuntu/data/exp/proj2410/BitDistiller/train
    # 指定数据集位置
    export HF_HOME="/home/ubuntu/data/exp/proj2410/hf_home"
    export MODEL_PATH=/home/ubuntu/data/exp/proj2410/model/Llama2-7b
    export CUDA_VISIBLE_DEVICES=4,5,6,7

    bash /home/ubuntu/data/exp/proj2410/BitDistiller/train/train.sh \
    /home/ubuntu/data/exp/proj2410/BitDistiller/data_gen/datasets/hf-llama-2-7b/mix_wiki_alpaca_8000.json \
    /home/ubuntu/data/exp/proj2410/BitDistiller/cache/hf-llama2-7b/int2-g128.pt \
    /home/ubuntu/data/exp/proj2410/BitDistiller/logs/hf-llama-2-7b/int2-g128/ 4
)