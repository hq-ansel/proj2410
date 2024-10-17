#!/bin/bash
(
    cd /home/ubuntu/data/exp/proj2410/BitDistiller/train
    # 指定数据集位置
    export HF_HOME="/home/ubuntu/data/exp/proj2410/hf_home"
    export MODEL_PATH=/home/ubuntu/data/exp/proj2410/model/Llama2-7b

    bash /home/ubuntu/data/exp/proj2410/BitDistiller/train/train.sh \
    /home/ubuntu/data/exp/proj2410/BitDistiller/data_gen/datasets/hf-llama-2-7b/mix_wiki_alpaca_8000.json \
    /home/ubuntu/data/exp/proj2410/quant_model/Llama-2-7B/bitdistiller \
    /home/ubuntu/data/exp/proj2410/BitDistiller/logs 4
)