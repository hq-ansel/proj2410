#!/bin/bash

(
    cd /home/ubuntu/data/exp/proj2410/
    # 指定数据集位置
    export HF_HOME="/home/ubuntu/data/exp/proj2410/hf_home"
    export CUDA_VISIBLE_DEVICES=1  # or e.g. 0,1,2,3
    export MODEL_PATH=/home/ubuntu/data/exp/proj2410/model/Llama2-7b
    export DATASET_PATH=pajama
    export AMP_ENABLED=True
    export CONFIG_PATH=/home/ubuntu/data/exp/proj2410/EfficientQAT/yaml/b2gs128-fast.yaml

    python -m EfficientQAT.main_e2e_qp  \
    --quant_model_path /home/ubuntu/data/exp/proj2410/quant_model/Llama2-7b/EfficientQAT/w2gs128-end2start \
    --model_family Llama-2 \
    --wbits 2 \
    --group_size 128 \
    --learning_rate 2e-5 \
    --dataset alpaca \
    --dataset_format alpaca \
    --output_dir /home/ubuntu/data/exp/proj2410/quant_model/Llama2-7b/EfficientQAT/w2gs128-end2start-alpaca-4096 \
    --do_train True \
    --do_mmlu_eval True \
    --source_max_len 384 \
    --target_max_len 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --save_strategy steps \
    --evaluation_strategy steps \
    --max_steps 10000 \
    --eval_steps 2000 \
    --eval_dataset_size 16 \
    --bf16 \
    --data_seed 42 \
    --max_grad_norm 0.3 \
    --group_by_length
)