#!/bin/bash
quant_model_paths=(
    /home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-gradual-quant
)
for quant_model_path in ${quant_model_paths[@]}; do
    (
        cd /home/ubuntu/data/exp/proj2410/
        # 指定数据集位置
        export HF_HOME="/home/ubuntu/data/exp/proj2410/hf_home"
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # or e.g. 0,1,2,3
        export MODEL_PATH=/home/ubuntu/data/exp/proj2410/model/Llama2-7b
        export DATASET_PATH=pajama
        export AMP_ENABLED=True

        python -m EfficientQAT.main_e2e_qp  \
        --quant_model_path $quant_model_path \
        --model_family Llama-2 \
        --wbits 2 \
        --group_size 128 \
        --learning_rate 2e-5 \
        --dataset redpajama \
        --dataset_format pt \
        --output_dir "${quant_model_path}/redpajama-4096" \
        --do_train True \
        --pt_context_len 4096 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --logging_steps 1 \
        --save_strategy epoch \
        --training_strategy epochs \
        --evaluation_strategy steps \
        --eval_steps 64 \
        --max_train_samples 4096 \
        --num_train_epochs 1 \
        --eval_dataset_size 64 \
        --bf16 \
        --data_seed 42 \
        --max_grad_norm 0.3 \
        --save_total_limit 1 \
        --preprocessing_num_workers 32 \
        --do_ppl_eval
    )
done