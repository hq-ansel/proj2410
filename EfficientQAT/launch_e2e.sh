#!/bin/bash
quant_model_paths=(
    # /home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast
# /home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-slide2
# /home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-slide4
# /home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-slide6
/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-slide8
# /home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-slide10
# /home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-slide12
)
for quant_model_path in ${quant_model_paths[@]}; do
    (
        cd /home/ubuntu/data/exp/proj2410/
        # 指定数据集位置
        export HF_HOME="/home/ubuntu/data/exp/proj2410/hf_home"
        export CUDA_VISIBLE_DEVICES=4,5,6,7 # or e.g. 0,1,2,3
        export MODEL_PATH=/home/ubuntu/data/exp/proj2410/model/Llama2-7b
        export DATASET_PATH=pajama
        export AMP_ENABLED=True
        export CONFIG_PATH=/home/ubuntu/data/exp/proj2410/EfficientQAT/yaml/b2gs128-fast.yaml

        python -m EfficientQAT.main_e2e_qp  \
        --quant_model_path $quant_model_path \
        --model_family Llama-2 \
        --wbits 2 \
        --group_size 128 \
        --learning_rate 2e-5 \
        --dataset redpajama \
        --dataset_format pt \
        --output_dir "${quant_model_path}-redpajama-4096" \
        --do_train True \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --logging_steps 10 \
        --save_strategy steps \
        --evaluation_strategy steps \
        --max_steps 4096 \
        --eval_steps 1024 \
         --num_train_epochs 1 \
        --eval_dataset_size 16 \
        --bf16 \
        --data_seed 42 \
        --max_grad_norm 0.3 \
        --save_total_limit 1 \
        --preprocessing_num_workers 32 \
        --do_ppl_eval
    )
done