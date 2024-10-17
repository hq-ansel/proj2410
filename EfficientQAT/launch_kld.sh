#!/bin/bash

# 定义配置文件列表
# config_files=(
#     "EfficientQAT/yaml/b4gs128-fkld.yaml"
#     "EfficientQAT/yaml/b4gs128-fkldrkld.yaml"
#     "EfficientQAT/yaml/b4gs128-msefkldrkdl.yaml"
#     "EfficientQAT/yaml/b4gs128-mseflkd.yaml"
#     "EfficientQAT/yaml/b4gs128-mserlkd.yaml"
#     "EfficientQAT/yaml/b4gs128-rkld.yaml"
# )
config_files=(
    "EfficientQAT/yaml/b2gs128-crossblock2.yaml"
    "EfficientQAT/yaml/b2gs128.yaml"
)
# 循环遍历每个配置文件并执行 Python 命令
for config_path in "${config_files[@]}"; do
    (
        # 在子 shell 中执行，确保不会影响主环境变量

        # 进入工作目录
        cd /home/ubuntu/data/exp/proj2410/

        # 设置环境变量
        export HF_HOME="/home/ubuntu/data/exp/proj2410/hf_home"
        # export CUDA_VISIBLE_DEVICES=1  # or e.g. 0,1,2,3
        export MODEL_PATH=/home/ubuntu/data/exp/proj2410/model/Llama2-7b
        export DATASET_PATH=pajama
        export SAVE_PATH=/home/ubuntu/data/exp/proj2410/quant_model/EfficientQAT/w4gs128/Llama2-7b
        export CONFIG_PATH="/home/ubuntu/data/exp/proj2410/$config_path"
        OUTPUT_DIR="/home/ubuntu/data/exp/proj2410/EfficientQAT/output/block_ap_log/Llama-2-7b-w4g128"
        export PYTHONPATH=$PYTHONPATH:/home/ubuntu/data/exp/proj2410/EfficientQAT

        echo "Running with config: $CONFIG_PATH"

        # 执行 python 脚本
        # python main_block_ap.py \
        python -m EfficientQAT.main_block_ap \
            --config_path $CONFIG_PATH \
            --model $MODEL_PATH \
            --output_dir $OUTPUT_DIR \
            --net Llama-2 \
            --wbits 4 \
            --group_size 128 \
            --quant_lr 1e-5 \
            --weight_lr 1e-5 \
            --batch_size 8 \
            --real_quant \
            --eval_ppl \
            --epochs 5 \
            --save_quant_dir $SAVE_PATH

        echo "Finished running with config: $CONFIG_PATH"
    )
done
