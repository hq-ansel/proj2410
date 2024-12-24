#!/bin/bash

# 定义配置文件列表
# )
config_files=(
    /home/ubuntu/data/exp/proj2410/EfficientQAT/yaml/qwen2.5-0.5b/gradual-study/qwen2.5-0.5b-b2gs128-gradual-quant-cli1-v2.yaml
    /home/ubuntu/data/exp/proj2410/EfficientQAT/yaml/qwen2.5-0.5b/gradual-study/qwen2.5-0.5b-b2gs128-gradual-quant-cli1.yaml
)
# 循环遍历每个配置文件并执行 Python 命令
# 设置并行参数，True 为并行，False 为串行
PARALLEL=False

# 循环遍历每个配置文件并执行 Python 命令
for config_path in "${config_files[@]}"; do
    # 如果 PARALLEL 为 True，则并行运行
    if [ "$PARALLEL" = True ]; then
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
            export CONFIG_PATH="$config_path"
            OUTPUT_DIR="/home/ubuntu/data/exp/proj2410/EfficientQAT/output/block_ap_log/Llama-2-7b-w4g128"
            export PYTHONPATH=$PYTHONPATH:/home/ubuntu/data/exp/proj2410/EfficientQAT
            # export AMP_ENABLED=True
            echo "Running with config: $CONFIG_PATH"

            # 执行 python 脚本
            python -m EfficientQAT.main_block_ap \
                --config_path $CONFIG_PATH \
                --wbits 4 \
                --group_size 128 \
                --quant_lr 1e-5 \
                --weight_lr 1e-5 \
                --batch_size 8 \
                --real_quant \
                --eval_ppl \
                --epochs 2 \
                --save_quant_dir $SAVE_PATH

            echo "Finished running with config: $CONFIG_PATH"
        ) &
    else
        # 串行运行的部分
        (
            cd /home/ubuntu/data/exp/proj2410/
            export HF_HOME="/home/ubuntu/data/exp/proj2410/hf_home"
            export MODEL_PATH=/home/ubuntu/data/exp/proj2410/model/Llama2-7b
            export DATASET_PATH=pajama
            export SAVE_PATH=/home/ubuntu/data/exp/proj2410/quant_model/EfficientQAT/w4gs128/Llama2-7b
            export CONFIG_PATH="$config_path"
            OUTPUT_DIR="/home/ubuntu/data/exp/proj2410/EfficientQAT/output/block_ap_log/Llama-2-7b-w4g128"
            export PYTHONPATH=$PYTHONPATH:/home/ubuntu/data/exp/proj2410/EfficientQAT
            echo "Running with config: $CONFIG_PATH"
            python -m EfficientQAT.main_block_ap \
                --config_path $CONFIG_PATH \
                --wbits 4 \
                --group_size 128 \
                --quant_lr 1e-5 \
                --weight_lr 1e-5 \
                --batch_size 8 \
                --real_quant \
                --eval_ppl \
                --epochs 2 \
                --save_quant_dir $SAVE_PATH
            echo "Finished running with config: $CONFIG_PATH"
        )
    fi
done

# 等待所有并行子进程完成
if [ "$PARALLEL" = True ]; then
    wait
fi
