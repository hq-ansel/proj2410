#!/bin/bash
(
cd /home/ubuntu/data/exp/aqlm
# 指定数据集位置
export HF_HOME="/home/ubuntu/data/exp/hf_home"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # or e.g. 0,1,2,3
export QUANTIZED_MODEL=/home/ubuntu/data/exp/quant_model/aqlm/n1024cb1ing8/llama-7b
export MODEL_PATH=/home/ubuntu/data/exp/model/llama-7b
export DATASET_PATH=pajama

# for 0-shot evals
python lmeval.py \
    --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=float16,parallelize=True \
    --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge \
    --batch_size 64 \
    --aqlm_checkpoint_path $QUANTIZED_MODEL # if evaluating quantized model
)

# if need 5 shot eval mmlu
# --tasks mmlu \
#   --num_fewshot 5 \
