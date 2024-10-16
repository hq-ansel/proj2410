# #!/bin/bash
# (
#     cd /home/ubuntu/data/exp/proj2410/BitDistiller/data_gen/generation
#     # 指定数据集位置
#     export CUDA_VISIBLE_DEVICES=4,5,6,7
#     export HF_HOME="/home/ubuntu/data/exp/proj2410/hf_home"
#     export MODEL_PATH=/home/ubuntu/data/exp/proj2410/model/Llama2-7b
#     export DATASET_NAME=wikitext
#     OUTPATH=/home/ubuntu/data/exp/proj2410/BitDistiller/data_gen/datasets
#     MAX_SAMPLE=3000

#     python generate_vllm.py  \
#     --base_model $MODEL_PATH \
#     --dataset_name $DATASET_NAME \
#     --out_path $OUTPATH \
#     --max_sample $MAX_SAMPLE
# )
#!/bin/bash
(
    cd /home/ubuntu/data/exp/proj2410/BitDistiller/data_gen/generation
    # 指定数据集位置
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    export HF_HOME="/home/ubuntu/data/exp/proj2410/hf_home"
    export MODEL_PATH=/home/ubuntu/data/exp/proj2410/model/Llama2-7b
    export DATASET_NAME=alpaca
    OUTPATH=/home/ubuntu/data/exp/proj2410/BitDistiller/data_gen/datasets
    MAX_SAMPLE=5000

    python generate_vllm.py  \
    --base_model $MODEL_PATH \
    --dataset_name $DATASET_NAME \
    --out_path $OUTPATH \
    --max_sample $MAX_SAMPLE;
    python mix_data.py;
)