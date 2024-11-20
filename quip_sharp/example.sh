# Example script for quantizing Llama 2 7b with QuIP#

# CKPT=ckpt
# HF=hf
# LOG=log
# HESS=hess

# mkdir $CKPT
# mkdir $HF
# mkdir $LOG

BASEMODEL=/home/ubuntu/data/exp/proj2410/model/Qwen2.5-0.5B
SAVE_PATH=/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/quip-sharp
HESS=$SAVE_PATH/hessian_6144
LOG=$SAVE_PATH/log
FINETUNE=/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/quip-sharp-ft


mkdir -p $SAVE_PATH
mkdir -p $HESS
mkdir -p $FINETUNE

# generate hessians (takes a while, only use this if there aren't pregenerated hessians)
# python -m quantize_llama.hessian_offline_llama \
#     --batch_size 4 \
#     --devset_size 4096 \
#     --ctx_size 2048 \
#     --base_model $BASEMODEL \
#     --save_path $HESS

# download hessians (see the relaxml HF org to see what we have already)
# python scripts/download_hf.py --folder_path $HESS/llama2_7b_6144 --repo_id relaxml/Hessians-Llama-2-7b-6144 --read_token <TOKEN>

# quantize with finetuning
python -m quantize_llama.quantize_finetune_llama \
 --save_path $SAVE_PATH \
 --codebook E8P12  \
 --scale_override 0.9 \
 --base_model $BASEMODEL  \
 --hessian_path $HESS/ \
 --devset_size 4096 \
 --ft_valid_size 128 >> $LOG 2>&1

# convert model to hf format for end to end fine tuning
CUDA_VISIBLE_DEVICES=0 python -m quantize_llama.hfize_llama  \
    --quantized_path $SAVE_PATH  \
    --hf_output_path $FINETUNE >> LOG 2>&1

# end to end fine tuning
python -m quantize_llama.finetune_e2e_llama  \
    --base_model $BASEMODEL  \
    --hf_path $FINETUNE  \
    --devset_size 4096  \
    --ft_valid_size 128  \
    --ft_epochs 8   \
    --ft_bs 1  \
    --ctx_size 2048  \
    --ft_update_freq 2  \
    --ft_train_mode  \
    --ckpt_path $SAVE_PATH >> LOG 2>&1

# eval
CUDA_VISIBLE_DEVICES=0 python -m quantize_llama.hfize_llama  \
    --quantized_path $SAVE_PATH  \
    --hf_output_path $FINETUNE >> LOG 2>&1 
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_ppl  \
    --hf_path $FINETUNE >> LOG 2>&1 
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_zeroshot  \
    --tasks arc_challenge,arc_easy,boolq,piqa,winogrande  \
    --batch_size 4  \
    --hf_path $FINETUNE >> LOG 2>&1 &  




