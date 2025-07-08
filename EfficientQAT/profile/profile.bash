export PYTHONPATH="/home/ubuntu/data/exp/proj2410:$PYTHONPATH"
nsys profile --multi-process python -m EfficientQAT.main_block_ap \
    --config_path /home/ubuntu/data/exp/proj2410/EfficientQAT/yaml/Llama3-8b/w4g128.yaml \
    --wbits 4 \
    --group_size 128 \
    --quant_lr 1e-5 \
    --weight_lr 1e-5 \
    --batch_size 8 \
    --real_quant \
    --eval_ppl \
    --epochs 2 \
    --save_quant_dir /home/ubuntu/data/exp/proj2410/quant_model/EfficientQAT/w4gs128/Llama3-8b
