nohup env CUDA_VISIBLE_DEVICES=0,3 python /home/ubuntu/data/exp/proj2410/opponent/AQLM/main.py \
/home/ubuntu/data/exp/proj2410/model/Llama2-7b \
/home/ubuntu/data/exp/proj2410/cache/redpajama_stream/home__ubuntu__data__exp__proj2410__model__Llama2-7b__093bae4a748cf254.pt \
--nsamples=1024 --val_size=128 --num_codebooks=1 --nbits_per_codebook=16 \
--in_group_size=8 --relative_mse_tolerance=0.01 --finetune_batch_size=32 \
--finetune_max_epochs=10 --finetune_early_stop=3 --finetune_keep_best \
--local_batch_size=1 --offload_activations \
--save /home/ubuntu/data/exp/proj2410/quant_model/Llama2-7B/aqlm/w2g128 \
> output.log 2>&1 &
