import os
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from easydict import EasyDict as edict

from EfficientQAT.datautils_block import generate_block_train_data, get_loaders, LazyLoadDataset,generate_llama_mask_and_position_embedding

os.environ["HF_HOME"] = "/home/ubuntu/data/exp/proj2410/hf_home"

# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="Script for data generation or reading")
    parser.add_argument("--mode", type=str, choices=["write", "read"], required=True,
                        help="Mode to run: 'write' to generate and save data, 'read' to load and process data.")
    return parser.parse_args()



# 初始化配置
args = edict({
    # "model": "/home/ubuntu/data/exp/proj2410/model/Qwen2.5-0.5B",
    "model": "/home/ubuntu/data/exp/proj2410/model/Llama2-7b",
    "model_arch" : "Llama2-7b",
    # "model_arch" : "Qwen2.5-0.5B",
    "out_dir": "test_data",
    "eval_ppl": True,
    "eval_tasks": "piqa,arc_easy,arc_challenge,hellaswag,winogrande",
    "max_memory": "24GB",
    "ppl_seqlen": 2048,
    "batch_size": 64,
    "calib_dataset": "redpajama",
    "train_size": 4096,
    "val_size": 64,
    "seed": 42,
    "eval_batch_size": 32,
    "training_seqlen": 2048
})

# 加载模型和分词器
config = AutoConfig.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    config=config,
    device_map='cpu',
    attn_implementation="eager",
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, legacy=False)

def main():
    cli_args = parse_args()
    
    if cli_args.mode == "write":
        # 获取训练和验证数据加载器
        trainloader, valloader = get_loaders(
            args.calib_dataset,
            tokenizer,
            args.train_size,
            args.val_size,
            seed=args.seed,
            seqlen=args.training_seqlen,
        )
        device = "cuda:0"
        # model.to(device)
        # 设置数据集存储路径
        dataset_dir = f"/home/ubuntu/data/exp/proj2410/cache/{args.model_arch}/{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}/"
        val_dataset_dir = os.path.join(dataset_dir, "val")
        train_dataset_dir = os.path.join(dataset_dir, "train")
        os.makedirs(train_dataset_dir, exist_ok=True)
        os.makedirs(val_dataset_dir, exist_ok=True)

        # 生成并保存训练数据
        with torch.no_grad():
            attention_mask, position_embeddings =  generate_block_train_data(model, trainloader, train_dataset_dir)
            assert attention_mask is not None and position_embeddings is not None , "Attention mask and position embeddings should not be None"
            generate_block_train_data(model, valloader, val_dataset_dir)
        # torch.save({"attention_mask": attention_mask, "position_embeddings": position_embeddings},
        #            os.path.join(dataset_dir, "mask_and_position_embedding.pt"))
        print(f"Data saved to {dataset_dir}")
    
    elif cli_args.mode == "read":
        dataset_dir = "/home/ubuntu/data/exp/proj2410/cache/Qwen2.5-0.5B/{}_{}_{}_{}".format(
            args.calib_dataset, args.train_size, args.val_size, args.training_seqlen
        )
        train_dataset = LazyLoadDataset(dataset_dir, 
                                  tmp_dir="/home/ubuntu/data/exp/proj2410/cache/tmp", 
                                  split="train", )
        device = "cuda:0"
        model.to(device)
        device = next(model.parameters()).device
        # 首先测试是否能够读取
        print(f"train_dataset length: {len(train_dataset)}" )
        print(f"train_dataset[0]: {train_dataset[0]}" )
        # 然后测试是不是可以用layer0 的输出 更新到layer1 的输入
        # attention_mask,position_embeddings = generate_llama_mask_and_position_embedding(seq_len=args.training_seqlen,
        #                                                                                 rotary_emb=model.model.rotary_emb,
        #                                                                                 hidden_size=model.config.hidden_size,
        #                                                                                 dtype=torch.float16,
        #                                                                                 device=device)
        mask_and_pos_embed = torch.load(os.path.join(dataset_dir, "mask_and_position_embedding.pt"), weights_only=True)
        attention_mask = mask_and_pos_embed["attention_mask"].to(device,dtype=torch.float16)
        position_embeddings = mask_and_pos_embed["position_embeddings"]
        for i in range(10):
            layer0 = model.model.layers[i]
            train_dataset.update_dataset(module=layer0, 
                                        layer_idx=i+1,
                                        batch_size=2,
                                        attention_mask=attention_mask,
                                        num_workers=2,
                                        position_embeddings=position_embeddings,
                                            )
            print(f"train_dataset[0] after update: {train_dataset[0]}" )
        
        


import torch.multiprocessing as mp

# 设置多线程启动方式
mp.set_start_method("spawn", force=True)

if __name__ == "__main__":
    main()
