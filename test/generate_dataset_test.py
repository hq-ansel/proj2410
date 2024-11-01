import os


from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch
from easydict import EasyDict as edict

args = edict({
    "model": "/home/ubuntu/data/exp/proj2410/model/Qwen-2.5-0.5B",
    "out_dir": "test_data",
})


args.eval_ppl = True
args.eval_tasks = ""
args.max_memory = "24GB"
args.ppl_seqlen = 2048
args.batch_size = 1
args.calib_dataset = "redpajama"
args.train_size = 4096
args.val_size = 64
args.seed = 42
args.eval_tasks="piqa,arc_easy,arc_challenge,hellaswag,winogrande"
args.eval_batch_size=32
args.training_seqlen = 2048

config = AutoConfig.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model,
                                        config=config,
                                        device_map='cpu',
                                        torch_dtype=torch.float16 )
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False,legacy=False)

from EfficientQAT.datautils_block import generate_block_train_data,get_loaders

trainloader, valloader = get_loaders(
                    args.calib_dataset,
                    tokenizer,
                    args.train_size,
                    args.val_size,
                    seed=args.seed,
                    seqlen=args.training_seqlen,
                )
dataset_dir =  f"/home/ubuntu/data/exp/proj2410/cache/Qwen-2.5-0.5B/{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}/"
val_dataset_dir =  os.path.join(dataset_dir, "val")
train_dataset_dir =  os.path.join(dataset_dir, "train")
os.makedirs(train_dataset_dir, exist_ok=True)
os.makedirs(val_dataset_dir, exist_ok=True)
generate_block_train_data(
    model,
    trainloader,
    train_dataset_dir,
)
generate_block_train_data(
    model,
    valloader,
    val_dataset_dir,
)
