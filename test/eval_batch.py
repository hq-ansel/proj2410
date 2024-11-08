
from EfficientQAT.quantize.int_linear_real import load_quantized_model
from EfficientQAT.main_block_ap import evaluate
from EfficientQAT.datautils_block import BlockTrainDataset, get_loaders
from EfficientQAT.quantize.crossblockquant import update_dataset
from template.datautils import *
from easydict import EasyDict
from transformers import AutoTokenizer, AutoModelForCausalLM

from EfficientQAT.main_block_ap import evaluate

quant_path_list = [
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-gradual-quant-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-end2start-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-end2start-align-end-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-slide2-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-slide2-algin-end-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-gradual-quant-slide2-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-gradual-quant-slide2-end2start-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-interpolate-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-gradual-interpolate-alpaca-4096/checkpoint-10000",
    "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-skip25-alpaca-4096/checkpoint-10000",
    "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-skip50-alpaca-4096/checkpoint-10000",
]


for quant_path in quant_path_list:
    quant_model,tokenizer = load_quantized_model(quant_path,2,128)
    args = EasyDict()
    args.eval_ppl = True
    args.eval_tasks = ""
    args.max_memory = "24GB"
    args.ppl_seqlen = 2048
    args.batch_size = 1
    args.calib_dataset = "redpajama"
    args.train_size = 1
    args.val_size = 1
    args.seed = 42
    args.eval_tasks="piqa,arc_easy,arc_challenge,hellaswag,winogrande"
    args.eval_batch_size=32
    args.training_seqlen = 2048
    evaluate(quant_model,tokenizer,args)
