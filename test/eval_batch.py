
from EfficientQAT.quantize.int_linear_real import load_quantized_model
from EfficientQAT.main_block_ap import evaluate
from EfficientQAT.datautils_block import BlockTrainDataset, get_loaders
from EfficientQAT.quantize.crossblockquant import update_dataset
from template.datautils import *
from easydict import EasyDict
from transformers import AutoTokenizer, AutoModelForCausalLM

from EfficientQAT.main_block_ap import evaluate

quant_path_list = [
    "/home/ubuntu/data/exp/proj2410/model/Qwen2.5-0.5B",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-redpajama-4096/checkpoint-4096",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide2-redpajama-4096/checkpoint-4096",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-gradual-quant-redpajama-4096/checkpoint-4096",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-gradual-quant-slide2-redpajama-4096/checkpoint-4096",
    # # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-slide2-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-slide4-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-slide6-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-slide8-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-slide10-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-slide12-alpaca-4096/checkpoint-10000",
]

for quant_path in quant_path_list:
    if "EfficientQAT" in quant_path:
        quant_model,tokenizer = load_quantized_model(quant_path,2,128)
    else:
        quant_model = AutoModelForCausalLM.from_pretrained(quant_path)
        tokenizer = AutoTokenizer.from_pretrained(quant_path)
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
