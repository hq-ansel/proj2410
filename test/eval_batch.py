
from EfficientQAT.quantize.int_linear_real import load_quantized_model
from EfficientQAT.main_block_ap import evaluate
from EfficientQAT.datautils_block import BlockTrainDataset, get_loaders
from EfficientQAT.quantize.crossblockquant import update_dataset
# from template.datautils import *
from easydict import EasyDict
from transformers import AutoTokenizer, AutoModelForCausalLM

from EfficientQAT.main_block_ap import evaluate

quant_path_list = [
    # 仅block wise
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-crossblock",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-gradual-quant",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-gradual-quant-slide2",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide2",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide12",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide4",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide6",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide8",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide10",

    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide2-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-gradual-quant-slide2-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide4-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide6-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide8-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide10-alpaca-4096/checkpoint-10000",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide12-alpaca-4096/checkpoint-10000",

    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-redpajama-4096/checkpoint-256",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-gradual-quant-redpajama-4096/checkpoint-256",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-gradual-quant-slide2-redpajama-4096/checkpoint-256",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide2-redpajama-4096/checkpoint-256",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide4-redpajama-4096/checkpoint-256",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide10-redpajama-4096/checkpoint-256",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide6-redpajama-4096/checkpoint-256",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide8-redpajama-4096/checkpoint-256",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-slide12-redpajama-4096/checkpoint-256",

    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-subspace",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-subspace-lr1",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-subspace-lr2",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-subspace-lr3",

    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-linearv2",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-gradual-quant",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-gradual-quant-cli1",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-gradual-quant-nocli",

    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-linearv2/redpajama-4096/checkpoint-256",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-lr1/redpajama-4096/checkpoint-256",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128/redpajama-4096/checkpoint-256",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-subspace-lr1/redpajama-4096/checkpoint-256",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-gradual-quant-nocli/redpajama-4096/checkpoint-256",

    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-freeze-weight",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-dampenloss",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-gradual-quant",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-gradual-quant-cli1-v2",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-7B/EfficientQAT/w2gs128",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-7B/EfficientQAT/w2gs128-gradual-quant",
    "/home/ubuntu/data/exp/proj2410/model/Llama3-8B",
    # "/home/ubuntu/data/exp/proj2410/model/Qwen2.5-3B",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-3B/EfficientQAT/w2gs128",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-3B/EfficientQAT/w2gs128-gradual-quant",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-3B/EfficientQAT/w3gs128",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-3B/EfficientQAT/w3gs128-gradual-quant",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama3-8B/EfficientQAT/w2gs128",

    # "/home/ubuntu/data/exp/proj2410/baseline/EfficientQAT/output/block_ap_models/Llama-3-8b-w2g128",
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama3-8B/EfficientQAT/w2gs128-gradual-quant"
    # "/home/ubuntu/data/exp/proj2410/quant_model/Llama3-8B/EfficientQAT/w4gs128",
]
# CUDA_VISIBLE_DEVICES=2 python -m test.eval_batch
import re
patterns = r"w(\d+)"
for quant_path in quant_path_list:
    if "EfficientQAT" in quant_path:
        match = re.search(patterns, quant_path).group(1)
        if match is None:
            wbit = 32
        else:
            wbit = int(match)
        quant_model,tokenizer = load_quantized_model(quant_path,wbit,128)
    else:
        quant_model = AutoModelForCausalLM.from_pretrained(quant_path,
                                                           device_map="auto")
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
    # args.eval_tasks="xsum,cnn_dailymail,openbookqa,copa,mathqa,rte"
    args.eval_batch_size=4
    args.training_seqlen = 2048
    evaluate(quant_model,tokenizer,args)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
