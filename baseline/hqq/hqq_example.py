import torch
from transformers import AutoModelForCausalLM, HqqConfig,AutoTokenizer

# All linear layers will use the same quantization config
quant_config = HqqConfig(nbits=2, group_size=128)

# Load and quantize
model = AutoModelForCausalLM.from_pretrained(
    "/home/ubuntu/data/exp/proj2410/model/Llama2-7B", 
    torch_dtype=torch.float16, 
    device_map="cuda", 
    quantization_config=quant_config
)
tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/data/exp/proj2410/model/Llama2-7B")

from EfficientQAT.main_block_ap import evaluate
from easydict import EasyDict

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
evaluate(model,tokenizer,args)