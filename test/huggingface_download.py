# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import huggingface_hub

huggingface_hub.login("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")


model_hub = "mistralai/Mixtral-8x7B-Instruct-v0.1"

save_dir = "/home/ubuntu/data/exp/proj2410/model"

tokenizer = AutoTokenizer.from_pretrained(model_hub)
model = AutoModelForCausalLM.from_pretrained(model_hub)
# 如果不存在 路径创建路径
target_dir =save_dir+"/"+model_hub
Path(target_dir).mkdir(parents=True, exist_ok=True)

model.save_pretrained(target_dir)
tokenizer.save_pretrained(target_dir)
# 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