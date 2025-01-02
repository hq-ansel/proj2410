# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


path_list = [
    # "Qwen/Qwen2.5-1.5B",
    # "Qwen/Qwen2.5-3B",
    # "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B", 
    "Qwen/Qwen2.5-32B",
]

save_dir = "/home/ubuntu/data/exp/proj2410/model"

for path in path_list:
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    save_path = os.path.join(save_dir, path.split("/")[-1])
    print(f"Saving model to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)