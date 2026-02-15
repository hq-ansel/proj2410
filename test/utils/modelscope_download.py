from modelscope import snapshot_download


model_hub_list = [
    "AI-ModelScope/Mixtral-8x7B-Instruct-v0.1",
    "deepseek-ai/deepseek-moe-16b-chat",
    "LLM-Research/phi-4",
    # "LLM-Research/Llama-3.2-3B",
    # "LLM-Research/Llama-3.2-1B",
]

for model_hub in model_hub_list:
    model_dir = snapshot_download(
        model_hub,
        cache_dir="/home/ubuntu/data/exp/proj2410/model",
    )