from datasets import load_dataset
import os
# os.environ["HF_Home"] = "/home/ubuntu/data/exp/proj2410/hf_home"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# huggingface-cli download togethercomputer/RedPajama-Data-1T-Sample --repo-type=dataset --local-dir /home/ubuntu/data/exp/proj2410/hf_home/datasets/RedPajama-Data-1T-Sample
ds = load_dataset(
    "/home/ubuntu/data/exp/proj2410/hf_home/datasets/RedPajama-Data-1T-Sample",
            # encoding="utf-16"
        )