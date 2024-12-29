import gc
import torch
import torch.nn as nn
import numpy as np
import random
import argparse

from transformers.models.qwen2 import Qwen2Model

seed =42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

tensor =torch.Tensor([ i*0.01 for i in range(4096)])
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
tensor = tensor.reshape(32,-1,32).float()

linear = nn.Linear(32, 16).to(device)

with torch.no_grad():

    inp_list = []

    for i in range(33):
        inp_list.append(tensor*i*1.1)
    inp_tensor = torch.cat(inp_list, dim=0).to(device)

    outputs = linear(inp_tensor)


    k = outputs.cpu()

    print(f"before tensor deleted {k}")

    del tensor,inp_tensor,outputs,linear
    gc.collect()
    torch.cuda.empty_cache()
    print(f"tensor deleted {k}")