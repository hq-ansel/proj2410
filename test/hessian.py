import matplotlib.pyplot as plt
import os
import argparse

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from easydict import EasyDict as edict
from pyhessian import hessian 
import copy
from easydict import EasyDict as edict

from EfficientQAT.datautils_block import generate_block_train_data, get_loaders, LazyLoadDataset,generate_llama_mask_and_position_embedding
from EfficientQAT.quantize.greedy_trainer import trans_quant_block
 
os.environ["HF_HOME"] = "/home/ubuntu/data/exp/proj2410/hf_home"

# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="Script for data generation or reading")
    # parser.add_argument("--mode", type=str, choices=["write", "read"], required=True,
    #                     help="Mode to run: 'write' to generate and save data, 'read' to load and process data.")
    return parser.parse_args()



# 初始化配置
args = edict({
    "model": "/home/ubuntu/data/exp/proj2410/model/Qwen2.5-0.5B",
    # "model": "/home/ubuntu/data/exp/proj2410/model/Llama2-7b",
    # "model_arch" : "Llama2-7b",
    "model_arch" : "Qwen2.5-0.5B",
    "out_dir": "test_data",
    "eval_ppl": True,
    "eval_tasks": "piqa,arc_easy,arc_challenge,hellaswag,winogrande",
    "max_memory": "24GB",
    "ppl_seqlen": 2048,
    "batch_size": 64,
    "calib_dataset": "redpajama",
    "train_size": 4096,
    "val_size": 64,
    "seed": 42,
    "eval_batch_size": 32,
    "training_seqlen": 2048
})

# 加载模型和分词器
config = AutoConfig.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    config=config,
    device_map='cpu',
    attn_implementation="eager",
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, legacy=False)

@torch.no_grad()
def get_blk_dff(blk:nn.Module):
    loss_list = []
    args = edict({
        "wbits": 2,
        "group_size":128,
    })
    quant_blk = copy.deepcopy(blk)
    trans_quant_block(quant_blk,args
        )
    
    for name, module in blk.named_modules():
        if isinstance(module, nn.Linear):
            qmodule = dict(quant_blk.named_modules()).get(name, None)
            assert qmodule is not None, f"quant_blk doesn't have module {name}"
            fp_weight = module.weight.data.detach()
            qweight = qmodule.weight_quantizer.fake_quant(qmodule.weight)
            loss_list.append(torch.norm(fp_weight - qweight,p=2))
    return torch.sum(torch.Tensor(loss_list))


def main():
    cli_args = parse_args()
    
    # 获取训练和验证数据加载器
    # trainloader, valloader = get_loaders(
    #     args.calib_dataset,
    #     tokenizer,
    #     args.train_size,
    #     args.val_size,
    #     seed=args.seed,
    #     seqlen=args.training_seqlen,
    # )
    device = "cuda:0"
    # model.to(device)
    train_dir = "/home/ubuntu/data/exp/proj2410/cache/Qwen2.5-0.5B/{}_{}_{}_{}".format(
            args.calib_dataset, args.train_size, args.val_size, args.training_seqlen
        )
    file_dir = os.path.join(train_dir, "train")
    num_layers = len(model.model.layers)
    num_subplots = (num_layers + 3) // 4
    fig, axs = plt.subplots((num_subplots + 1) // 2, 2, figsize=(12, num_subplots * 2))
    axs = axs.flatten()  # 展平成一维数组方便索引
    hawq = []
    for subplot_idx in range(num_subplots):
        layer_indices = range(subplot_idx * 4, min((subplot_idx + 1) * 4, num_layers))
        for layer_idx in layer_indices:
            if layer_idx == 0:
                file_path = os.path.join(file_dir, f"input_layer{layer_idx}_0.pt")
            else:
                file_path = os.path.join(file_dir, f"output_layer{layer_idx-1}_0.pt")
            inp = torch.load(file_path, weights_only=True)
            file_path = os.path.join(file_dir, f"output_layer{layer_idx}_0.pt")
            target = torch.load(file_path, weights_only=True)

            mask_and_pos_embed = torch.load(os.path.join(train_dir, "mask_and_position_embedding.pt"), weights_only=True)
            
            attention_mask = mask_and_pos_embed["attention_mask"].to(device,dtype=torch.float32)
            position_embeddings = mask_and_pos_embed["position_embeddings"]
            position_embeddings = (position_embeddings[0].to(device,dtype=torch.float32),position_embeddings[1].to(device,dtype=torch.float32))
            inp.unsqueeze_(0)
            inp = {
                "hidden_states": inp.to(device,dtype=torch.float32),
                "attention_mask": attention_mask.to(device,dtype=torch.float32),
                "position_embeddings": position_embeddings
            }
            target.unsqueeze_(0)
            blk = model.model.layers[layer_idx]
            blk= blk.to(device)
            criterion = torch.nn.MSELoss()
            # print(f"blk {blk} criterion {criterion} inp {inp} target {target}")
            hessian_comp = hessian(blk, criterion, data=(inp, target.to(device)),cuda=True)
            #  compute the top eigenvalue of the Hessian matrix
            top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
            # print("The top Hessian eigenvalue of this model is %.4f"%top_eigenvalues[-1])
            hawq.append((layer_idx,top_eigenvalues[-1]*get_blk_dff(blk)))
            print(f" mse of blkid {layer_idx} {get_blk_dff(blk)}")
            #  compute the top two eigenvalues of the Hessian matrix
            top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
            # print("The top two eigenvalues of this model are: %.4f %.4f"% (top_eigenvalues[-1],top_eigenvalues[-2]))
            def get_params(model_orig,  model_perb, direction, alpha):
                """
                Args:
                    model_orig: original model
                    model_perb: perturbed model
                    direction: direction of perturbation
                    alpha: step size of perturbation
                Returns:
                    perturbed model
                """
                for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
                    m_perb.data = m_orig.data + alpha * d
                return model_perb
            
            
            # lambda is a small scalar that we use to perturb the model parameters along the eigenvectors 
            lams = np.linspace(-0.5, 0.5, 21).astype(np.float32)
            loss_list = []
            blk_disturb = copy.deepcopy(blk)
            blk_disturb.to(device)
            for i, lam in enumerate(lams):
                # compute the direction of the perturbation
                blk_disturb = get_params(blk, blk_disturb, top_eigenvector[0], lam)
                loss_list.append(criterion(blk_disturb(**inp)[0], target.to(device)).item())
            # 在当前子图中绘制损失函数与扰动量的关系
            axs[subplot_idx].plot(lams, loss_list, label=f"Layer {layer_idx}")
        axs[subplot_idx].set_xlabel("Perturbation amount")
        axs[subplot_idx].set_ylabel("Loss")
        axs[subplot_idx].set_title(f"Layers {subplot_idx * 4} to {min((subplot_idx + 1) * 4 - 1, num_layers - 1)}")
        axs[subplot_idx].legend()
    plt.tight_layout()
    plt.savefig("/home/ubuntu/data/exp/proj2410/test/loss_vs_perturb/combined_plot.png")
    plt.close()
    # print(hawq)
    sorted_hawq = sorted(hawq, key=lambda x: x[1])
    # print(f"The HAWQ of the model is {sorted_hawq}" )
    # print(f"The HAWQ of block ids {[x[0] for x in sorted_hawq]}")

import torch.multiprocessing as mp

# 设置多线程启动方式
mp.set_start_method("spawn", force=True)

if __name__ == "__main__":
    main()
