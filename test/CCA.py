import argparse
from typing import List, Tuple
import os
import copy
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from easydict import EasyDict as edict
from pyhessian import hessian 
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

from EfficientQAT.datautils_block import generate_block_train_data, get_loaders, LazyLoadDataset,generate_llama_mask_and_position_embedding
from EfficientQAT.quantize.greedy_trainer import trans_quant_block,timer
from EfficientQAT.quantize.int_linear_real import load_quantized_model,QuantLinear


os.environ["HF_HOME"] = "/home/ubuntu/data/exp/proj2410/hf_home"


import torch
import torch.linalg as linalg

class CCASimilarity:
    def __init__(self, epsilon=1e-6, threshold=0.98):
        self.epsilon = epsilon
        self.threshold = threshold

    def positivedef_matrix_sqrt(self, array):
        """计算正定矩阵的平方根。
        
        参数:
            array (torch.Tensor): 代表正定矩阵的二维张量。
        
        返回:
            torch.Tensor: 矩阵的平方根。
        """
        # 使用特征分解计算矩阵的平方根
        w, v = linalg.eigh(array)
        wsqrt = torch.sqrt(w)
        sqrtarray = v @ torch.diag(wsqrt) @ v.T.conj()
        return sqrtarray

    def remove_small(self, sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon):
        """移除范数小于阈值 epsilon 的方向。
        
        参数:
            sigma_xx, sigma_xy, sigma_yx, sigma_yy (torch.Tensor): 协方差矩阵。
            epsilon (float): 小于此值的方向将被丢弃。
        
        返回:
            经过过滤的协方差矩阵和非零元素的索引掩码的元组。
        """
        x_diag = torch.abs(torch.diag(sigma_xx))
        y_diag = torch.abs(torch.diag(sigma_yy))
        x_idxs = x_diag >= epsilon
        y_idxs = y_diag >= epsilon
        
        sigma_xx_crop = sigma_xx[x_idxs][:, x_idxs]
        sigma_xy_crop = sigma_xy[x_idxs][:, y_idxs]
        sigma_yx_crop = sigma_yx[y_idxs][:, x_idxs]
        sigma_yy_crop = sigma_yy[y_idxs][:, y_idxs]
        
        return sigma_xx_crop, sigma_xy_crop, sigma_yx_crop, sigma_yy_crop, x_idxs, y_idxs

    def compute_ccas(self, sigma_xx, sigma_xy, sigma_yx, sigma_yy, verbose=True):
        """计算协方差矩阵之间的 CCA。
        
        参数:
            sigma_xx, sigma_xy, sigma_yx, sigma_yy (torch.Tensor): 协方差矩阵。
            verbose (bool): 是否打印中间步骤。
        
        返回:
            包含典型方向、奇异值和逆平方根的元组。
        """
        sigma_xx, sigma_xy, sigma_yx, sigma_yy, x_idxs, y_idxs = self.remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, self.epsilon)

        numx = sigma_xx.shape[0]
        numy = sigma_yy.shape[0]

        # 如果没有有效的数据，则返回零张量
        if numx == 0 or numy == 0:
            return ([torch.zeros_like(sigma_xx)], [torch.zeros_like(sigma_yy)], x_idxs, y_idxs)

        if verbose:
            print("添加 epsilon 到对角线并计算逆矩阵")
        sigma_xx += self.epsilon * torch.eye(numx)
        sigma_yy += self.epsilon * torch.eye(numy)
        inv_xx = torch.inverse(sigma_xx)
        inv_yy = torch.inverse(sigma_yy)

        invsqrt_xx = self.positivedef_matrix_sqrt(inv_xx)
        invsqrt_yy = self.positivedef_matrix_sqrt(inv_yy)

        if verbose:
            print("计算 CCA")
        arr = invsqrt_xx @ sigma_xy @ invsqrt_yy
        u, s, v = torch.svd(arr)

        return [u, torch.abs(s), v], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs

    def get_cca_similarity(self, acts1, acts2, compute_coefs=True, compute_dirns=False, verbose=True):
        """计算两个激活矩阵之间的 CCA 相似度。
        
        参数:
            acts1, acts2 (torch.Tensor): 激活矩阵，形状为 (神经元数量, 数据点数量)。
            compute_coefs (bool): 是否计算系数矩阵。
            compute_dirns (bool): 是否计算 CCA 方向。
            verbose (bool): 是否打印中间步骤。
        
        返回:
            包含 CCA 相似度结果的字典，包括系数、方向和摘要统计信息。
        """
        # 检查输入维度是否匹配
        assert acts1.shape[1] == acts2.shape[1], "acts1 和 acts2 维度不匹配"
        assert acts1.shape[0] < acts1.shape[1], "输入应为 (神经元数量, 数据点数量)"

        numx, numy = acts1.shape[0], acts2.shape[0]
        # 计算联合协方差矩阵
        covariance = torch.cov(torch.cat((acts1, acts2), dim=0))
        sigmaxx = covariance[:numx, :numx]
        sigmaxy = covariance[:numx, numx:]
        sigmayx = covariance[numx:, :numx]
        sigmayy = covariance[numx:, numx:]

        # 标准化协方差矩阵以提高 CCA 计算的稳定性
        xmax = torch.max(torch.abs(sigmaxx))
        ymax = torch.max(torch.abs(sigmayy))
        sigmaxx /= xmax
        sigmayy /= ymax
        sigmaxy /= torch.sqrt(xmax * ymax)
        sigmayx /= torch.sqrt(xmax * ymax)

        [u, s, v], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs = self.compute_ccas(sigmaxx, sigmaxy, sigmayx, sigmayy, verbose)

        # 初始化结果字典
        if compute_coefs:
            return_dict = {
                "coef_x": u.T,
                "invsqrt_xx": invsqrt_xx,
                "coef_y": v,
                "invsqrt_yy": invsqrt_yy,
                "cca_coef1": s,
                "cca_coef2": s,
                "x_idxs": x_idxs,
                "y_idxs": y_idxs,
                "neuron_means1": torch.mean(acts1, dim=1,keepdim=True),
                "neuron_means2": torch.mean(acts2, dim=1,keepdim=True),
                "mean": (torch.mean(s), torch.mean(s)),
                "sum": (torch.sum(s), torch.sum(s))
            }

            # 计算 CCA 方向（如果需要）
            if compute_dirns:
                cca_dirns1 = invsqrt_xx @ (acts1 - acts1.mean(dim=1, keepdim=True))
                cca_dirns2 = invsqrt_yy @ (acts2 - acts2.mean(dim=1, keepdim=True))
                return_dict["cca_dirns1"] = cca_dirns1
                return_dict["cca_dirns2"] = cca_dirns2
        else:
            return_dict = {"cca_coef1": s, "cca_coef2": s}

        return return_dict
    @timer
    def compute_pwcca(self, acts1, acts2):
        """计算投影加权的 CCA 系数。
        
        参数:
            acts1, acts2 (torch.Tensor): 激活矩阵，形状为 (神经元数量, 数据点数量)。
        
        返回:
            原始 CCA 系数均值、加权均值以及原始 CCA 系数。
        """
        sresults = self.get_cca_similarity(acts1, acts2, compute_coefs=True, compute_dirns=False, verbose=False)
        
        # 确定投影方向
        if torch.sum(sresults["x_idxs"]) <= torch.sum(sresults["y_idxs"]):
            dirns = sresults["coef_x"] @ (acts1[sresults["x_idxs"]] - sresults["neuron_means1"][sresults["x_idxs"]]) + sresults["neuron_means1"][sresults["x_idxs"]]
            coefs = sresults["cca_coef1"]
            acts = acts1
            idxs = sresults["x_idxs"]
        else:
            dirns = sresults["coef_y"] @ (acts2[sresults["y_idxs"]] - sresults["neuron_means2"][sresults["y_idxs"]]) + sresults["neuron_means2"][sresults["y_idxs"]]
            coefs = sresults["cca_coef2"]
            acts = acts2
            idxs = sresults["y_idxs"]
        
        # 使用 QR 分解来计算权重
        P, _ = torch.linalg.qr(dirns.T)
        weights = torch.sum(torch.abs(P.T @ acts[idxs].T), dim=1)
        weights = weights / torch.sum(weights)
        
        sresults["pwcca_coef"] = coefs
        sresults["pwcca_coef_sum"] = torch.sum(coefs*weights)
        sresults["pwcca_weights"] = weights
        return sresults

# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="Script for data generation or reading")
    # parser.add_argument("--mode", type=str, choices=["write", "read"], required=True,
    #                     help="Mode to run: 'write' to generate and save data, 'read' to load and process data.")
    return parser.parse_args()

def get_model_and_tokenizer(model_path:str) -> Tuple[AutoModelForCausalLM,AutoTokenizer]:
    quantized = "bit" in model_path
    if "bit" in model_path:
        wbits =2
        group_size = 128
        model,tokenizer = load_quantized_model(model_path,
                                                    wbits=wbits,
                                                    group_size=group_size)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model,tokenizer,quantized

@timer
def get_activation_in_layers(model:AutoModelForCausalLM,
                                   tokenizer:AutoTokenizer,
                                   valloader:List[Tuple[torch.Tensor,torch.Tensor]]
                                   )-> List[torch.Tensor]:
    class BatchDataSet(Dataset):
        def __init__(self, _loader:List[Tuple[torch.Tensor,torch.Tensor]]):
            self.data = []
            # self.batch_size = 32
            for inp,out in _loader:
                self.data.append((inp.squeeze(0),out.squeeze(0)))
        def __len__(self):
            return len(self.data)
        def __getitem__(self, index):
            return self.data[index]
    dataloader = DataLoader(BatchDataSet(valloader),batch_size=8,shuffle=False)
    hook_list = []
    activations = [[] for _ in range(len(model.model.layers)+1)]
    def get_out_hook(idx):
        def hook(module, input, output):
            activations[idx].append(output[0].detach().cpu())
        return hook
    for idx,layer in enumerate(model.model.layers):
        hook_list.append(layer.register_forward_hook(get_out_hook(idx+1)))
    model.to(device)
    with torch.no_grad():
        for inp,out in dataloader:
            inp = inp.to(device)
            out = out.to(device)
            assert inp.dim() == 2
            activations[0].append(inp.detach().cpu())
            model(inp)
    for hook in hook_list:
        hook.remove()
    model.cpu()
    for idx in range(len(activations)):
        if idx == 0:
            continue
        activations[idx] = torch.cat(activations[idx],dim=0)
    return activations

def compare_cca_in_quantized_model(base_model:AutoModelForCausalLM,
                                   tokenizer:AutoTokenizer,
                                   valloader:List[Tuple[torch.Tensor,torch.Tensor]]):
    quantized_path = [
        "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast",
        "raw_2bit", # 代表直接执行量化
        "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-slide2",
        "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast-gradual-quant-slide2-alpaca-4096/checkpoint-10000",
    ]
    base_model_path = "/home/ubuntu/data/exp/proj2410/model/Qwen2.5-0.5B"
    base_activations = get_activation_in_layers(base_model,tokenizer,valloader)
    # hidden_size = base_model.config.hidden_size
    for path in quantized_path:
        args = edict({
            "wbits": 2,
            "group_size": 128,
        })
        if path == "raw_2bit":
            compared_model = copy.deepcopy(base_model)
            for idx in range(len(compared_model.model.layers)):
                compared_model.model.layers[idx] = trans_quant_block(compared_model.model.layers[idx], args)
        else:
            compared_model,_ = load_quantized_model(path,args.wbits,args.group_size)
        compared_activations = get_activation_in_layers(compared_model,tokenizer,valloader)
        cca = CCASimilarity()

        # 计算3个cca距离
        svcca_distance,pwcca_distance = [0],[0]
        for i in range(len(base_activations)):
            if i == 0:
                continue
            base_act = base_activations[i]
            compared_act = compared_activations[i]

            dim = base_act.shape[-1]
            base_act = base_act.reshape(-1,dim).T
            compared_act = compared_act.reshape(-1,dim).T
            base_act = base_act.contiguous()
            compared_act = compared_act.contiguous()
            
            sresults = cca.compute_pwcca(base_act,compared_act)
            svcca_dist = 1 - torch.mean(sresults["cca_coef1"])
            pwcca_dist = 1 - sresults["pwcca_coef_sum"]
            svcca_distance.append(svcca_dist.item())
            pwcca_distance.append(pwcca_dist.item())
            print(f"layer {i} svcca_dist: {svcca_dist}, pwcca_dist: {pwcca_dist}")
        # get json
        res_dict = {
            "base_model_path": "/home/ubuntu/data/exp/proj2410/model/Qwen2.5-0.5B",
            "quantized_model_path": path,
            "svcca_distance": svcca_distance,
            "pwcca_distance": pwcca_distance,
        }
        name1 = base_model_path.split("/")[-1]
        name2 = path.split("/")[-1]
        with open(f"/home/ubuntu/data/exp/proj2410/test/cca/{name1}_{name2}.json", "w") as f:
            json.dump(res_dict, f, indent=4)
        
        # save json
        # plot

def main():
    cli_args = parse_args()
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
    # 加载模型和 tokenizer
    base_model,tokenizer,quantized = get_model_and_tokenizer(args.model)
    # 加载模型参数
    # 获取训练和验证数据加载器
    trainloader, valloader = get_loaders(
        args.calib_dataset,
        tokenizer,
        args.train_size,
        args.val_size,
        seed=args.seed,
        seqlen=args.training_seqlen,
    )
    # model.to(device)
    compare_cca_in_quantized_model(base_model,tokenizer,valloader)
    

if __name__ == '__main__':
    device = "cuda:0"
    main()