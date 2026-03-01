"""
GPTQ (Gradient-based Post-training Quantization) PTQ 实现

GPTQ 是一种基于二阶信息的逐层量化方法。
核心思想：
1. 使用校准数据计算 Hessian 矩阵（二阶导数信息）
2. 使用 Cholesky 分解求解最优量化
3. 逐列量化权重，每次量化后更新剩余权重的误差

参考：https://arxiv.org/abs/2210.17323
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math


@dataclass
class GPTQConfig:
    """GPTQ 量化配置

    Attributes:
        n_bits: 量化位数 (默认：4)
        group_size: 分组大小 (默认：128)
        damp: 阻尼系数，用于 Hessian 正则化 (默认：0.01)
        percdamp: 阻尼百分比 (默认：0.01)
        block_size: 块大小 (默认：128)
        act_order: 是否使用激活顺序 (默认：False)
    """
    n_bits: int = 4
    group_size: int = 128
    damp: float = 0.01
    percdamp: float = 0.01
    block_size: int = 128
    act_order: bool = False
    static_groups: bool = False


class GPTQQuantizer:
    """GPTQ 量化器

    对 Linear 层进行 GPTQ 量化，使用二阶信息最小化量化误差。
    """

    def __init__(
        self,
        module: nn.Linear,
        config: GPTQConfig,
        name: str = "",
    ):
        self.module = module
        self.config = config
        self.name = name

        # 量化参数
        self.n_bits = config.n_bits
        self.group_size = config.group_size

        # 量化后的权重和参数
        self.qweight = None
        self.scales = None
        self.qzeros = None

        # Hessian 相关
        self.H = None
        self.nsamples = 0

        # 激活顺序（用于 act_order）
        self.perm = None
        self.invperm = None

    @property
    def weight(self) -> torch.Tensor:
        return self.module.weight

    @property
    def in_features(self) -> int:
        return self.module.in_features

    @property
    def out_features(self) -> int:
        return self.module.out_features

    @property
    def device(self) -> torch.device:
        return self.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.weight.dtype

    def add_input(self, inp: torch.Tensor) -> None:
        """添加输入样本，累积 Hessian 矩阵

        GPTQ 的核心：H = (2/nsamples) * X^T X，其中 X 是输入激活
        """
        self.module.eval()

        # 处理不同的输入格式
        # inp 可能是：
        # - [batch, seq_len, in_features] (3D)
        # - [batch, in_features] (2D)
        # - [batch, num_heads, seq_len, in_features] (4D, 需要展平)
        if inp.ndim == 4:
            # 4D 输入：展平为 [batch * num_heads * seq_len, in_features]
            inp = inp.reshape(-1, inp.shape[-1])
        elif inp.ndim == 3:
            # 3D 输入：展平为 [batch * seq_len, in_features]
            inp = inp.reshape(-1, inp.shape[-1])
        elif inp.ndim != 2:
            # 其他维度，尝试提取最后一个维度作为特征
            inp = inp.reshape(-1, inp.shape[-1])

        inp = inp.double()
        self.nsamples += inp.shape[0]

        # 累积 Hessian: H += (2/nsamples) * X^T X
        # 使用增量更新方式
        if self.H is None:
            self.H = torch.zeros((self.in_features, self.in_features), device=self.device, dtype=torch.double)

        inp = inp.to(self.device)
        self.H += 2.0 / self.nsamples * inp.T @ inp

    def quantize(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """执行 GPTQ 量化

        使用 Cholesky 分解和逐列量化来最小化量化误差。

        Returns:
            (qweight, scales, qzeros) 元组
        """
        self.module.eval()

        W = self.weight.data.double()
        dev = self.weight.device
        rows, columns = W.shape

        # 初始化 Hessian
        if self.H is None:
            raise ValueError("Must call add_input() to collect Hessian before quantize()")

        H = self.H

        # 阻尼处理：提高数值稳定性
        damp = self.config.damp
        if damp > 0:
            diag = torch.arange(columns, device=dev)
            H[diag, diag] += damp * H[diag, diag].mean()

        # 排列（可选的激活顺序）
        perm = None
        if self.config.act_order:
            # 按 Hessian 对角线大小排列
            diag = torch.diag(H)
            perm = torch.argsort(diag, descending=True)
            self.perm = perm
            W = W[:, perm]
            H = H[perm][:, perm]

        # 逐块量化
        block_size = self.config.block_size
        Q = torch.zeros_like(W, dtype=torch.double)

        for i1 in range(0, columns, block_size):
            i2 = min(i1 + block_size, columns)
            count = i2 - i1

            # 当前块的 Hessian 子矩阵
            H_block = H[i1:i2, i1:i2].double()
            W_block = W[:, i1:i2].double()

            # Cholesky 分解
            try:
                L = torch.linalg.cholesky(H_block)
            except torch.linalg.LinAlgError:
                # 如果不是正定的，添加正则化
                reg = torch.eye(count, device=dev, dtype=torch.double) * 1e-6
                H_block = H_block + reg
                L = torch.linalg.cholesky(H_block)

            Linv = torch.linalg.solve_triangular(L, torch.eye(count, device=dev, dtype=torch.double), upper=False)
            Hinv = Linv.T @ Linv

            # 量化当前块
            q_scale = self._get_quant_scale(W_block, Hinv)
            Q[:, i1:i2] = self._quantize_block(W_block, Hinv, q_scale)

        # 恢复原始顺序
        if self.config.act_order and self.perm is not None:
            self.invperm = torch.argsort(self.perm)
            Q = Q[:, self.invperm]

        # 计算量化误差
        err = (W - Q).abs().sum()
        print(f"GPTQ quantization error: {err.item():.6f}")

        # 存储结果
        self.qweight = Q.to(self.weight.dtype)

        # 计算 scales 和 qzeros（per-group）
        self._compute_group_params(Q)

        return self.qweight, self.scales, self.qzeros

    def _get_quant_scale(self, W: torch.Tensor, Hinv: torch.Tensor) -> float:
        """计算量化缩放因子"""
        # 简化的缩放计算
        W_max = W.abs().max()
        qmax = 2 ** self.n_bits - 1
        return float(W_max.item()) / (qmax / 2)

    def _quantize_block(self, W: torch.Tensor, Hinv: torch.Tensor, scale: float) -> torch.Tensor:
        """量化一个权重块

        GPTQ 的核心算法：
        1. 逐列量化
        2. 每次量化后更新剩余权重的误差
        """
        rows, columns = W.shape
        dev = W.device

        Q = torch.zeros_like(W, dtype=torch.double)
        qmin = 0
        qmax = 2 ** self.n_bits - 1

        for j in range(columns):
            w = W[:, j]
            d = Hinv[j, j]

            # 最优量化值
            q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale

            # 误差
            err = (w - q) / d

            # 更新 Q
            Q[:, j] = q

            # 更新剩余的权重（误差传播）
            if j < columns - 1:
                W[:, j+1:] -= err.unsqueeze(1) @ Hinv[j, j+1:].unsqueeze(0)

        return Q

    def _compute_group_params(self, Q: torch.Tensor) -> None:
        """计算 per-group 的 scales 和 qzeros"""
        if self.group_size <= 0:
            return

        W = Q.reshape(-1)
        num_groups = (W.shape[0] + self.group_size - 1) // self.group_size
        padded_size = num_groups * self.group_size

        if padded_size > W.shape[0]:
            W = F.pad(W, (0, padded_size - W.shape[0]))

        W_groups = W.reshape(num_groups, self.group_size)

        qmin = 0
        qmax = 2 ** self.n_bits - 1

        # 计算每组的 min/max
        W_min = W_groups.min(dim=-1, keepdim=True)[0]
        W_max = W_groups.max(dim=-1, keepdim=True)[0]

        # 对称量化
        W_max_abs = W_groups.abs().max(dim=-1, keepdim=True)[0]
        self.scales = (W_max_abs / (qmax / 2)).reshape(num_groups)
        self.qzeros = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """使用前向传播"""
        if self.qweight is None:
            raise ValueError("Must call quantize() first")

        weight = self.qweight.to(self.weight.dtype)
        return F.linear(x, weight, self.module.bias)


def apply_gptq_to_model(
    model: nn.Module,
    config: GPTQConfig,
    calib_dataloader: Any = None,
    verbose: bool = False,
) -> Dict[str, GPTQQuantizer]:
    """对模型应用 GPTQ 量化

    Args:
        model: 要量化的模型
        config: GPTQ 配置
        calib_dataloader: 校准数据加载器
        verbose: 是否打印详细信息

    Returns:
        量化器字典，key 为层名称，value 为 GPTQQuantizer
    """
    quantizers = {}

    # 第一步：收集所有 Linear 层并注册 hook
    layers_to_quantize = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 跳过输出层
            if 'lm_head' in name or 'output' in name:
                if verbose:
                    print(f"Skipping {name} (output layer)")
                continue

            if verbose:
                print(f"Preparing GPTQ for {name}: {module.in_features} -> {module.out_features}")

            quantizer = GPTQQuantizer(module, config, name=name)
            quantizers[name] = quantizer
            layers_to_quantize.append((name, module, quantizer))

            # 注册 hook 收集输入
            def make_hook(q):
                def hook(module, inputs, outputs):
                    # 获取输入
                    if isinstance(inputs, tuple):
                        inp = inputs[0]
                    else:
                        inp = inputs
                    q.add_input(inp)
                return hook

            module.register_forward_hook(make_hook(quantizer))

    # 第二步：运行校准数据收集 Hessian
    if calib_dataloader is not None:
        model.eval()
        with torch.no_grad():
            for batch in calib_dataloader:
                if isinstance(batch, dict):
                    inputs = batch.get('input_ids', batch)
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch

                if isinstance(inputs, torch.Tensor):
                    if inputs.ndim == 2:
                        inputs = inputs.unsqueeze(0)
                    model(inputs.to(next(model.parameters()).device))

        if verbose:
            print(f"Collected Hessian from {len(calib_dataloader)} batches")

    # 第三步：移除 hook 并执行量化
    for name, module, quantizer in layers_to_quantize:
        # 移除 hook
        for hook in module._forward_hooks.values():
            pass  # hooks are automatically managed
        module._forward_hooks.clear()

        # 执行量化
        if verbose:
            print(f"Quantizing {name}...")
        quantizer.quantize()

        # 替换 forward
        original_forward = module.forward
        def make_quantized_forward(q, orig_fwd):
            def forward(x):
                return q.forward(x)
            return forward
        module.forward = make_quantized_forward(quantizer, original_forward)

    if verbose:
        print(f"Applied GPTQ to {len(quantizers)} layers")

    return quantizers


__all__ = [
    "GPTQConfig",
    "GPTQQuantizer",
    "apply_gptq_to_model",
]
