# gradual.py
"""渐进式量化器，支持根据优先级逐步增加被量化的 groups"""
import torch
from torch import nn
from typing import Dict, List, Optional

from .uniform_affine import UniformAffineQuantizer
from .config import QuantConfig

class GradualMixin:
    """渐进量化混入类，提供 group_mask 相关功能"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantization_position_ratio = 0.0  # 量化位置比例（0-1）
        self.interpolate_ratio = 0.0  # 插值比例

        # 外部调度器可以设置显式 mask
        self.group_mask: Optional[torch.BoolTensor] = None
        # 外部调度器可以设置每个 group 的软量化比例 (0-1)
        self.group_ratio: Optional[torch.Tensor] = None

    def update_position_ratio(self, new_ratio: float) -> None:
        """更新量化位置比例

        Args:
            new_ratio: 新的比例值，范围 [0, 1]
        """
        self.quantization_position_ratio = float(new_ratio)

    def update_interpolate_ratio(self, new_ratio: float) -> None:
        """更新插值比例

        Args:
            new_ratio: 新的插值比例，范围 [0, 1]
        """
        self.interpolate_ratio = float(new_ratio)

    def set_group_mask(self, mask: Optional[torch.BoolTensor]) -> None:
        """设置分组掩码，用于控制哪些 groups 被量化

        Args:
            mask: 分组掩码，shape 为 [num_groups]，dtype 为 bool
                  如果 mask 为 None，则回退到 ratio 逻辑
        """
        self.group_mask = mask

    def clear_group_mask(self) -> None:
        """清除分组掩码，恢复使用 ratio 逻辑"""
        self.group_mask = None

    def set_group_ratio(self, ratios: Optional[torch.Tensor]) -> None:
        """设置每个 group 的软量化比例

        Args:
            ratios: 比例张量，shape 为 [num_groups]，dtype 为 float
                    如果为 None，则回退到 mask/ratio 逻辑
        """
        self.group_ratio = ratios

    def clear_group_ratio(self) -> None:
        """清除每个 group 的软量化比例"""
        self.group_ratio = None

    def _split_quant_groups(self, x: torch.Tensor) -> int:
        """根据比例计算要量化的分组数量

        Args:
            x: 输入张量

        Returns:
            要量化的分组数量，至少为 1
        """
        total_groups = x.shape[0]
        quantized_groups = max(int(total_groups * self.quantization_position_ratio), 1)
        return quantized_groups


class GradualQuantizer(GradualMixin, UniformAffineQuantizer):
    """渐进式量化器：继承混入类和均匀仿射量化器"""
    def __init__(self, prefix: str, weight: torch.Tensor, config: QuantConfig):
        """初始化渐进式量化器

        Args:
            prefix: 量化器前缀标识符
            weight: 权重张量
            config: 量化配置
        """
        super().__init__(prefix, weight, config)
        self.quantization_position_ratio = 0.0
        self.interpolate_ratio = 0.0
        self.group_mask = None
        self.group_ratio = None

        # 用于调度器的标识符
        self.prefix = prefix
        self.ramp_len = max(int(getattr(config, "ramp_len", 0)), 0)
        self.ramp_mode = getattr(config, "ramp_mode", "linear")
        self.ramp_sigmoid_a = float(getattr(config, "ramp_sigmoid_a", 10.0))
        if self.ramp_mode not in ("linear", "sigmoid"):
            self.ramp_mode = "linear"

        # 只保存必要的元数据，不持有 weight 引用
        self._num_elements = weight.numel()  # 总元素数
        self._device = weight.device  # 设备信息
        # 用于 MagnitudePriorityCalculator 获取 weight 的回调（默认返回 None）
        self._get_weight_for_priority = lambda: None
        self._current_step: Optional[int] = None
        self._ramp_start_steps: Dict[int, int] = {}
        self._prev_group_mask: Optional[torch.BoolTensor] = None

    def set_weight_provider(self, provider_fn):
        """设置 weight 提供者函数（由 IntQuantLinear 调用）

        Args:
            provider_fn: 返回当前 weight 张量的函数
        """
        self._get_weight_for_priority = provider_fn

    def get_weight_for_priority(self):
        """获取当前 weight（供 MagnitudePriorityCalculator 使用）"""
        return self._get_weight_for_priority()

    def set_current_step(self, step: int) -> None:
        self._current_step = int(step)

    def _update_ramp_state(self, mask: torch.BoolTensor) -> None:
        if self.ramp_len <= 0:
            self._prev_group_mask = mask.detach().clone()
            self._ramp_start_steps.clear()
            return
        if self._current_step is None:
            self._prev_group_mask = None
            self._ramp_start_steps.clear()
            return

        if self._prev_group_mask is None or self._prev_group_mask.numel() != mask.numel():
            prev = torch.zeros_like(mask, device=mask.device)
        else:
            prev = self._prev_group_mask.to(device=mask.device)

        new_mask = mask & (~prev)
        if new_mask.any():
            for idx in torch.nonzero(new_mask, as_tuple=True)[0].tolist():
                self._ramp_start_steps.setdefault(int(idx), self._current_step)

        if self._ramp_start_steps:
            for idx in list(self._ramp_start_steps.keys()):
                if idx >= mask.numel() or not bool(mask[idx]):
                    del self._ramp_start_steps[idx]

        self._prev_group_mask = mask.detach().clone()

    def _compute_ramp_lambda(self, indices: List[int], device: torch.device) -> torch.Tensor:
        if not indices or self.ramp_len <= 0 or self._current_step is None:
            return torch.ones(len(indices), device=device)

        start_steps = torch.tensor(
            [self._ramp_start_steps.get(int(idx), self._current_step) for idx in indices],
            device=device,
            dtype=torch.float32,
        )
        t = (float(self._current_step) - start_steps) / float(self.ramp_len)
        if self.ramp_mode == "sigmoid":
            lam = torch.sigmoid(self.ramp_sigmoid_a * (t - 0.5))
        else:
            lam = torch.clamp(t, 0.0, 1.0)

        for idx, value in zip(indices, lam.detach().cpu().tolist()):
            if value >= 1.0 - 1e-6:
                self._ramp_start_steps.pop(int(idx), None)
        return lam

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        """假量化：支持按 group_mask 或 group_ratio 部分/软量化

        Args:
            x: 输入张量

        Returns:
            假量化后的张量
        """
        scale, round_zero_point = self.cal_qparams(self.scale, self.zero_point)

        ori_shape = x.shape
        x = x.reshape(-1, self.group_size)  # [G, group_size]
        G = x.shape[0]

        # 1) 软量化比例优先（每个 group 一个比例）
        ratios = self.group_ratio
        if ratios is not None:
            if ratios.numel() != G:
                raise ValueError(f"group_ratio 长度不匹配：得到 {ratios.numel()}，期望 {G}")
            ratios = ratios.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
            mask = ratios > 0
            if not mask.any():
                return x.reshape(ori_shape)

            selected_indices = torch.nonzero(mask, as_tuple=True)[0]
            x_quant = torch.index_select(x, 0, selected_indices)
            selected_scale = torch.index_select(scale, 0, selected_indices)
            if round_zero_point is None:
                selected_zp = None
            else:
                selected_zp = torch.index_select(round_zero_point, 0, selected_indices)

            x_int = self._quantize(x_quant, selected_scale, selected_zp)
            if self.is_tracking:
                x_int = self.weight_freeze_tracker(x_int)
            x_dequant = self._dequantize(x_int, selected_scale, selected_zp)

            selected_ratio = torch.index_select(ratios, 0, selected_indices).view(-1, 1)
            x_mix = x_quant + (x_dequant - x_quant) * selected_ratio

            out = x.clone()
            if selected_indices.numel() > 0:
                out.index_copy_(0, selected_indices, x_mix)
            return out.reshape(ori_shape)

        # 2) 决定 mask：显式 mask 优先
        mask = self.group_mask
        if mask is not None:
            if mask.numel() != G:
                raise ValueError(f"group_mask 长度不匹配：得到 {mask.numel()}，期望 {G}")
            mask = mask.to(device=x.device, dtype=torch.bool)
        else:
            # 3) 回退：基于 ratio -> 前 k 个 groups
            qg = self._split_quant_groups(x)
            if qg <= 0:
                return x.reshape(ori_shape)
            if qg >= G:
                mask = torch.ones(G, device=x.device, dtype=torch.bool)

            # 前 qg 个 groups 量化
            mask = torch.zeros(G, device=x.device, dtype=torch.bool)
            mask[:qg] = True

        # 3) 根据 mask 应用部分量化（支持任意 groups）
        if not mask.any():
            return x.reshape(ori_shape)

        self._update_ramp_state(mask)

        # 部分量化
        selected_indices = torch.nonzero(mask, as_tuple=True)[0]

        # 使用 index_select 保持 2D 形状
        x_quant = torch.index_select(x, 0, selected_indices)

        # 只选择对应的 scale 和 zero_point，避免广播问题
        selected_scale = torch.index_select(scale, 0, selected_indices)
        if round_zero_point is None:
            selected_zp = None
        else:
            selected_zp = torch.index_select(round_zero_point, 0, selected_indices)

        x_int = self._quantize(x_quant, selected_scale, selected_zp)
        if self.is_tracking:
            x_int = self.weight_freeze_tracker(x_int)
        x_dequant = self._dequantize(x_int, selected_scale, selected_zp)

        # 插值逻辑（不启用 ramp 时保持兼容）
        if self.interpolate_ratio > 0.0 and self.ramp_len <= 0:
            r = self.interpolate_ratio
            x_dequant = x_dequant * r + x_quant * (1 - r)

        # scatter 回原位置，使用 index_copy_ 保持形状
        out = x.clone()
        if selected_indices.numel() > 0:
            out.index_copy_(0, selected_indices, x_dequant)

        if self._ramp_start_steps and self.ramp_len > 0 and self._current_step is not None:
            ramp_indices = [idx for idx in self._ramp_start_steps.keys() if idx < G and bool(mask[idx])]
            if ramp_indices:
                ramp_idx_tensor = torch.tensor(ramp_indices, device=x.device, dtype=torch.long)
                x_fp = torch.index_select(x, 0, ramp_idx_tensor)
                x_dequant_ramp = torch.index_select(out, 0, ramp_idx_tensor)
                lam = self._compute_ramp_lambda(ramp_indices, device=x.device).view(-1, 1)
                x_act = x_fp * (1.0 - lam) + x_dequant_ramp * lam
                out.index_copy_(0, ramp_idx_tensor, x_act)
        return out.reshape(ori_shape)


def _collect_gradual_quantizers(module: nn.Module) -> List["GradualQuantizer"]:
    """收集模块中的所有渐进式量化器

    Args:
        module: PyTorch 模块

    Returns:
        渐进式量化器列表
    """
    return [m for m in module.modules() if isinstance(m, GradualQuantizer)]


class GradualQuantContext:
    """
    上下文管理器：根据训练进度更新量化比例

    Usage:
    -------
    >>> manager = GradualQuantContext(model, total_steps=1000, warmup_steps=100)
    >>> with manager as sched:
    ...     for step in range(1, 1001):
    ...         sched.step(step)  # 同步此步骤的比例
    ...         loss = model(input_ids)
    ...         loss.backward()

    比例在 warmup 后从 start_ratio 线性增加到 end_ratio，
    并限制在 [0, 1] 范围内。
    """

    def __init__(
        self,
        module: nn.Module,
        total_steps: int,
        start_ratio: float = 0.0,
        end_ratio: float = 1.0,
        warmup_steps: int = 0,
    ):
        """初始化渐进量化上下文管理器

        Args:
            module: 包含量化器的模块
            total_steps: 总训练步数
            start_ratio: 起始比例
            end_ratio: 结束比例
            warmup_steps: warmup 步数
        """
        if total_steps <= 0:
            raise ValueError("total_steps 必须为正数")
        if warmup_steps < 0:
            raise ValueError("warmup_steps 必须为非负数")
        self.total_steps = total_steps
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.warmup_steps = warmup_steps
        self.quantizers: List[GradualQuantizer] = _collect_gradual_quantizers(module)
        self._orig: List[float] = []

    def __enter__(self):
        """进入上下文，保存原始比例"""
        self._orig = [q.quantization_position_ratio for q in self.quantizers]
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """退出上下文，恢复原始比例"""
        for q, ratio in zip(self.quantizers, self._orig):
            q.update_position_ratio(ratio)

    def _compute_ratio(self, step: int) -> float:
        """计算当前步骤的量化比例值

        Args:
            step (int): 当前训练步骤数

        Returns:
            float: 计算得到的比例值，范围在 0.0 到 1.0 之间
        """
        if step <= self.warmup_steps:
            progress = 0.0
        else:
            denom = max(self.total_steps - self.warmup_steps, 1)
            progress = min(max((step - self.warmup_steps) / denom, 0.0), 1.0)
        ratio = self.start_ratio + progress * (self.end_ratio - self.start_ratio)
        return float(min(max(ratio, 0.0), 1.0))

    def step(self, step: int) -> None:
        """更新所有追踪的量化器的当前全局步骤的比例

        Args:
            step (int): 当前全局步骤
        """
        ratio = self._compute_ratio(step)
        for q in self.quantizers:
            q.update_position_ratio(ratio)
