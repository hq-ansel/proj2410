from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Protocol

import torch
from torch import nn

# import GradualQuantizer
from .gradual import GradualQuantizer, _collect_gradual_quantizers

@dataclass
class ScheduleState:
    """调度状态，包含当前步骤、轮次和指标"""
    step: int  # 当前全局步骤
    epoch: int = 0  # 当前轮次
    metrics: Optional[Dict[str, float]] = None  # 可选的训练指标


class PriorityCalculator(Protocol):
    """
    优先级计算器协议

    计算每个 quantizer 的 group 优先级，返回 Dict[prefix, Tensor[G]]，
    其中值越大表示优先级越高（越早被量化）
    """
    def __call__(
        self,
        quantizers: List[GradualQuantizer],
        state: ScheduleState,
    ) -> Dict[str, torch.Tensor]:
        """计算优先级

        Args:
            quantizers: 渐进式量化器列表
            state: 当前调度状态

        Returns:
            优先级字典，key 为 prefix，value 为每个 group 的优先级张量
        """
        ...


def _get_total_elements(q: "GradualQuantizer") -> int:
    weight = getattr(q, "get_weight_for_priority", lambda: None)()
    if weight is not None:
        try:
            shape_numel = 1
            for dim in weight.shape:
                shape_numel *= int(dim)
            if shape_numel > 0:
                return int(shape_numel)
        except Exception:
            pass
    return int(getattr(q, "_num_elements_full", getattr(q, "_num_elements")))


class BudgetPolicy(Protocol):
    """预算策略协议：根据状态计算要量化的分组数量"""
    def budget(self, state: ScheduleState, total_groups: int) -> int:
        """计算预算（要量化的分组数量）

        Args:
            state: 当前调度状态
            total_groups: 总分组数量

        Returns:
            要量化的分组数量 k
        """
        ...


class RatioBudget:
    """
    基于比例的预算策略（可以是 linear/cosine 等，通过覆盖 ratio() 方法实现）
    """

    def __init__(self, start_ratio: float = 0.0, end_ratio: float = 1.0,
                 total_steps: int = 1000, warmup_steps: int = 0):
        """初始化比例预算策略

        Args:
            start_ratio: 起始比例
            end_ratio: 结束比例
            total_steps: 总训练步数
            warmup_steps: warmup 步数
        """
        self.start_ratio = float(start_ratio)
        self.end_ratio = float(end_ratio)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)

    def ratio(self, state: ScheduleState) -> float:
        """计算当前状态的比例

        Args:
            state: 当前调度状态

        Returns:
            计算得到的比例值
        """
        s = state.step
        if s <= self.warmup_steps:
            p = 0.0
        else:
            denom = max(self.total_steps - self.warmup_steps, 1)
            p = min(max((s - self.warmup_steps) / denom, 0.0), 1.0)
        r = self.start_ratio + p * (self.end_ratio - self.start_ratio)
        return float(min(max(r, 0.0), 1.0))

    def budget(self, state: ScheduleState, total_groups: int) -> int:
        """计算预算（要量化的分组数量）

        Args:
            state: 当前调度状态
            total_groups: 总分组数量

        Returns:
            要量化的分组数量 k，范围 [0, total_groups]
        """
        k = int(total_groups * self.ratio(state))
        return max(min(k, total_groups), 0)


class RatioAssigner(Protocol):
    """比例分配器协议：根据优先级分数与全局比例分配每个 group 的软量化比例"""
    def assign(self, scores: torch.Tensor, ratio: float) -> torch.Tensor:
        """分配每个 group 的比例

        Args:
            scores: 每个 group 的优先级分数，shape 为 [G]
            ratio: 全局比例（0-1）

        Returns:
            每个 group 的比例张量，shape 为 [G]，范围 [0, 1]
        """
        ...


class GroupSelector(Protocol):
    """分组选择器协议：根据优先级分数选择要量化的 groups"""
    def select_mask(self, scores: torch.Tensor, k: int) -> torch.BoolTensor:
        """根据优先级分数选择 mask

        Args:
            scores: 每个组的优先级分数，shape 为 [G]
            k: 要选择的分组数量

        Returns:
            布尔掩码，True 表示对应的 group 要被量化
        """
        ...


def _to_local_tensor(x: torch.Tensor) -> torch.Tensor:
    if hasattr(x, "to_local"):
        return x.to_local()
    return x


class TopKSelector:
    """TopK 选择器：选择分数最高的 k 个 groups"""
    def select_mask(self, scores: torch.Tensor, k: int) -> torch.BoolTensor:
        """选择前 k 个最高分数的 groups

        Args:
            scores: 每个组的优先级分数，shape 为 [G]
            k: 要选择的分组数量

        Returns:
            布尔掩码，前 k 个最高分数为 True
        """
        scores = _to_local_tensor(scores)
        g = scores.numel()
        if k <= 0:
            return torch.zeros(g, dtype=torch.bool, device=scores.device)
        if k >= g:
            return torch.ones(g, dtype=torch.bool, device=scores.device)
        idx = torch.topk(scores, k, largest=True).indices
        idx = _to_local_tensor(idx)
        mask = torch.zeros(g, dtype=torch.bool, device=scores.device)
        mask[idx] = True
        return mask


class ThresholdSelector:
    """阈值选择器：选择分数大于等于阈值的 groups"""
    def __init__(self, threshold: float):
        """初始化阈值选择器

        Args:
            threshold: 选择阈值
        """
        self.threshold = float(threshold)

    def select_mask(self, scores: torch.Tensor, k: int) -> torch.BoolTensor:
        """根据阈值选择 mask

        Args:
            scores: 每个组的优先级分数，shape 为 [G]
            k: 要选择的分组数量（忽略，基于阈值选择）

        Returns:
            布尔掩码，分数 >= threshold 的为 True
        """
        scores = _to_local_tensor(scores)
        # 忽略 k；基于阈值选择
        return (scores >= self.threshold)


def default_scores(total_groups: int, device: torch.device) -> torch.Tensor:
    """
    默认优先级：所有 groups 等优先级（全为 1）

    Args:
        total_groups: 总分组数量
        device: 设备

    Returns:
        全为 1 的张量
    """
    return torch.ones(total_groups, device=device)


def _normalize_scores(
    scores: Optional[torch.Tensor],
    total_groups: int,
    device: torch.device,
) -> torch.Tensor:
    if scores is None:
        return default_scores(total_groups, device=device)
    scores = _to_local_tensor(scores)
    if scores.numel() != total_groups:
        return default_scores(total_groups, device=device)
    return scores.to(device=device)


def _normalize_ratios(
    ratios: Optional[torch.Tensor],
    total_groups: int,
    device: torch.device,
    default_ratio: float,
) -> torch.Tensor:
    if ratios is None or ratios.numel() != total_groups:
        return torch.full((total_groups,), float(default_ratio), device=device)
    ratios = _to_local_tensor(ratios)
    ratios = ratios.to(device=device, dtype=torch.float32)
    return ratios.clamp(0.0, 1.0)


def _budget_ratio(budget_policy: BudgetPolicy, state: ScheduleState, total_groups: int) -> float:
    if total_groups <= 0:
        return 0.0
    ratio_fn = getattr(budget_policy, "ratio", None)
    if callable(ratio_fn):
        try:
            return float(min(max(ratio_fn(state), 0.0), 1.0))
        except Exception:
            pass
    k = budget_policy.budget(state, total_groups)
    return float(min(max(k / max(total_groups, 1), 0.0), 1.0))


class UniformRatioAssigner:
    """所有 groups 使用相同的软量化比例"""
    def assign(self, scores: torch.Tensor, ratio: float) -> torch.Tensor:
        return torch.full_like(scores, float(ratio))


class ScoreProportionalRatioAssigner:
    """按优先级分数比例分配软量化比例（总量约等于 ratio * G）"""
    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)

    def assign(self, scores: torch.Tensor, ratio: float) -> torch.Tensor:
        scores = _to_local_tensor(scores).float()
        scores = scores - scores.min()
        denom = scores.sum()
        if denom <= self.eps:
            return torch.full_like(scores, float(ratio))
        weights = scores / denom
        ratios = weights * float(ratio) * scores.numel()
        return ratios


class UniformPriorityCalculator:
    """
    默认优先级计算器：所有 groups 等优先级
    返回 Dict[prefix, Tensor[G]]，所有值为 1
    """
    def __call__(
        self,
        quantizers: List[GradualQuantizer],
        state: ScheduleState,
    ) -> Dict[str, torch.Tensor]:
        """计算等优先级

        Args:
            quantizers: 渐进式量化器列表
            state: 当前调度状态

        Returns:
            优先级字典，所有值为 1
        """
        return {
            getattr(q, "prefix", str(id(q))): torch.ones(_get_total_elements(q) // q.group_size, device=q._device)
            for q in quantizers
        }


class MagnitudePriorityCalculator:
    """
    基于权重大小（L2 norm）计算优先级
    权重越小（范数越小）优先级越高（越早量化）
    """
    def __call__(
        self,
        quantizers: List[GradualQuantizer],
        state: ScheduleState,
    ) -> Dict[str, torch.Tensor]:
        """基于权重范数计算优先级

        Args:
            quantizers: 渐进式量化器列表
            state: 当前调度状态

        Returns:
            优先级字典，值为每个 group 的 L2 范数
        """
        priorities = {}
        for q in quantizers:
            prefix = getattr(q, "prefix", str(id(q)))
            # 通过回调获取 weight（避免直接持有引用）
            weight = getattr(q, "get_weight_for_priority", lambda: None)()
            if weight is None:
                # 如果无法获取 weight，使用默认全 1 优先级
                num_groups = _get_total_elements(q) // q.group_size
                priorities[prefix] = torch.ones(num_groups, device=q._device)
                continue

            group_size = q.group_size
            num_groups = weight.numel() // group_size

            # Reshape to [G, group_size]
            weight_groups = weight.view(-1, group_size)
            # Compute L2 norm per group (smaller norm -> higher priority)
            group_norms = torch.norm(weight_groups, p=2, dim=1)
            priorities[prefix] = 1.0 / (group_norms + 1e-8)
        return priorities


class QuantizationScheduler:
    """
    核心调度逻辑：
    - 接收优先级计算函数
    - 每步动态计算 quantizer 的优先级
    - 计算预算 k/ratio
    - 使用选择器输出 mask 或 ratio 并设置 quantizer.group_mask/group_ratio
    """
    def __init__(
        self,
        budget_policy: BudgetPolicy,
        selector: GroupSelector,
        priority_calculator: Optional[PriorityCalculator] = None,
        ratio_assigner: Optional[RatioAssigner] = None,
    ):
        """初始化量化调度器

        Args:
            budget_policy: 预算策略
            selector: 分组选择器
            priority_calculator: 优先级计算器，默认为 UniformPriorityCalculator
        """
        self.budget_policy = budget_policy
        self.selector = selector
        self.priority_calculator = priority_calculator or UniformPriorityCalculator()
        self.ratio_assigner = ratio_assigner

    @torch.no_grad()
    def apply(
        self,
        state: ScheduleState,
        quantizers: List,  # List[GradualQuantizer]
    ) -> None:
        """应用调度：动态计算优先级并更新 group_mask/group_ratio

        Args:
            state: 当前调度状态
            quantizers: 渐进式量化器列表
        """
        # 动态计算当前 step 的 priorities
        priorities = self.priority_calculator(quantizers, state)

        for q in quantizers:
            if hasattr(q, "set_current_step"):
                q.set_current_step(state.step)
            prefix = getattr(q, "prefix", str(id(q)))
            # 为此量化器计算总分组数：使用保存的元数据
            total_groups = _get_total_elements(q) // q.group_size
            scores = _normalize_scores(priorities.get(prefix), total_groups, device=q._device)
            if self.ratio_assigner is not None and hasattr(q, "set_group_ratio"):
                ratio = _budget_ratio(self.budget_policy, state, total_groups)
                ratios = self.ratio_assigner.assign(scores, ratio)
                ratios = _normalize_ratios(ratios, total_groups, device=q._device, default_ratio=ratio)
                q.set_group_ratio(ratios)
                q.set_group_mask(None)
            else:
                k = self.budget_policy.budget(state, total_groups)
                mask = self.selector.select_mask(scores, k)
                q.set_group_mask(mask)
                if hasattr(q, "set_group_ratio"):
                    q.set_group_ratio(None)


class GradualQuantController:
    """
    模型集成的薄包装器：收集 quantizers 并提供钩子
    """
    def __init__(self, model: nn.Module, scheduler: QuantizationScheduler):
        """初始化渐进量化控制器

        Args:
            model: PyTorch 模型
            scheduler: 量化调度器
        """
        self.model = model
        self.scheduler = scheduler
        self.quantizers = [m for m in model.modules() if isinstance(m, GradualQuantizer)]

    def on_step_end(self, step: int, epoch: int = 0,
                    metrics: Optional[Dict[str, float]] = None) -> None:
        """在步骤结束时调用：更新所有 quantizers 的 group_mask/group_ratio

        Args:
            step: 当前全局步骤
            epoch: 当前轮次
            metrics: 可选的训练指标
        """
        state = ScheduleState(step=step, epoch=epoch, metrics=metrics)
        self.scheduler.apply(state, self.quantizers)

    def on_epoch_end(self, epoch: int, step: int,
                     metrics: Optional[Dict[str, float]] = None) -> None:
        """在轮次结束时调用：更新所有 quantizers 的 group_mask/group_ratio

        Args:
            epoch: 当前轮次
            step: 当前全局步骤
            metrics: 可选的训练指标
        """
        state = ScheduleState(step=step, epoch=epoch, metrics=metrics)
        self.scheduler.apply(state, self.quantizers)

    def set_priority_calculator(self, priority_calculator: PriorityCalculator) -> None:
        """设置自定义的优先级计算器

        Args:
            priority_calculator: 优先级计算器
        """
        self.scheduler.priority_calculator = priority_calculator


# 使用方式示例
# budget = RatioBudget(start_ratio=0.0, end_ratio=1.0, total_steps=10000, warmup_steps=500)
# selector = TopKSelector()
# priority_calc = UniformPriorityCalculator()  # 或 MagnitudePriorityCalculator() 或自定义
# scheduler = QuantizationScheduler(budget, selector, priority_calc)
# controller = GradualQuantController(model, scheduler)
#
# for step in range(1, 10001):
#     loss = model(batch)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad(set_to_none=True)
#
#     # 自动根据 priority_calculator 计算 priorities 并更新 group_mask
#     controller.on_step_end(step=step)
#
#     # 或者动态切换优先级计算策略
#     if step == 5000:
#         controller.set_priority_calculator(MagnitudePriorityCalculator())
