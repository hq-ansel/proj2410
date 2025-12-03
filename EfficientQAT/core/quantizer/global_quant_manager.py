from typing import Dict, List, Optional
import torch.nn as nn


class ErrorRecord:
    def __init__(self, prefix: str, step: int, err_type: str, value: float):
        self.prefix = prefix
        self.step = step
        self.err_type = err_type  # "input" / "weight" / "grad"
        self.value = value


class ErrorStatsCenter:
    """非常轻量的误差统计中心，占个坑，后续你可以自己扩展。"""
    def __init__(self):
        self.records: List[ErrorRecord] = []

    def add_error(self, prefix: str, step: int, err_type: str, value) -> None:
        if hasattr(value, "detach"):
            value = value.detach().mean().item()
        self.records.append(ErrorRecord(prefix, step, err_type, float(value)))

    def summary(self):
        # 简单示例：按 (prefix, err_type) 做平均
        stats = {}
        for r in self.records:
            key = (r.prefix, r.err_type)
            if key not in stats:
                stats[key] = {"sum": 0.0, "cnt": 0}
            stats[key]["sum"] += r.value
            stats[key]["cnt"] += 1

        result = {}
        for (prefix, err_type), v in stats.items():
            result[(prefix, err_type)] = v["sum"] / max(v["cnt"], 1)
        return result


class GlobalQuantManager:
    _instance: Optional["GlobalQuantManager"] = None

    def __init__(self):
        # {prefix: QuantizerBase}
        self.quantizers: Dict[str, nn.Module] = {}
        self.error_center = ErrorStatsCenter()

    # -------- 单例获取 --------
    @classmethod
    def get(cls) -> "GlobalQuantManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -------- 注册 / 查询 --------
    def register_quantizer(self, prefix: str, quantizer: nn.Module) -> None:
        """在量化器 __init__ 中调用，将自己注册进来。"""
        self.quantizers[prefix] = quantizer

    def unregister_quantizer(self, prefix: str) -> None:
        """如果你会动态创建/销毁量化器，可以用这个。"""
        if prefix in self.quantizers:
            del self.quantizers[prefix]

    def get_quantizer(self, prefix: str) -> Optional[nn.Module]:
        return self.quantizers.get(prefix)

    def all_quantizers(self) -> List[nn.Module]:
        return list(self.quantizers.values())

    def clear(self) -> None:
        """清空注册表和误差统计（例如一个实验结束后调用）。"""
        self.quantizers.clear()
        self.error_center = ErrorStatsCenter()

    # -------- 给优化器 / 外界用：收集 scale / zero_point 参数 --------
    def get_all_scale_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for q in self.quantizers.values():
            # 假设量化器内有属性 `scale`
            if hasattr(q, "scale") and isinstance(q.scale, nn.Parameter):
                params.append(q.scale)
        return params

    def get_all_zero_point_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for q in self.quantizers.values():
            # 假设量化器内有属性 `zero_point`
            if hasattr(q, "zero_point") and isinstance(q.zero_point, nn.Parameter):
                params.append(q.zero_point)
        return params

    def get_param_groups(self, model: nn.Module,
                         base_lr: float,
                         scale_lr: Optional[float] = None,
                         zp_lr: Optional[float] = None):
        """
        小工具：直接给优化器用的 param_groups。
        scale_lr / zp_lr 不传时默认用 base_lr。
        """
        if scale_lr is None:
            scale_lr = base_lr
        if zp_lr is None:
            zp_lr = base_lr

        return [
            {"params": model.parameters(), "lr": base_lr},
            {"params": self.get_all_scale_params(), "lr": scale_lr},
            {"params": self.get_all_zero_point_params(), "lr": zp_lr},
        ]
