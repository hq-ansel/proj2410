# config.py
"""量化配置类，定义所有量化相关参数"""
from dataclasses import dataclass

@dataclass
class QuantConfig:
    """量化参数配置类

    Attributes:
        quant_type (str): 量化类型 ('uniform_affine' 或 'gradual')
        n_bits (int): 量化位数 (默认: 8)
        group_size (int): 分组大小 (默认: 128)
        clamp_method (str): 截断方法 ('STE' 或 'MAD') (默认: 'STE')
        round_method (str): 舍入方法 ('ste' 或 'highpass') (默认: 'ste')
        stat_quant (bool): 是否统计量化信息 (默认: False)
        iterative_freezing (bool): 是否启用渐进式冻结 (默认: False)
        iterative_freezing_sheduler (str): 冻结调度器类型 ('linear' 或 'step') (默认: 'linear')
        is_tracking (bool): 是否启用量化振荡追踪 (默认: False)
        freeze_momentum (float): 冻结动量值 (默认: 0.004)
        freeze_threshold (float): 冻结阈值 (默认: 0.0)
        interpolate (bool): 是否启用插值 (默认: False)
        lora_rank (int): LoRA 秩 (0 表示不使用 LoRA，常见值为 32) (默认: 0)
        decay_rate (float): 秩衰减正则化率 (默认: 0.01)
        shrinking_ratio (float): 秩衰减正则化收缩比 (默认: 0.5)
        ramp_len (int): per-group 过渡步数，0 表示禁用 (默认: 0)
        ramp_mode (str): 过渡调度函数 ('linear' 或 'sigmoid') (默认: 'linear')
        ramp_sigmoid_a (float): sigmoid 过渡斜率 (默认: 10.0)
    """
    quant_type: str = "uniform_affine"  # "uniform_affine" | "gradual" - 量化类型
    n_bits: int = 8  # 量化位数
    group_size: int = 128  # 分组大小
    clamp_method: str = "STE"  # "STE" | "MAD" - 截断方法
    round_method: str = "ste"  # "ste" | "highpass" - 舍入方法

    stat_quant: bool = False  # 是否统计量化信息

    iterative_freezing: bool = False  # 是否启用渐进式冻结
    iterative_freezing_sheduler: str = "linear"  # "linear" | "step" - 冻结调度器类型

    is_tracking: bool = False  # 是否启用量化振荡追踪

    freeze_momentum: float = 0.004  # 冻结动量值
    freeze_threshold: float = 0.0  # 冻结阈值

    interpolate: bool = False  # 是否启用插值

    lora_rank: int = 0  # 0 表示不使用 LoRA，常见值为 32
    decay_rate: float = 0.01  # 秩衰减正则化率
    shrinking_ratio: float = 0.5  # 秩衰减正则化收缩比
    ramp_len: int = 0  # 渐进混合的过渡步数，0 表示禁用
    ramp_mode: str = "linear"  # "linear" | "sigmoid" - 过渡调度函数
    ramp_sigmoid_a: float = 10.0  # sigmoid 过渡的斜率
