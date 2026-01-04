# tracking.py
"""量化振荡追踪模块，用于追踪量化权重的振荡情况并进行权重冻结"""
from typing import Any
import torch


class TrackOscillation(torch.nn.Module):
    """
    量化振荡追踪器

    主要参考：
    - https://github.com/nbasyl/OFQ/blob/7ed37d1dd33d39395edbf49fcbbc52f678ecf961/src/quantization/quantizer/lsq.py#L111
    - https://github.com/Qualcomm-AI-research/oscillations-qat/blob/9064d8540c1705242f08b864f06661247012ee4d/utils/oscillation_tracking_utils.py#L26

    这是量化器 int_forward 函数的包装器，用于追踪整数域内的振荡情况。
    """

    def __init__(self,
                 momentum: float = 0.01,
                 freeze_threshold: float = 0,
                 use_ema_x_int: bool = True
                 ) -> None:
        """初始化振荡追踪器

        Args:
            momentum: 指数移动平均（EMA）动量
            freeze_threshold: 冻结阈值，至少应为 momentum 值的 2-3 值
            use_ema_x_int: 是否使用 EMA 的 x_int 进行冻结
        """
        super(TrackOscillation, self).__init__()
        self.momentum = momentum

        self.prev_x_int = None  # 上一次的整数量化值
        self.prev_switch_dir = None  # 上一次的切换方向

        # 统计日志
        self.ema_oscillation = None  # EMA 振荡统计
        self.oscillated_sum = None  # 累计振荡次数
        self.total_oscillation = None  # 总振荡张量
        self.iters_since_reset = None  # 自重置以来的迭代次数

        # 权重冻结相关变量
        self.freeze_threshold = freeze_threshold
        self.use_ema_x_int = use_ema_x_int
        self.frozen = None  # 冻结掩码
        self.frozen_x_int = None  # 冻结的整数权重
        self.ema_x_int = None  # 整数权重的 EMA

    def __call__(self,
                 x_int: torch.Tensor,
                 skip_tracking: bool = False,
                 *args: Any,
                 **kwargs: Any
                 ) -> torch.Tensor:
        """前向传播：追踪振荡并选择性冻结权重

        Args:
            x_int: 量化后的整数张量
            skip_tracking: 是否跳过追踪
            *args: 额外参数（未使用）
            **kwargs: 额外关键字参数（未使用）

        Returns:
            处理后的整数张量（可能被冻结）
        """
        # 应用权重冻结
        if self.frozen is not None:
            x_int = ~self.frozen * x_int + self.frozen * self.frozen_x_int

        if skip_tracking:
            return x_int

        with torch.no_grad():
            # 检查是否正确初始化，否则进行初始化
            self.check_init(x_int)

            # 检测 x_int 的差异（使用舍入避免整数不精确性）
            delta_x_int = torch.round(self.prev_x_int - x_int).detach()  # 应该是 {-1, 0, 1}
            switch_dir = torch.sign(delta_x_int)  # 切换方向，{-1, 0, 1}
            # 切换的二值掩码
            switched = delta_x_int != 0

            # 检测振荡（prev_switch_dir * switch_dir == -1）
            oscillated = (self.prev_switch_dir * switch_dir) == -1
            self.ema_oscillation = (
                self.momentum * oscillated + (1 - self.momentum) * self.ema_oscillation
            )

            # 更新切换方向
            self.prev_switch_dir[switched] = switch_dir[switched]
            self.prev_x_int = x_int
            self.oscillated_sum = oscillated.sum()
            self.total_oscillation += oscillated
            self.iters_since_reset += 1

            # 冻结部分权重
            if self.freeze_threshold > 0:
                freeze_weights = self.ema_oscillation > self.freeze_threshold
                self.frozen[freeze_weights] = True  # 标记为冻结
                if self.use_ema_x_int:
                    self.frozen_x_int[freeze_weights] = torch.round(self.ema_x_int[freeze_weights])
                    # 更新 x_int 的 EMA，用于冻结
                    self.ema_x_int = self.momentum * x_int + (1 - self.momentum) * self.ema_x_int
                else:
                    self.frozen_x_int[freeze_weights] = x_int[freeze_weights]

        return x_int

    def check_init(self,
                   x_int: torch.Tensor
                   ) -> None:
        """检查并初始化追踪状态

        Args:
            x_int: 量化后的整数张量
        """
        if self.prev_x_int is None:
            # 初始化 prev_switch_dir 为 0
            self.prev_switch_dir = torch.zeros_like(x_int)
            self.prev_x_int = x_int.detach()
            self.ema_oscillation = torch.zeros_like(x_int)
            self.oscillated_sum = 0
            self.total_oscillation = torch.zeros_like(x_int)
            # print("Init tracking", x_int.shape)
        else:
            assert (
                self.prev_x_int.shape == x_int.shape
            ), "Tracking shape does not match current tensor shape."

        # 初始化权重冻结相关变量
        if self.frozen is None and self.freeze_threshold > 0:
            self.frozen = torch.zeros_like(x_int, dtype=torch.bool)
            self.frozen_x_int = torch.zeros_like(x_int)
            if self.use_ema_x_int:
                self.ema_x_int = x_int.detach().clone()
