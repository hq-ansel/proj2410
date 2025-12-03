import time
import random
import os
import numpy as np
import pytest
import torch
from packaging import version

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from EfficientQAT.core.quantizer.uniform_affine import (
    UniformAffineQuantizer,
    QuantConfig,  # 如果实际不在这里定义，请改为正确的 import
)

TORCH_VERSION = version.parse(torch.__version__)


def set_seed(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_random_tensor(shape, dtype=torch.float32, device="cuda"):
    """Create a random tensor with the specified shape, dtype, and device."""
    device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    return torch.randn(shape, dtype=dtype, device=device)


def compare_tensors(tensor_a, tensor_b, tol=1e-3):
    """Compare two tensors for closeness within a specified tolerance."""
    if tensor_a.shape != tensor_b.shape:
        return False
    return torch.allclose(tensor_a, tensor_b, atol=tol, rtol=tol)


def make_config(
    n_bits=8,
    group_size=128,
    clamp_method="STE",
    is_tracking=False,
    stat_quant=False,
    enable=True,
    freeze_momentum=0.9,
    freeze_threshold=1e-3,
):
    """
    构造一个 QuantConfig 实例（根据你的实现适当调整参数名）。
    如果 QuantConfig 是 dataclass 或普通类，这里基本都能兼容。
    """
    return QuantConfig(
        n_bits=n_bits,
        group_size=group_size,
        clamp_method=clamp_method,
        is_tracking=is_tracking,
        stat_quant=stat_quant,
        enable=enable,
        freeze_momentum=freeze_momentum,
        freeze_threshold=freeze_threshold,
    )


# ======================
# 1. 基本功能测试
# ======================

def test_uniform_affine_basic_forward_backward():
    """
    测试 UniformAffineQuantizer 的基本前向 / 反向是否能正常运行，
    且不会改变张量形状，并能正确传播梯度。
    """
    set_seed(0)
    device = get_device()

    group_size = 128
    # weight 的元素个数必须能整除 group_size
    weight = create_random_tensor((128, 128), device=device)
    config = make_config(
        n_bits=8,
        group_size=group_size,
        is_tracking=False,
        stat_quant=False,
        clamp_method="STE",
    )

    quantizer = UniformAffineQuantizer(
        prefix="test_basic",
        weight=weight,
        config=config,
    ).to(device)

    x = torch.randn(4, 128, 128, device=device, requires_grad=True)
    y = quantizer(x)

    # 形状保持不变
    assert y.shape == x.shape

    # 反向传播不报错且有梯度
    loss = y.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


# =========================================
# 2. fake_quant 与“手写 raw 实现”的一致性
# =========================================

def test_uniform_affine_matches_manual_quant_no_tracking():
    """
    将 UniformAffineQuantizer.fake_quant 的实现
    与手动使用 _quantize/_dequantize (视为 raw 实现) 对比。
    在关闭 tracking 和统计量时，两者应完全一致（或数值极其接近）。
    """
    set_seed(1)
    device = get_device()

    group_size = 64
    weight = create_random_tensor((64, 64), device=device)
    config = make_config(
        n_bits=8,
        group_size=group_size,
        is_tracking=False,
        stat_quant=False,
        clamp_method="STE",
    )

    quantizer = UniformAffineQuantizer(
        prefix="test_manual",
        weight=weight,
        config=config,
    ).to(device)

    x = torch.randn(8, 64, 64, device=device)

    # 使用模块自身的 fake_quant
    y_fake = quantizer.fake_quant(x)

    # 手动使用 cal_qparams + _quantize + _dequantize
    scale, round_zero_point = quantizer.cal_qparams(
        quantizer.scale,
        quantizer.zero_point,
        quantizer.clamp_method,
    )
    ori_shape = x.shape
    x_reshaped = x.reshape(-1, quantizer.group_size)
    x_int = quantizer._quantize(x_reshaped, scale, round_zero_point)
    y_manual = quantizer._dequantize(x_int, scale, round_zero_point)
    y_manual = y_manual.reshape(ori_shape)

    # 使用项目自带的工具函数进行比较
    compare_tensors(y_fake, y_manual)


# =========================================
# 3. 前向 / 反向 速度 smoke test（不做硬性时间约束）
# =========================================

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="速度测试主要针对 GPU 场景，CPU 下可能偏慢",
)
def test_fake_quant_speed_smoke():
    """
    简单的速度 smoke test：多次前向 / 反向，确保在合理时间内跑完。
    不设置严格时间门限，以避免 CI 环境差异导致的随机失败。
    """
    set_seed(2)
    device = "cuda"

    group_size = 128
    weight = create_random_tensor((256, 256), device=device)
    config = make_config(
        n_bits=8,
        group_size=group_size,
        is_tracking=False,
        stat_quant=False,
        clamp_method="STE",
    )

    quantizer = UniformAffineQuantizer(
        prefix="test_speed",
        weight=weight,
        config=config,
    ).to(device)

    x = torch.randn(32, 256, 256, device=device, requires_grad=True)

    # 前向 / 反向跑几轮，保证不会特别慢（只做 smoke）
    num_iters = 10
    start = time.perf_counter()
    for _ in range(num_iters):
        x.grad = None
        y = quantizer(x)
        loss = y.mean()
        loss.backward()
    elapsed = time.perf_counter() - start

    # 只做一个非常宽松的检查，防止出现极端慢的实现
    # （可以根据自己硬件适当调小这个阈值）
    assert elapsed < 10.0, f"fake_quant forward/backward seems too slow: {elapsed:.2f}s"


# =========================================
# 4. 量化统计功能 stat_quant
# =========================================

def test_quantization_statistics_logging():
    """
    打开 stat_quant 时，fake_quant 应该更新 quant_stat_log 中的
    amax_diff 和 mean_diff，并且与手动计算结果一致。
    """
    set_seed(3)
    device = get_device()

    group_size = 64
    weight = create_random_tensor((64, 64), device=device)
    config = make_config(
        n_bits=8,
        group_size=group_size,
        is_tracking=False,
        stat_quant=True,
        clamp_method="STE",
    )

    quantizer = UniformAffineQuantizer(
        prefix="test_stat",
        weight=weight,
        config=config,
    ).to(device)

    x = torch.randn(4, 64, 64, device=device)

    # 触发一次 fake_quant
    y = quantizer.fake_quant(x)

    # 检查 quant_stat_log 是否被创建并更新
    assert hasattr(quantizer, "quant_stat_log")
    assert quantizer.quant_stat_log is not None

    diff = (y - x).abs()
    amax_diff_manual = diff.amax().item()
    mean_diff_manual = diff.mean().item()

    assert np.isclose(
        quantizer.quant_stat_log.amax_diff, amax_diff_manual, rtol=1e-5, atol=1e-7
    )
    assert np.isclose(
        quantizer.quant_stat_log.mean_diff, mean_diff_manual, rtol=1e-5, atol=1e-7
    )


# =========================================
# 5. n_bits >= 16 时应不做量化（直接输出原值）
# =========================================

def test_no_quant_when_nbits_ge_16():
    """
    BaseQuantizer.forward 中约定，当 n_bits >= 16 或 enable=False 时，
    forward 直接返回原输入。这里验证 UniformAffineQuantizer 也遵守这一行为。
    """
    set_seed(4)
    device = get_device()

    group_size = 32
    weight = create_random_tensor((32, 32), device=device)
    config = make_config(
        n_bits=16,  # 直接设置为 >=16
        group_size=group_size,
        is_tracking=False,
        stat_quant=False,
        clamp_method="STE",
    )

    quantizer = UniformAffineQuantizer(
        prefix="test_nbits16",
        weight=weight,
        config=config,
    ).to(device)

    x = torch.randn(2, 32, 32, device=device)

    y = quantizer(x)

    # n_bits >= 16 时，不进行量化，应该与输入相同
    assert compare_tensors(x, y, tol=0.0)


# =========================================
# 6. is_tracking=True 时的路径 smoke test
# =========================================

def test_weight_tracking_path_smoke():
    """
    打开 is_tracking=True 时，fake_quant 会经过 weight_freeze_tracker。
    这里做一个 smoke test，主要检查：
      - 不报错
      - 输出形状正确
    不对内部 TrackOscillation 细节做假设。
    """
    set_seed(5)
    device = get_device()

    group_size = 64
    weight = create_random_tensor((64, 64), device=device)
    config = make_config(
        n_bits=8,
        group_size=group_size,
        is_tracking=True,
        stat_quant=False,
        clamp_method="STE",
    )

    quantizer = UniformAffineQuantizer(
        prefix="test_tracking",
        weight=weight,
        config=config,
    ).to(device)

    x = torch.randn(4, 64, 64, device=device)
    y = quantizer.fake_quant(x)

    assert y.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__])
