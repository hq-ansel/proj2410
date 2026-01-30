from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class QuantizerArguments:
    """
    CLI/config arguments for weight quantization, plus optimizer knobs for quantizer params.
    """

    quant_type: Literal["uniform_affine", "gradual"] = field(
        default="uniform_affine",
        metadata={"help": "Quantizer type to use: uniform_affine | gradual."},
    )
    n_bits: int = field(
        default=8,
        metadata={"help": "Number of bits for weight quantization."},
    )
    group_size: int = field(
        default=128,
        metadata={"help": "Per-group size for weight quantization; -1 means per-channel."},
    )
    clamp_method: Literal["STE", "MAD"] = field(
        default="STE",
        metadata={"help": "Clamp method for scale (STE or MAD)."},
    )
    round_method: Literal["ste", "highpass"] = field(
        default="ste",
        metadata={"help": "Rounding method used inside quantizer."},
    )
    symmetric: bool = field(
        default=False,
        metadata={"help": "Enable symmetric quantization (zero_point=None)."},
    )
    stat_quant: bool = field(
        default=False,
        metadata={"help": "Whether to record quantization statistics (amax/mean diff)."},
    )
    iterative_freezing: bool = field(
        default=False,
        metadata={"help": "Enable iterative freezing for gradual quantizer."},
    )
    iterative_freezing_sheduler: Literal["linear", "step"] = field(
        default="linear",
        metadata={"help": "Scheduler type for iterative freezing."},
    )
    is_tracking: bool = field(
        default=False,
        metadata={"help": "Track weight oscillation to optionally freeze groups."},
    )
    freeze_momentum: float = field(
        default=0.004,
        metadata={"help": "Momentum for freeze tracker EMA."},
    )
    freeze_threshold: float = field(
        default=0.0,
        metadata={"help": "Threshold for marking a group as stable/freezed."},
    )
    interpolate: bool = field(
        default=False,
        metadata={"help": "Enable interpolation between quantized and float weights (gradual quant)."},
    )
    lora_rank: int = field(
        default=0,
        metadata={"help": "LoRA rank; 0 disables LoRA."},
    )
    decay_rate: float = field(
        default=0.01,
        metadata={"help": "Rank decay regularization rate for LoRA."},
    )
    shrinking_ratio: float = field(
        default=0.5,
        metadata={"help": "Shrinking ratio used in rank decay regularization."},
    )
    ramp_len: int = field(
        default=0,
        metadata={"help": "Ramp length (steps) for per-group mixing; 0 disables."},
    )
    ramp_mode: Literal["linear", "sigmoid"] = field(
        default="linear",
        metadata={"help": "Ramp schedule for mixing: linear | sigmoid."},
    )
    ramp_sigmoid_a: float = field(
        default=10.0,
        metadata={"help": "Slope parameter for sigmoid ramp schedule."},
    )
    quant_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for quantizer params (scale/zero_point); defaults to train.lr."},
    )
    quant_weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for quantizer params (scale/zero_point)."},
    )
    enable_activation_quant: bool = field(
        default=False,
        metadata={"help": "Enable activation quantizer (if supported)."},
    )
    activation_n_bits: int = field(
        default=8,
        metadata={"help": "Number of bits for activation quantization."},
    )
    activation_group_size: int = field(
        default=128,
        metadata={"help": "Per-group size for activation quantization; -1 means per-channel."},
    )
    activation_clamp_method: Literal["STE", "MAD"] = field(
        default="STE",
        metadata={"help": "Clamp method for activation quantizer (STE or MAD)."},
    )
    activation_round_method: Literal["ste", "highpass"] = field(
        default="ste",
        metadata={"help": "Rounding method for activation quantizer."},
    )
    enable_qweight_l2_reg: bool = field(
        default=False,
        metadata={"help": "Enable L2 regularization between quantized and float weights."},
    )
    qweight_l2_reg_lambda: float = field(
        default=0.0,
        metadata={"help": "Regularization coefficient for quantized weight L2 loss."},
    )
    # Gradual quantization parameters
    enable_gradual_quant: bool = field(
        default=False,
        metadata={"help": "Enable gradual quantization with warmup."},
    )
    qat_warmup_steps: int = field(
        default=0,
        metadata={"help": "Warmup steps before enabling full quantization (gradual mode only)."},
    )
    gradual_start_ratio: float = field(
        default=0.0,
        metadata={"help": "Starting quantization ratio for gradual quantization."},
    )
    gradual_end_ratio: float = field(
        default=1.0,
        metadata={"help": "Fraction of training steps to reach full quantization (1.0)."},
    )
    # Priority calculation for gradual quantization
    priority_type: str = field(
        default="uniform",
        metadata={"help": "Priority calculation type: uniform | magnitude | custom"},
    )
    enable_infra: bool = field(
        default=False,
        metadata={"help": "Enable Inference-Ready (Infra) mode using fused kernels."},
    )


__all__ = ["QuantizerArguments"]
