# config.py
from dataclasses import dataclass

@dataclass
class QuantConfig:
    """Configuration class for quantization parameters.
    
    Attributes:
        quant_type (str): Type of quantization ('uniform_affine' or 'gradual')
        n_bits (int): Number of bits for quantization (default: 8)
        group_size (int): Group size for quantization (default: 128)
        clamp_method (str): Clamping method ('STE' or 'MAD') (default: 'STE')
        round_method (str): Rounding method ('ste' or 'highpass') (default: 'ste')
        stat_quant (bool): Whether to stat quantize info (default: False)
        iterative_freezing (bool): Enable iterative freezing (default: False)
        iterative_freezing_sheduler (str): Freezing scheduler type ('linear' or 'step') (default: 'linear')
        is_tracking (bool): Whether tracking is enabled (default: False)
        freeze_momentum (float): Freeze momentum value (default: 0.004)
        freeze_threshold (float): Freeze threshold value (default: 0.0)
        interpolate (bool): Whether interpolation is enabled (default: False)
        lora_rank (int): LoRA rank (0 means no LoRA, common choice is 32) (default: 0)
        decay_rate (float): Rank decay regularization rate (default: 0.01)
        shrinking_ratio (float): Rank decay regularization shrinking ratio (default: 0.5)
    """
    quant_type: str = "uniform_affine" # "uniform_affine" | "gradual"
    n_bits: int = 8
    group_size: int = 128
    clamp_method: str = "STE"         # "STE" | "MAD"
    round_method: str = "ste"         # "ste" | "highpass"
    
    stat_quant:bool = False      # whether to stat quantize info

    iterative_freezing: bool = False
    iterative_freezing_sheduler: str = "linear"  # "linear" | "step"

    is_tracking : bool = False
    
    freeze_momentum: float = 0.004
    freeze_threshold: float = 0.0
    
    interpolate: bool = False 
    
    lora_rank: int = 0 # 0 means no LoRA common choice is 32
    decay_rate: float = 0.01  # for rank decay regularization
    shrinking_ratio: float = 0.5  # for rank decay regularization
