# config.py
from dataclasses import dataclass

@dataclass
class QuantConfig:
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
