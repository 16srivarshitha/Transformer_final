from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    # Basic transformer params
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 1024
    vocab_size: int = 32000
    
    # # Multi-scale attention params
    # scale_factors: List[int] = None  # [1, 2, 4] for different scales
    # n_scale_heads: int = 4  # heads per scale
    
    # # Adaptive positional encoding params
    # max_relative_position: int = 128
    # adaptive_pos_dim: int = 64
    # learnable_pos: bool = True
    
    # Other params
    label_smoothing: float = 0.1
    tie_weights: bool = True
    
    def __post_init__(self):
        if self.scale_factors is None:
            self.scale_factors = [1, 2]
        
        # Ensure n_heads is divisible by scales
        total_scale_heads = len(self.scale_factors) * self.n_scale_heads
        if self.n_heads % total_scale_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by total scale heads ({total_scale_heads})")