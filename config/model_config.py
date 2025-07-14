from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    # Basic transformer params
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.3
    max_seq_len: int = 1024
    vocab_size: int = 32000
    
    # Other params
    label_smoothing: float = 0.2
    tie_weights: bool = True
    