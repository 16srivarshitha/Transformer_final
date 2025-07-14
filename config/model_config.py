# config/model_config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Basic transformer params
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 512 
    vocab_size: int = 32000 
    
    # Other params
    label_smoothing: float = 0.1
    tie_weights: bool = True