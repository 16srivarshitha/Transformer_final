import torch.nn as nn
from .attention import MultiHeadAttention
import torch
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6) 
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        attn_output = self.self_attention(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_output)
        # Check after first residual
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("NaN/Inf after first residual in EncoderLayer")

        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)
        # Check after second residual
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("NaN/Inf after second residual in EncoderLayer")
        return x

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

