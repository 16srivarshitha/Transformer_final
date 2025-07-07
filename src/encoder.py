import torch
import torch.nn as nn
from attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_multi_scale=False):
        super().__init__()
        

            
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
            
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention + residual
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward + residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1, use_multi_scale=True):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, use_multi_scale)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x