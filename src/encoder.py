import torch
import torch.nn as nn
from .attention import MultiHeadAttention
import math
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
            
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Pre-norm self-attention + residual
        normed_x = self.norm1(x)
        attn_output = self.self_attention(normed_x, normed_x, normed_x, mask)
        x = x + self.dropout(attn_output)
        
        # Pre-norm feed-forward + residual
        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout(ff_output)
        
        return x

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm for pre-norm architecture
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.final_norm(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx=0, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.embedding(x) * math.sqrt(self.d_model))