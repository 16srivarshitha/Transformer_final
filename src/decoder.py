import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
            
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6) # or 1e-12, common values
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        # Pre-LN Masked self-attention + residual
        norm_x_self_attn = self.norm1(x) # Apply LayerNorm BEFORE self-attention
        attn_output = self.self_attention(norm_x_self_attn, norm_x_self_attn, norm_x_self_attn, tgt_mask)
        x = x + self.dropout(attn_output) # Residual connection (Add)
        
        # Pre-LN Cross-attention + residual
        norm_x_cross_attn = self.norm2(x) # Apply LayerNorm BEFORE cross-attention
        # Note: encoder_output is already normalized by the Encoder's final LayerNorm 
        cross_attn_output = self.cross_attention(norm_x_cross_attn, encoder_output, encoder_output, src_mask)
        x = x + self.dropout(cross_attn_output) # Residual connection (Add)
        
        # Pre-LN Feed-forward + residual
        norm_x_ff = self.norm3(x) # Apply LayerNorm BEFORE feed-forward
        ff_output = self.feed_forward(norm_x_ff)
        x = x + self.dropout(ff_output) # Residual connection (Add)
        
        return x

class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return x