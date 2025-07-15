import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, query.size(1), self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, key.size(1), self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, value.size(1), self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Use dtype-appropriate masking value
            mask_value = torch.finfo(scores.dtype).min
            
            # Handle different mask dimensions
            if mask.dim() == 2:  # (batch_size, seq_len) - padding mask
                # Expand to (batch_size, 1, 1, seq_len) for broadcasting
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:  # (batch_size, seq_len, seq_len) - attention mask
                # Expand to (batch_size, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
            # If mask.dim() == 4, it's already the right shape
            
            scores = scores.masked_fill(mask == True, mask_value)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, query.size(1), self.d_model)
        return self.w_o(output)