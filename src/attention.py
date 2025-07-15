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
        batch_size, seq_len, _ = query.size()
        
        # Linear transformations
        Q = self.w_q(query)  # [batch_size, seq_len, d_model]
        K = self.w_k(key)    # [batch_size, seq_len, d_model]
        V = self.w_v(value)  # [batch_size, seq_len, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, n_heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            # Handle different mask shapes
            if mask.dim() == 2:  # [batch_size, seq_len] - padding mask
                # Expand to [batch_size, 1, 1, seq_len] for broadcasting
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:  # [batch_size, seq_len, seq_len] - causal mask
                # Expand to [batch_size, 1, seq_len, seq_len] for broadcasting
                mask = mask.unsqueeze(1)
            
            scores = scores.masked_fill(mask == True, -1e4)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # [batch_size, n_heads, seq_len, d_k]
        
        # Reshape back to original dimensions
        output = output.transpose(1, 2).contiguous()  # [batch_size, seq_len, n_heads, d_k]
        output = output.view(batch_size, seq_len, self.d_model)  # [batch_size, seq_len, d_model]
        
        return self.w_o(output)