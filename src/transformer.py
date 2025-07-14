import torch
torch.cuda.empty_cache()
import torch.nn as nn
from .embeddings import TokenEmbedding
from .embeddings import PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder

class EnhancedTransformer(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.pad_token_id = tokenizer.pad_token_id
        
        # Embeddings
        self.src_embedding = TokenEmbedding(config.vocab_size, config.d_model, config.dropout)
        self.tgt_embedding = TokenEmbedding(config.vocab_size, config.d_model, config.dropout)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        
        # Encoder and Decoder
        self.encoder = Encoder(
            config.n_layers, 
            config.d_model, 
            config.n_heads, 
            config.d_ff, 
            config.dropout
        )
        
        self.decoder = Decoder(
            config.n_layers,
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        # Tie weights if specified
        if config.tie_weights:
            self.output_projection.weight = self.tgt_embedding.embedding.weight
            
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
        
    def create_mask(self, src, tgt):
        batch_size = src.size(0)
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)
        
        # Shape: [batch_size, 1, 1, src_seq_len]
        src_mask = (src != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        
        # Target padding mask: 1 for valid tokens, 0 for padding
        # Shape: [batch_size, 1, tgt_seq_len]
        tgt_padding_mask = (tgt != self.pad_token_id).unsqueeze(1)
        
        # Causal mask for decoder: prevent looking at future tokens
        # Shape: [tgt_seq_len, tgt_seq_len]
        causal_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, device=tgt.device))
        
        # Combine padding and causal masks
        # Shape: [batch_size, 1, tgt_seq_len, tgt_seq_len]
        tgt_mask = tgt_padding_mask.unsqueeze(-1) & causal_mask.unsqueeze(0)
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.create_mask(src, tgt)
        
        # Encoder
        src_emb = self.pos_encoding(self.src_embedding(src))
        encoder_output = self.encoder(src_emb, src_mask)
        
        # Decoder
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt))
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask)
        
        # Output projection
        output = self.output_projection(decoder_output)
        
        return output