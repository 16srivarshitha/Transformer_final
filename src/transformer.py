import torch
import torch.nn as nn
from .embeddings import TokenEmbedding, PositionalEncoding 
from .encoder import Encoder
from .decoder import Decoder

class EnhancedTransformer(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.pad_token_id = tokenizer.pad_token_id
        
        self.src_embedding = TokenEmbedding(config.vocab_size, config.d_model, padding_idx=self.pad_token_id)
        self.tgt_embedding = TokenEmbedding(config.vocab_size, config.d_model, padding_idx=self.pad_token_id)
        
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len, dropout=config.dropout)
        
        self.encoder = Encoder(config.n_layers, config.d_model, config.n_heads, config.d_ff, config.dropout)
        self.decoder = Decoder(config.n_layers, config.d_model, config.n_heads, config.d_ff, config.dropout)
        
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        if config.tie_weights:
            self.output_projection.weight = self.tgt_embedding.embedding.weight
            
        self._init_weights()
        
    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'output_projection' in name:
                    # Special initialization for output layer
                    nn.init.normal_(p, mean=0.0, std=0.02)
                else:
                    nn.init.xavier_uniform_(p)

    def create_mask(self, src, tgt):
        """Creates masks for attention mechanisms. True values are masked."""
        # Source padding mask: True for padding tokens
        # Shape: [batch_size, 1, 1, src_seq_len]
        src_mask = (src == self.pad_token_id).unsqueeze(1).unsqueeze(2)

        # Target padding mask: True for padding tokens
        tgt_padding_mask = (tgt == self.pad_token_id).unsqueeze(1) # Shape: [batch_size, 1, tgt_seq_len]
        
        # Causal (look-ahead) mask
        seq_len = tgt.size(1)
        # Shape: [1, tgt_seq_len, tgt_seq_len]
        causal_mask = torch.triu(torch.ones(1, seq_len, seq_len, device=tgt.device), diagonal=1).bool()
        
        # Combine target padding and causal masks
        # The masks are broadcasted together.
        # Shape: [batch_size, 1, tgt_seq_len, tgt_seq_len]
        tgt_mask = tgt_padding_mask.unsqueeze(1) | causal_mask

        return src_mask, tgt_mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if src_mask is None or tgt_mask is None:
            src_mask, tgt_mask = self.create_mask(src, tgt)
        
        src_emb = self.pos_encoding(self.src_embedding(src))
        encoder_output = self.encoder(src_emb, src_mask)
        
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt))
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask)
        
        return self.output_projection(decoder_output)