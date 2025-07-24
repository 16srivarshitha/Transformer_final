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

        self.final_encoder_norm = nn.LayerNorm(config.d_model)
        self.final_decoder_norm = nn.LayerNorm(config.d_model)
        
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        if config.tie_weights:
            self.output_projection.weight = self.tgt_embedding.embedding.weight
            
        self._init_weights()
        

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_mask(self, src, tgt):
        batch_size, src_len = src.size()
        _, tgt_len = tgt.size()
        
        # Source padding mask: [batch_size, 1, 1, src_len]
        src_mask = (src == self.pad_token_id).unsqueeze(1).unsqueeze(2)
        
        # Target padding mask: [batch_size, 1, 1, tgt_len]
        tgt_padding_mask = (tgt == self.pad_token_id).unsqueeze(1).unsqueeze(1)
        
        # Causal mask: [1, 1, tgt_len, tgt_len]
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Combine masks: [batch_size, 1, tgt_len, tgt_len]
        tgt_mask = tgt_padding_mask | causal_mask
        
        return src_mask, tgt_mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if src_mask is None or tgt_mask is None:
            src_mask, tgt_mask = self.create_mask(src, tgt)
        
        # *** ADD THESE CHECKS ***
        if torch.isnan(src).any():
            print("NaN detected in source input to forward pass!")
        if torch.isnan(tgt).any():
            print("NaN detected in target input to forward pass!")
        
        src_emb = self.pos_encoding(self.src_embedding(src))
        
        # Check embeddings
        if torch.isnan(src_emb).any():
            print("NaN in source embeddings!")
        
        encoder_output = self.encoder(src_emb, src_mask)
        
        # Check encoder output
        if torch.isnan(encoder_output).any():
            print("NaN in encoder output!")
        
        encoder_output = self.final_encoder_norm(encoder_output)
        
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt))
        
        if torch.isnan(tgt_emb).any():
            print("NaN in target embeddings!")
        
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask)
        
        if torch.isnan(decoder_output).any():
            print("NaN in decoder output!")
        
        decoder_output = self.final_decoder_norm(decoder_output)
        logits = self.output_projection(decoder_output)
        
        if torch.isnan(logits).any():
            print("NaN in final logits!")
        
        return logits