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
        
        # DEBUG: Print mask shapes and values
        # print(f"DEBUG: src_mask shape: {src_mask.shape}, tgt_mask shape: {tgt_mask.shape}")
        # print(f"DEBUG: src_mask sample: {src_mask[0, 0, 0, :10]}")
        # print(f"DEBUG: tgt_mask sample: {tgt_mask[0, 0, :5, :5]}")
        
        src_emb = self.pos_encoding(self.src_embedding(src))
        encoder_output = self.encoder(src_emb, src_mask)

        encoder_output = self.final_encoder_norm(encoder_output)
        
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt))
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask)
        decoder_output = self.final_decoder_norm(decoder_output)
        
        logits = self.output_projection(decoder_output)
        
        # DEBUG: Check output distribution
        # print(f"DEBUG: logits shape: {logits.shape}")
        # print(f"DEBUG: logits range: {logits.min():.3f} to {logits.max():.3f}")
        # print(f"DEBUG: logits std: {logits.std():.3f}")
        
        return logits