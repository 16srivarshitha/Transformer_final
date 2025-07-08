import torch
torch.cuda.empty_cache()
import torch.nn as nn
from .embeddings import TokenEmbedding
from .embeddings import PositionalEncoding
from encoder import Encoder
from decoder import Decoder

class EnhancedTransformer(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.pad_token_id = tokenizer.pad_token_id
        
        # Embeddings
        self.src_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.tgt_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        
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
            
        self.dropout = nn.Dropout(config.dropout)
        
    def create_mask(self, src, tgt):
        src_mask = (src != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        
        # Causal mask for decoder
        seq_len = tgt.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(tgt.device)
        tgt_mask = tgt_mask & causal_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.create_mask(src, tgt)
        
        # Encoder
        src_emb = self.pos_encoding(self.src_embedding(src))
        src_emb = self.dropout(src_emb)
        encoder_output = self.encoder(src_emb, src_mask)
        
        # Decoder
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt))
        tgt_emb = self.dropout(tgt_emb)
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask)
        
        # Output projection
        output = self.output_projection(decoder_output)
        
        return output