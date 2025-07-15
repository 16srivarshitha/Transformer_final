# Transformer Model for Machine Translation (from Scratch)

A complete implementation of the Transformer architecture for neural machine translation, built from scratch in PyTorch following the seminal "Attention Is All You Need" paper by Vaswani et al.

## Overview

This project provides a faithful implementation of the original Transformer model, featuring all core components including multi-head self-attention mechanisms, positional encoding, layer normalization, and complete encoder-decoder stacks with residual connections. The model is trained on the Multi30k English-to-German dataset and demonstrates the fundamental principles of modern neural machine translation.

## Key Features

- **Complete Transformer Architecture**: Full implementation of encoder-decoder stacks with multi-head attention
- **Multi-Head Self-Attention**: Custom implementation of scaled dot-product attention mechanism
- **Positional Encoding**: Sinusoidal position embeddings for sequence ordering
- **Layer Normalization**: Pre-layer normalization with residual connections
- **Masked Self-Attention**: Proper masking for decoder self-attention to prevent future token leakage
- **Cross-Attention**: Encoder-decoder attention mechanism for translation
- **Teacher Forcing**: Training strategy for faster convergence
- **Custom Loss Function**: Label smoothing and padding token handling

## Architecture Details

### Encoder
- 6 identical layers
- Multi-head self-attention (8 heads)
- Position-wise feed-forward networks
- Residual connections and layer normalization
- Dropout for regularization

### Decoder
- 6 identical layers
- Masked multi-head self-attention
- Encoder-decoder attention
- Position-wise feed-forward networks
- Residual connections and layer normalization

### Attention Mechanism
- Scaled dot-product attention
- Query, Key, Value projections
- Multi-head parallel processing
- Attention dropout

## Dataset

The model is trained on the Multi30k dataset, which contains:
- **Language pair**: English to German
- **Training examples**: ~29,000 sentence pairs
- **Validation examples**: ~1,014 sentence pairs
- **Test examples**: ~1,000 sentence pairs
- **Domain**: Image captions and descriptions

## Performance

- **BLEU Score**: 0.2851 (after single epoch)
- **Training time**: Approximately 3-4 hours on GPU
- **Model parameters**: ~79.6M parameters
- **Vocabulary size**: ~10,000 tokens per language

## Requirements

```
torch>=1.8.0
torchtext>=0.9.0
numpy>=1.19.0
matplotlib>=3.3.0
tqdm>=4.60.0
sacrebleu>=2.0.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/16srivarshitha/Transformer_using_pytorch.git
cd transformer_using_pytorch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model from scratch:
```bash
python train_tokenizer.py
python train.py 
```

### Evaluation

Evaluate model performance on test set:
```bash
python evaluate.py --model_path checkpoints/best_model.pth --test_data data/test.txt
```

## Project Structure

```
transformer_using_pytorch/
├── config/
│   ├── __init__.py                 # Package initialization
│   ├── data_config.py              # Dataset configuration parameters
│   ├── model_config.py             # Model architecture parameters
│   └── training_config.py          # Training hyperparameters
├── src/
│   ├── attention.py                # Multi-head attention mechanism
│   ├── embeddings.py               # Token and positional embeddings
│   ├── encoder.py                  # Encoder implementation
│   ├── decoder.py                  # Decoder implementation
│   ├── transformer.py              # Main Transformer model
│   ├── dataset.py                  # Dataset loading and preprocessing
│   ├── trainer.py                  # Training loop and utilities
│   └── evaluation_metrics.py       # BLEU score and other metrics
├── train.py                        # Main training script
└── README.md                       # This file


## Implementation Highlights

### Multi-Head Attention
```python
def multi_head_attention(self, query, key, value, mask=None):
    batch_size = query.size(0)
    
    # Linear projections
    Q = self.w_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    K = self.w_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    V = self.w_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    
    # Apply attention
    x, self.attn = self.attention(Q, K, V, mask=mask)
    
    # Concatenate heads
    x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    
    return self.w_o(x)
```

### Positional Encoding
```python
def positional_encoding(self, seq_len, d_model):
    pos = torch.arange(seq_len).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        -(math.log(10000.0) / d_model))
    
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    
    return pe
```

## Training Details

### Hyperparameters
- **Model dimension**: 512
- **Number of heads**: 8
- **Number of layers**: 6
- **Dropout rate**: 0.1
- **Learning rate**: 5e-4
- **Batch size**: 4
- **Accumulation Steps**: 16
- **Optimizer**: Adam with β1=0.9, β2=0.98, ε=1e-9

### Training Strategy
- **Warmup**: 4000 steps with increasing learning rate
- **Label smoothing**: 0.1
- **Gradient clipping**: 1.0
- **Early stopping**: Patience of 5 epochs

## Evaluation Metrics

- **BLEU Score**: Primary metric for translation quality
- **Perplexity**: Model confidence measure
- **Training Loss**: Cross-entropy with label smoothing
- **Validation Loss**: Monitored for early stopping


## Future Improvements
### Visualisations:
- Encoder self-attention maps
- Decoder self-attention maps
- Cross-attention between encoder and decoder
- Token-level attention weights
  
- Implement beam search for better inference
- Add support for different language pairs
- Experiment with different positional encodings
- Implement model compression techniques
- Add support for longer sequences

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." Advances in Neural Information Processing Systems.
2. Multi30k Dataset: Elliott, D., et al. (2016). "Multi30K: Multilingual English-German Image Descriptions."
