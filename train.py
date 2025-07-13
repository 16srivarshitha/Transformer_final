import torch
import os
import random
import numpy as np
import argparse

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from src.transformer import EnhancedTransformer
from src.dataset import create_dataloaders
from src.trainer import Trainer

def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Transformer model.")
    

    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Base learning rate.')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of encoder/decoder layers.')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension.')
    parser.add_argument('--dataset_name', type=str, default='bentrevett/multi30k', help='Dataset to use (e.g., multi30k, opus100).')
    parser.add_argument('--subset_size', type=int, default=None, help='Use a random subset of the dataset (e.g., 5000).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from.')
    parser.add_argument('--no_compile', action='store_true', help='Disable torch.compile for compatibility.')

    return parser.parse_args()

def main():
    args = get_args()
    
    set_seed(args.seed)
    print(f"--- Reproducibility ensured with random seed: {args.seed} ---")

    print("\n--- Initializing Configurations ---")

    model_config = ModelConfig()
    training_config = TrainingConfig()

    training_config.num_epochs = args.num_epochs
    training_config.batch_size = args.batch_size
    training_config.learning_rate = args.learning_rate 
    model_config.n_layers = args.num_layers
    model_config.d_model = args.d_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("\n--- Loading Data ---")
    train_loader, val_loader, tokenizer = create_dataloaders(
        model_config, 
        training_config,
        dataset_name=args.dataset_name,
        subset_size=args.subset_size,
        seed=args.seed
    )
    print(f"Data loaded from '{args.dataset_name}'. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model_config.vocab_size = tokenizer.vocab_size
    print(f"\nTokenizer loaded. Actual Vocab Size: {model_config.vocab_size}")
    
    print("\n--- Building Model ---")
    model = EnhancedTransformer(model_config, tokenizer)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {num_params / 1e6:.2f}M parameters.")

    model = model.to(device)
    print(f"Model vocab size: {model.output_projection.out_features}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Token ID 0: '{tokenizer.decode([0])}'")
    print(f"Token ID 1: '{tokenizer.decode([1])}'")
    print(f"Token ID 2: '{tokenizer.decode([2])}'")

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming training from checkpoint: {args.resume_from}")
        model.load_state_dict(torch.load(args.resume_from, map_location=device))

    if not args.no_compile and torch.cuda.is_available():
        print("Compiling the model for faster execution...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed: {e}. Running uncompiled.")

    print("\n--- Initializing Trainer ---")
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer,
        config=training_config, 
        device=device
    )
    
    print("\n" + "="*50)
    print("           STARTING TRAINING")
    print("="*50)
    trainer.train(train_loader, val_loader)
    
    
    print("\n--- Training Finished ---")

if __name__ == '__main__':
    main()