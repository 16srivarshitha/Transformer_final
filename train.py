import torch
import sys
import os

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from src.transformer import EnhancedTransformer
from src.dataset import create_dataloaders
from src.trainer import Trainer
from src.evaluation_metrics import EvaluationMetrics

def main():
    print("--- Initializing Configurations ---")
    model_config = ModelConfig()
    training_config = TrainingConfig()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n--- Loading Data ---")
    train_loader, val_loader, tokenizer = create_dataloaders(model_config, training_config)
    print(f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model_config.vocab_size = tokenizer.vocab_size
    print(f"\nTokenizer loaded. Actual Vocab Size: {model_config.vocab_size}")
    print(f"Using Pad Token ID: {tokenizer.pad_token_id}") # Confirming the correct ID is used
    

    print("\n--- Building Model ---")
    model = EnhancedTransformer(model_config, tokenizer).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {num_params / 1e6:.2f}M parameters.")
    
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
    print("Running final evaluation on the best model...")
    
    best_model_path = 'best_model.pth'
    if os.path.exists(best_model_path):

        model = EnhancedTransformer(model_config, tokenizer).to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Best model weights loaded for final evaluation.")
    else:
        print("Warning: No 'best_model.pth' found. Evaluating the last state of the model.")

    evaluator = EvaluationMetrics(tokenizer)
    
    final_perplexity = evaluator.calculate_perplexity(model, val_loader, device)
    inference_speed = evaluator.measure_inference_speed(model, val_loader, device)
    
    print("\n--- Final Results ---")
    print(f"  - Final Validation Perplexity: {final_perplexity:.4f}")
    print(f"  - Inference Speed: {inference_speed:.2f} tokens/sec")
    

if __name__ == '__main__':
    main()