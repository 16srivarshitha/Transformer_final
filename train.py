# train.py

import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from src.transformer import EnhancedTransformer
from src.dataset import create_dataloaders
from src.trainer import Trainer
from src.evaluation_metrics import EvaluationMetrics

def main():
    print("Loading configurations...")
    model_config = ModelConfig()
    training_config = TrainingConfig()

    model_config.vocab_size = 32000 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Creating dataloaders...")
    train_loader, val_loader, tokenizer = create_dataloaders(model_config, training_config)
    print(f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    print("Initializing the model...")
    model = EnhancedTransformer(model_config).to(device)
    print(f"Model initialized. Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    trainer = Trainer(model, training_config, device)
    
    trainer.train(train_loader, val_loader, tokenizer)
    
    print("\n--- Training Finished ---")
    print("Running final evaluation on the best model...")
    
    best_model_path = 'best_model.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Best model weights loaded for final evaluation.")
    else:
        print("Warning: No 'best_model.pth' found. Evaluating the last state of the model.")

    evaluator = EvaluationMetrics(tokenizer)
    
    final_perplexity = evaluator.calculate_perplexity(model, val_loader, device)
    inference_speed = evaluator.measure_inference_speed(model, val_loader, device)
    
    print("\n--- Final Results ---")
    print(f"  - Final Validation Perplexity: {final_perplexity:.4f}")
    print(f"  - Inference Speed: {inference_speed:.2f} tokens/sec")
    
    torch.save(model.state_dict(), 'final_model.pth')
    print("\nFinal model state saved to 'final_model.pth'")

if __name__ == '__main__':
    main()