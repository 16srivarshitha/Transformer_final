import torch

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from src.transformer import EnhancedTransformer
from src.dataset import create_dataloaders
from src.trainer import Trainer
# from src.evaluation_metrics import EvaluationMetrics  

def main():
    print("--- Initializing Configurations ---")
    model_config = ModelConfig()
    training_config = TrainingConfig()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("\n--- Loading Data ---")
    train_loader, val_loader, tokenizer = create_dataloaders(model_config, training_config)
    print(f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model_config.vocab_size = tokenizer.vocab_size
    print(f"\nTokenizer loaded. Actual Vocab Size: {model_config.vocab_size}")
    print(f"Using Pad Token ID: {tokenizer.pad_token_id}") 
    
    print("\n--- Building Model ---")
    model = EnhancedTransformer(model_config, tokenizer)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {num_params / 1e6:.2f}M parameters.")

    # Move to device first
    model = model.to(device)

    # Then compile (choose one):
    print("Compiling the model for faster execution...")
    model = torch.compile(model)  # OR torch.jit.script(model) for JIT

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