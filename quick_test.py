import torch
from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from src.transformer import EnhancedTransformer
from src.dataset import create_dataloaders
from src.trainer import Trainer

def main():
    print("1. Setting up configurations...")
    model_config = ModelConfig()
    training_config = TrainingConfig()
    training_config.batch_size = 4
    training_config.num_workers = 0  # Disable multi-processing
    
    print("2. Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("3. Loading data...")
    try:
        train_loader, val_loader, tokenizer = create_dataloaders(
            model_config,
            training_config,
            tokenizer_path="en-de-tokenizer.json",
            subset_size=200,  # Small subset
            seed=42
        )
        print(f" Data loaded. Train: {len(train_loader)}, Val: {len(val_loader)} batches")
    except Exception as e:
        print(f" Data loading failed: {e}")
        return
    
    print("4. Creating model...")
    try:
        model_config.vocab_size = len(tokenizer)
        model = EnhancedTransformer(model_config, tokenizer).to(device)
        print(f"✓ Model created")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return
    
    print("5. Testing first batch from train_loader...")
    try:
        first_batch = next(iter(train_loader))
        print(f" Got first training batch: src shape {first_batch['src'].shape}")
    except Exception as e:
        print(f" Failed to get first batch: {e}")
        return
    
    print("6. Testing model forward pass...")
    try:
        src = first_batch['src'].to(device)
        tgt = first_batch['tgt'].to(device)
        tgt_input = tgt[:, :-1]
        
        with torch.no_grad():
            output = model(src, tgt_input)
        print(f" Forward pass successful: output shape {output.shape}")
    except Exception as e:
        print(f" Forward pass failed: {e}")
        return
    
    print("7. Creating trainer...")
    try:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            config=training_config,
            device=device
        )
        print(" Trainer created")
    except Exception as e:
        print(f" Trainer creation failed: {e}")
        return
    
    print("8. Testing one training step...")
    try:
        model.train()
        # Just test the setup, don't actually train
        print(" All setup successful!")
        print("\nNow testing where training hangs...")
        
        # Test if the issue is in train_epoch
        print("9. Starting train_epoch (will stop after first batch)...")
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            print(f"  Processing batch {batch_idx + 1}...")
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
            
            # Forward pass
            output = model(src, tgt_input)
            print(f" Forward pass done for batch {batch_idx + 1}")
            
            # Just test first batch
            if batch_idx == 0:
                print(" First training batch processed successfully!")
                break
                
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED! The issue might be:")
    print("1. Slow data loading (try --num_workers 0)")
    print("2. Long diagnostic output")
    print("3. Input waiting in terminal")
    print("="*50)

if __name__ == '__main__':
    main()