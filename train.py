import torch
import os
import random
import numpy as np
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import json


from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from src.transformer import EnhancedTransformer
from src.dataset import create_dataloaders
from src.trainer import Trainer
from src.evaluation_metrics import EvaluationMetrics


def validate_dataset_splits(train_loader, val_loader, tokenizer, logger):
    """
    Validate that training and validation splits don't overlap
    """
    logger.info("Validating dataset splits for leakage...")
    
    # Sample a few batches from each split
    train_samples = []
    val_samples = []
    
    # Get training samples
    for i, batch in enumerate(train_loader):
        if i >= 10:  # Check first 10 batches
            break
        for j in range(min(4, batch['src'].size(0))):  # Max 4 samples per batch
            src_text = tokenizer.decode(batch['src'][j].tolist(), skip_special_tokens=True).strip()
            tgt_text = tokenizer.decode(batch['tgt'][j].tolist(), skip_special_tokens=True).strip()
            train_samples.append((src_text.lower(), tgt_text.lower()))
    
    # Get validation samples
    for i, batch in enumerate(val_loader):
        if i >= 10:  # Check first 10 batches
            break
        for j in range(min(4, batch['src'].size(0))):  # Max 4 samples per batch
            src_text = tokenizer.decode(batch['src'][j].tolist(), skip_special_tokens=True).strip()
            tgt_text = tokenizer.decode(batch['tgt'][j].tolist(), skip_special_tokens=True).strip()
            val_samples.append((src_text.lower(), tgt_text.lower()))
    
    # Check for overlaps
    train_set = set(train_samples)
    val_set = set(val_samples)
    overlap = train_set.intersection(val_set)
    
    if overlap:
        logger.error(f"CRITICAL: Found {len(overlap)} overlapping samples between train and validation!")
        for i, (src, tgt) in enumerate(list(overlap)[:3]):  # Show first 3
            logger.error(f"  Overlap {i+1}: '{src[:50]}...' -> '{tgt[:50]}...'")
        return False
    else:
        logger.info(f"✓ Dataset validation passed - no overlap found in sampled data")
        return True

def diagnose_bleu_issue(model, val_loader, tokenizer, device):
    """
    Diagnostic function to understand why BLEU is so high
    """
    print("\n" + "="*60)
    print("DIAGNOSING BLEU SCORE ISSUE")
    print("="*60)
    
    model.eval()
    evaluator = EvaluationMetrics(tokenizer)
    
    # Take just first batch for detailed analysis
    batch = next(iter(val_loader))
    src = batch['src'].to(device)
    tgt = batch['tgt'].to(device)
    
    with torch.no_grad():
        generated = evaluator.greedy_decode(model, src, device)
    
    print(f"Analyzing {src.size(0)} samples from first validation batch:\n")
    
    identical_to_source = 0
    identical_to_target = 0
    high_overlap = 0
    
    for i in range(min(5, src.size(0))):  # Check first 5 samples
        # Get texts
        src_text = tokenizer.decode(src[i].cpu().tolist(), skip_special_tokens=True).strip()
        
        tgt_tokens = tgt[i].cpu().tolist()
        if tgt_tokens[0] == tokenizer.bos_token_id:
            tgt_tokens = tgt_tokens[1:]
        if tokenizer.eos_token_id in tgt_tokens:
            tgt_tokens = tgt_tokens[:tgt_tokens.index(tokenizer.eos_token_id)]
        tgt_text = tokenizer.decode(tgt_tokens, skip_special_tokens=True).strip()
        
        pred_tokens = generated[i].cpu().tolist()
        if tokenizer.eos_token_id in pred_tokens:
            pred_tokens = pred_tokens[:pred_tokens.index(tokenizer.eos_token_id)]
        pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
        
        print(f"Sample {i+1}:")
        print(f"  Source (EN): '{src_text}'")
        print(f"  Target (DE): '{tgt_text}'")
        print(f"  Prediction: '{pred_text}'")
        
        # Check for issues
        if pred_text.lower() == src_text.lower():
            print(f" ISSUE: Prediction identical to source!")
            identical_to_source += 1
        elif pred_text.lower() == tgt_text.lower():
            print(f" ISSUE: Prediction identical to target!")
            identical_to_target += 1
        
        # Check word overlap
        pred_words = set(pred_text.lower().split())
        tgt_words = set(tgt_text.lower().split())
        if len(pred_words) > 0 and len(tgt_words) > 0:
            overlap = len(pred_words.intersection(tgt_words)) / len(tgt_words.union(pred_words))
            if overlap > 0.8:
                print(f"  High word overlap: {overlap:.2f}")
                high_overlap += 1
        
        print()
    
    print("DIAGNOSTIC SUMMARY:")
    print(f"  Identical to source: {identical_to_source}/5")
    print(f"  Identical to target: {identical_to_target}/5") 
    print(f"  High word overlap: {high_overlap}/5")
    
    if identical_to_target > 2:
        print("\n LIKELY CAUSE: Data leakage - model seeing targets during training!")
    elif identical_to_source > 2:
        print("\n LIKELY CAUSE: Model just copying input - not learning translation!")
    elif high_overlap > 3:
        print("\n  POSSIBLE CAUSE: Very similar train/val data or simple dataset!")
    
    print("="*60)

# REST OF YOUR EXISTING FUNCTIONS (setup_logging, set_seed, etc.)
def setup_logging(log_dir="logs", log_level=logging.INFO):
    """Set up logging configuration"""
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description="Train a Transformer model.")
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=10, 
                       help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=5e-4, 
                       help='Base learning rate.')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps.')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Number of warmup steps.')
    
    # Model parameters
    parser.add_argument('--num_layers', type=int, default=6, 
                       help='Number of encoder/decoder layers.')
    parser.add_argument('--d_model', type=int, default=512, 
                       help='Model dimension.')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads.')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate.')
    
    # Data parameters
    parser.add_argument('--dataset_name', type=str, default='bentrevett/multi30k', 
                       help='Dataset to use (e.g., multi30k, opus100).')
    parser.add_argument('--subset_size', type=int, default=None, 
                       help='Use a random subset of the dataset (e.g., 5000).')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length.')
    
    # System parameters
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility.')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers.')
    
    # Checkpoint and resume
    parser.add_argument('--resume_from', type=str, default=None, 
                       help='Path to checkpoint to resume training from.')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints.')
    parser.add_argument('--save_every', type=int, default=1,
                       help='Save checkpoint every N epochs.')
    
    # Optimization
    parser.add_argument('--no_compile', action='store_true', 
                       help='Disable torch.compile for compatibility.')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training.')
    
    # Testing and debugging
    parser.add_argument('--dry_run', action='store_true', 
                       help='Run a quick test on a small subset.')
    parser.add_argument('--validate_first', action='store_true',
                       help='Run validation before training starts.')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging.')
    
    # ADD THIS NEW ARGUMENT FOR DIAGNOSTICS
    parser.add_argument('--diagnose_bleu', action='store_true',
                       help='Run BLEU diagnostic before training.')
    
    return parser.parse_args()

def validate_args(args, logger):
    """Validate command line arguments"""
    errors = []
    
    # Validate ranges
    if args.num_epochs <= 0:
        errors.append("num_epochs must be positive")
    if args.batch_size <= 0:
        errors.append("batch_size must be positive")
    if args.learning_rate <= 0:
        errors.append("learning_rate must be positive")
    if args.num_layers <= 0:
        errors.append("num_layers must be positive")
    if args.d_model <= 0:
        errors.append("d_model must be positive")
    if args.num_heads <= 0:
        errors.append("num_heads must be positive")
    if not (0 <= args.dropout <= 1):
        errors.append("dropout must be between 0 and 1")
    
    # Validate model architecture
    if args.d_model % args.num_heads != 0:
        errors.append(f"d_model ({args.d_model}) must be divisible by num_heads ({args.num_heads})")
    
    # Validate paths
    if args.resume_from and not os.path.exists(args.resume_from):
        errors.append(f"Resume checkpoint not found: {args.resume_from}")
    
    if errors:
        logger.error("Validation errors found:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    logger.info("Argument validation passed")

def setup_directories(args):
    """Create necessary directories"""
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

def save_config(args, model_config, training_config, save_dir):
    """Save configuration to file"""
    config_dict = {
        'args': vars(args),
        'model_config': model_config.__dict__,
        'training_config': training_config.__dict__,
        'timestamp': datetime.now().isoformat()
    }
    
    config_path = Path(save_dir) / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    return config_path

def setup_model_and_data(args, model_config, training_config, device, logger):
    """Set up model and data loaders"""
    logger.info("Loading data...")
    
    try:
        train_loader, val_loader, tokenizer = create_dataloaders(
            model_config, 
            training_config,
            tokenizer_path="en-de-tokenizer.json",
            dataset_name=args.dataset_name,
            subset_size=args.subset_size,
            seed=args.seed
        )
        logger.info(f"Data loaded from '{args.dataset_name}'. "
                   f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Update model config with actual vocab size
    model_config.vocab_size = tokenizer.vocab_size
    logger.info(f"Tokenizer loaded. Vocab Size: {model_config.vocab_size}")
    
    # Create model
    logger.info("Building model...")
    try:
        model = EnhancedTransformer(model_config, tokenizer)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created with {num_params / 1e6:.2f}M parameters")
        
        model = model.to(device)
        
        # Validate model output dimensions
        if model.output_projection.out_features != len(tokenizer):
            logger.warning(f"Model output dim ({model.output_projection.out_features}) != "
                          f"tokenizer vocab size ({len(tokenizer)})")
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        sys.exit(1)
    
    return train_loader, val_loader, tokenizer, model

def setup_optimization(model, args, device, logger):
    """Setup model optimization (compilation, mixed precision, etc.)"""
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        logger.info(f"Resuming training from checkpoint: {args.resume_from}")
        try:
            checkpoint = torch.load(args.resume_from, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                logger.info(f"Resumed from epoch {start_epoch}")
            else:
                # Legacy checkpoint format
                model.load_state_dict(checkpoint)
                logger.info("Loaded legacy checkpoint format")
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            sys.exit(1)
    
    return model, start_epoch

def run_validation_check(trainer, val_loader, logger):
    """Run initial validation to check model setup"""
    logger.info("Running initial validation check...")
    try:
        with torch.no_grad():
            perplexity, bleu_score = trainer.validate(val_loader)
        logger.info(f"Initial validation - Perplexity: {perplexity:.4f}, BLEU: {bleu_score:.4f}")
    except Exception as e:
        logger.error(f"Initial validation failed: {e}")
        sys.exit(1)

def main():
    # Setup
    args = get_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(log_level=log_level)
    
    # Dry run configuration
    if args.dry_run:
        logger.info("!!! RUNNING IN DRY RUN MODE !!!")
        args.subset_size = 200
        args.num_epochs = 2
        args.batch_size = 16
        args.save_every = 100
    
    # Validate arguments
    validate_args(args, logger)
    
    # Setup directories
    setup_directories(args)
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    torch.autograd.set_detect_anomaly(True)
    
    # Initialize configurations
    logger.info("Initializing configurations...")
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Override with command line arguments
    training_config.num_epochs = args.num_epochs
    training_config.batch_size = args.batch_size
    training_config.learning_rate = args.learning_rate
    training_config.accumulation_steps = args.accumulation_steps
    training_config.warmup_steps = args.warmup_steps
    
    model_config.n_layers = args.num_layers
    model_config.d_model = args.d_model
    model_config.n_heads = args.num_heads
    model_config.dropout = args.dropout
    
    # Save configuration
    config_path = save_config(args, model_config, training_config, args.save_dir)
    logger.info(f"Configuration saved to: {config_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
    
    # Setup model and data
    train_loader, val_loader, tokenizer, model = setup_model_and_data(
        args, model_config, training_config, device, logger
    )
    
    if not validate_dataset_splits(train_loader, val_loader, tokenizer, logger):
        logger.error("Dataset validation failed - stopping training")
        sys.exit(1)
    
    # Setup optimization
    model, start_epoch = setup_optimization(model, args, device, logger)
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config,
        device=device
    )
    
    if args.diagnose_bleu:
        logger.info("Running BLEU diagnostic...")
        try:
            diagnose_bleu_issue(model, val_loader, tokenizer, device)
        except Exception as e:
            logger.error(f"Diagnostic failed: {e}")
        
        print("\n" + "="*60)
        print("Diagnostic completed. Continue with training?")
        print("="*60)
        
        # try:
        #     sys.stdout.flush()  # Force flush output
        #     response = input("Type 'y' to continue or 'n' to exit: ").strip().lower()
        #     print(f"DEBUG: You entered '{response}'")  # Debug output
            
        #     if response not in ['y', 'yes']:
        #         logger.info(f"Training cancelled by user (response: '{response}')")
        #         print("Training cancelled. Exiting...")
        #         return
        #     else:
        #         logger.info("User chose to continue with training")
        #         print("Continuing with training...")
                
        # except KeyboardInterrupt:
        #     logger.info("Training cancelled by user (Ctrl+C)")
        #     return
        # except Exception as e:
        #     logger.error(f"Input handling error: {e}")
        #     logger.info("Assuming user wants to continue...")
    
    logger.info("Auto-continuing training after BLEU diagnostic.")
print("Continuing with training...")

if __name__ == '__main__':
    main()