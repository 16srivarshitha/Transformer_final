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
    
    # Compile model if requested and available
    if not args.no_compile and torch.cuda.is_available():
        logger.info("Compiling model for faster execution...")
        try:
            model = torch.compile(model)
            logger.info("Model compilation successful")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Running uncompiled.")
    
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
    
    # Initial validation check
    if args.validate_first:
        run_validation_check(trainer, val_loader, logger)
    
    # Start training
    logger.info("="*60)
    logger.info("           STARTING TRAINING")
    logger.info("="*60)
    
    try:
        trainer.train(train_loader, val_loader)
        logger.info("Training completed successfully!")

        trainer.plot_history()
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save current state
        checkpoint_path = Path(args.save_dir) / "interrupted_checkpoint.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': trainer.global_step,
            'config': training_config
        }, checkpoint_path)
        logger.info(f"Saved interrupted state to: {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()