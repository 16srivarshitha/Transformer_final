import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from src.transformer import EnhancedTransformer
from src.dataset import create_dataloaders
from src.trainer_ddp import Trainer 

def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device

def cleanup_ddp():
    """Destroys the distributed process group."""
    dist.destroy_process_group()

def main():
    """Main training function to be run by each process."""
    rank, world_size, device = setup_ddp()

    if rank == 0:
        print(f"--- DDP Initialized. World size: {world_size} ---")
        print("--- Initializing Configurations ---")
    
    model_config = ModelConfig()
    training_config = TrainingConfig()

    train_loader, val_loader, tokenizer = create_dataloaders(
        model_config,
        training_config,
        use_ddp=True,
        rank=rank,
        world_size=world_size
    )
    
    model_config.vocab_size = tokenizer.get_vocab_size()
    if rank == 0:
        print(f"\nTokenizer loaded. Vocab Size: {model_config.vocab_size}")
        print(f"Using Pad Token ID: {tokenizer.pad_token_id}")

    model = EnhancedTransformer(model_config, tokenizer).to(device)
    model = DDP(model, device_ids=[rank])

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created with {num_params / 1e6:.2f}M parameters.")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config,
        device=device,
        rank=rank
    )

    if rank == 0:
        print("\n" + "="*50)
        print("           STARTING DDP TRAINING")
        print("="*50)

    trainer.train(train_loader, val_loader)

    cleanup_ddp()

if __name__ == '__main__':
    main()