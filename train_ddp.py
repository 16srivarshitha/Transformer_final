import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from src.transformer import EnhancedTransformer
from src.dataset import create_dataloaders 
from src.trainer_ddp import Trainer 


def setup(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' 
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Destroys the distributed process group."""
    dist.destroy_process_group()

def main_worker(rank, world_size, model_config, training_config):
    print(f"--- Running DDP on rank {rank}. ---")
    setup(rank, world_size)

    device = torch.device(f'cuda:{rank}')

    train_loader, val_loader, tokenizer = create_dataloaders(
        model_config,
        training_config,
        use_ddp=True, # We need to tell our dataloader function to use the sampler
        rank=rank,
        world_size=world_size
    )
    
    # This only needs to be done once, but make sure all processes know the size
    model_config.vocab_size = tokenizer.vocab_size
    if rank == 0: # Only print from the main process to avoid clutter
        print(f"\nTokenizer loaded. Vocab Size: {model_config.vocab_size}")
        print(f"Using Pad Token ID: {tokenizer.pad_token_id}")

    # --- Model Building and Wrapping with DDP ---
    model = EnhancedTransformer(model_config, tokenizer).to(device)
    # Wrap the model with DDP. This handles all gradient synchronization.
    model = DDP(model, device_ids=[rank], find_unused_parameters=True) # find_unused_parameters can help with some models

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created with {num_params / 1e6:.2f}M parameters.")

    # Note: torch.compile is not yet fully stable with DDP in all cases.
    # It's safer to omit it for now, or test it carefully.
    # model = torch.compile(model)

    # --- Trainer Initialization and Training ---
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config,
        device=device,
        rank=rank # Pass rank to the trainer for logging
    )

    if rank == 0:
        print("\n" + "="*50)
        print("           STARTING DDP TRAINING")
        print("="*50)

    trainer.train(train_loader, val_loader)

    # --- Cleanup ---
    cleanup()
    if rank == 0:
        print("\n--- Training Finished ---")


# --- Step 3: The Launcher ---
if __name__ == '__main__':
    print("--- Initializing Configurations for DDP ---")
    model_config = ModelConfig()
    training_config = TrainingConfig()

    # world_size is the number of GPUs you have
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs.")

    # mp.spawn will create 'world_size' processes and run the 'main_worker'
    # function in each of them. It passes the rank (0, 1, ...) as the first argument.
    mp.spawn(
        main_worker,
        args=(world_size, model_config, training_config),
        nprocs=world_size,
        join=True
    )