from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Training params
    batch_size: int = 4
    accumulation_steps: int = 16
    learning_rate: float = 1.0        
    num_epochs: int = 10              
    warmup_steps: int = 1000         
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    
    # Optimizer params
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-9
    weight_decay: float = 0.01
    
    # Scheduler params
    scheduler: str = "warmup_cosine"
    min_lr: float = 1e-6
    
    # Checkpointing
    save_every: int = 5000
    eval_every: int = 2500
    keep_last_n: int = 3
    
    # Logging
    log_every: int = 100
    wandb_project: str = "enhanced-transformer"
    
    # Device
    device: str = "cuda"
    mixed_precision: bool = True
    compile_model: bool = False