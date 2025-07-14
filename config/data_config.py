from dataclasses import dataclass
from typing import Optional

@dataclass
class DataConfig:
    # Dataset params
    dataset_name: str = "bentrevett/multi30k"
    source_lang: str = "en"
    target_lang: str = "de"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    
    # Tokenization
    tokenizer_name: str = "sentencepiece"
    vocab_size: int = 32000
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    
    # Preprocessing
    max_length: int = 1024
    min_length: int = 1
    filter_long_sequences: bool = True
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Paths
    data_dir: str = "./data"
    cache_dir: str = "./data/cache"
    tokenizer_path: Optional[str] = None