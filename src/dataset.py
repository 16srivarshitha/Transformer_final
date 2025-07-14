import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler 
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from torch.nn.utils.rnn import pad_sequence
from functools import partial

class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, lang_keys=('en', 'de')):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lang_keys = lang_keys

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]  
        
        if 'translation' in item:
            src_text = item['translation'][self.lang_keys[0]]  # Accesses item['translation']['en']
            tgt_text = item['translation'][self.lang_keys[1]]  # Accesses item['translation']['de']
        else:
            src_text = item[self.lang_keys[0]]
            tgt_text = item[self.lang_keys[1]]

        src_tokens = self.tokenizer.encode(
            src_text, max_length=self.max_length-2, truncation=True, add_special_tokens=False
        )
        tgt_tokens = self.tokenizer.encode(
            tgt_text, max_length=self.max_length-2, truncation=True, add_special_tokens=False
        )

        src_tokens = [self.tokenizer.bos_token_id] + src_tokens[1:-1] + [self.tokenizer.eos_token_id]
        tgt_tokens = [self.tokenizer.bos_token_id] + tgt_tokens[1:-1] + [self.tokenizer.eos_token_id]

        return {
            'src': torch.tensor(src_tokens, dtype=torch.long),
            'tgt': torch.tensor(tgt_tokens, dtype=torch.long)
        }

def collate_fn(batch, pad_token_id):
    src_batch, tgt_batch = [], []
    for item in batch:
        src_batch.append(item['src'])
        tgt_batch.append(item['tgt'])
    
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_token_id)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_token_id)
    
    return {'src': src_batch, 'tgt': tgt_batch}

def create_dataloaders(
    model_config, 
    training_config,
    tokenizer_path, 
    use_ddp=False, 
    rank=0, 
    world_size=1,
    dataset_name='bentrevett/multi30k',
    dataset_config=None,
    subset_size=None,
    val_split_fraction=0.1,
    seed=42
):
    
    tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'en-de-tokenizer.json')
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run tokenizer script first.")

    if rank == 0:
        print(f"Loading tokenizer from {tokenizer_path}...")
        
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    pad_id = tokenizer.pad_token_id    

    if rank == 0:
        print(f"Loading dataset '{dataset_name}' ({dataset_config})...")
    
    
    try:

        train_split = 'train'
        val_split = 'validation'
        full_dataset = load_dataset(dataset_name, dataset_config)
        train_data = full_dataset[train_split]
        val_data = full_dataset[val_split]
    except (KeyError, ValueError):

        if rank == 0:
            print("No standard validation split found. Creating one manually.")
        dataset = load_dataset(dataset_name, dataset_config, split='train')
        
        if subset_size is not None and subset_size < len(dataset):
            dataset = dataset.shuffle(seed=seed).select(range(subset_size))

        split_dataset = dataset.train_test_split(test_size=val_split_fraction, shuffle=True, seed=seed)
        train_data = split_dataset['train']
        val_data = split_dataset['test']

    if rank == 0:
        print(f"Using {len(train_data):,} samples for training and {len(val_data):,} for validation.")
        
    train_dataset = TranslationDataset(train_data, tokenizer, model_config.max_seq_len)
    val_dataset = TranslationDataset(val_data, tokenizer, model_config.max_seq_len)
    
    collate_with_pad = partial(collate_fn, pad_token_id=pad_id)
    
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True

    num_workers = getattr(training_config, 'num_workers', 2)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_config.batch_size, 
        collate_fn=collate_with_pad,
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=training_config.batch_size, 
        collate_fn=collate_with_pad,
        sampler=val_sampler,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer