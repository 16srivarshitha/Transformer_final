import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler 
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from torch.nn.utils.rnn import pad_sequence
from functools import partial

class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]['translation']
        src_text = item['en']
        tgt_text = item['de']

        src_tokens = self.tokenizer.encode(
            src_text, max_length=self.max_length-2, truncation=True, add_special_tokens=False
        )
        tgt_tokens = self.tokenizer.encode(
            tgt_text, max_length=self.max_length-2, truncation=True, add_special_tokens=False
        )

        src_tokens = [self.tokenizer.bos_token_id] + src_tokens + [self.tokenizer.eos_token_id]
        tgt_tokens = [self.tokenizer.bos_token_id] + tgt_tokens + [self.tokenizer.eos_token_id]

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


def create_dataloaders(model_config, training_config, use_ddp=False, rank=0, world_size=1):
    tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'de-en-tokenizer.json')
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run tokenizer script first.")

    if rank == 0:
        print(f"Loading tokenizer from {tokenizer_path}...")
        
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    tokenizer.unk_token = "<unk>"
    pad_id = tokenizer.pad_token_id

    if rank == 0:
        print("Loading and preparing dataset...")
    dataset = load_dataset('opus100', 'de-en', split='train').select(range(training_config.dataset_subset))
    
    train_size = int(0.9 * len(dataset))
    train_data = dataset.select(range(train_size))
    val_data = dataset.select(range(train_size, len(dataset)))

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

    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_config.batch_size, 
        collate_fn=collate_with_pad,
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=2, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=training_config.batch_size, 
        collate_fn=collate_with_pad,
        sampler=val_sampler,
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer