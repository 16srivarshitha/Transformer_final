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
            src_text = item['translation'][self.lang_keys[0]]
            tgt_text = item['translation'][self.lang_keys[1]]
        else:
            src_text = item[self.lang_keys[0]]
            tgt_text = item[self.lang_keys[1]]

        src_encoding = self.tokenizer.encode(
            src_text, 
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True
        )
        tgt_encoding = self.tokenizer.encode(
            tgt_text,
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True
        )

        if isinstance(src_encoding, list):
            src_token_ids = src_encoding
            tgt_token_ids = tgt_encoding
        else:
            src_token_ids = src_encoding.ids
            tgt_token_ids = tgt_encoding.ids

        bos_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 1
        eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2

        src_final_tokens = [bos_id] + src_token_ids + [eos_id]
        tgt_final_tokens = [bos_id] + tgt_token_ids + [eos_id]
        
        return {
            'src': torch.tensor(src_final_tokens, dtype=torch.long),
            'tgt': torch.tensor(tgt_final_tokens, dtype=torch.long)
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
    
    # Ensure special tokens are properly set with fallbacks
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 3  # Common pad token ID
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = 1  # Common BOS token ID
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = 2  # Common EOS token ID
    
    # Set the token strings if they're missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens([tokenizer.pad_token_id])[0]
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.convert_ids_to_tokens([tokenizer.bos_token_id])[0]
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens([tokenizer.eos_token_id])[0]
    
    pad_id = tokenizer.pad_token_id    

    if rank == 0:
        print(f"Loading dataset '{dataset_name}' ({dataset_config})...")
        print(f"Special tokens - BOS: {tokenizer.bos_token_id}, EOS: {tokenizer.eos_token_id}, PAD: {tokenizer.pad_token_id}")
    try:
        print(f"Loading '{dataset_name}' dataset...")
        train_data = load_dataset(dataset_name, name=None, split='train')
        val_data = load_dataset(dataset_name, name=None, split='validation')

        if subset_size is not None and subset_size < len(train_data):
            print(f"Using a subset of {subset_size} training samples.")
            train_data = train_data.shuffle(seed=seed + 1).select(range(subset_size))
        if rank == 0:
            print(f"Using {len(train_data):,} samples for training and {len(val_data):,} for validation.")

        print("Converting datasets to lists to ensure stability...")
        train_data_list = list(train_data)
        val_data_list = list(val_data)
        print("Conversion complete.")

        train_dataset = TranslationDataset(train_data_list, tokenizer, model_config.max_seq_len)
        val_dataset = TranslationDataset(val_data_list, tokenizer, model_config.max_seq_len)
            
        collate_with_pad = partial(collate_fn, pad_token_id=pad_id)

    except Exception as e:
        print(f"FATAL: Failed to load or process dataset. Error: {e}")
            
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