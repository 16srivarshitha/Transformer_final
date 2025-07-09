import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import os
from torch.nn.utils.rnn import pad_sequence


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

        # First tokenize
        src_tokens = self.tokenizer.encode(
            src_text,
            max_length=self.max_length-2,  # Leave space for BOS/EOS
            truncation=True,
            add_special_tokens=False
        )

        tgt_tokens = self.tokenizer.encode(
            tgt_text,
            max_length=self.max_length-2,  # Leave space for BOS/EOS
            truncation=True,
            add_special_tokens=False
        )

        # Then add BOS/EOS
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

def create_dataloaders(model_config, training_config):

    tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'de-en-tokenizer.json')

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer file not found at {tokenizer_path}. "
            "Please run 'python train_tokenizer.py' in the root directory first."
        )
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    tokenizer.unk_token = "<unk>"

    pad_id = tokenizer.pad_token_id
 
    print("Loading and preparing dataset...")
    dataset = load_dataset('opus100', 'de-en', split='train').select(range(10000))
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_data = dataset.select(range(train_size))
    val_data = dataset.select(range(train_size, train_size + val_size))

    from functools import partial
    collate_with_pad = partial(collate_fn, pad_token_id=pad_id)
    
    # Create datasets
    train_dataset = TranslationDataset(train_data, tokenizer, model_config.max_seq_len)
    val_dataset = TranslationDataset(val_data, tokenizer, model_config.max_seq_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_config.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=collate_with_pad 
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=training_config.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_with_pad
    )
    
    return train_loader, val_loader, tokenizer