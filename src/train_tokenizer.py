
import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# path for the tokenizer
TOKENIZER_FILE = "de-en-tokenizer.json"

def train_new_tokenizer():
    """
    Trains a new BPE tokenizer on the opus100 de-en dataset
    and saves it to a file.
    """
    print("Loading opus100 dataset for tokenizer training...")
    # Load the dataset
    dataset = load_dataset('opus100', 'de-en', split='train')

    def get_training_corpus(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]['translation']
            yield [item['en'] for item in batch] + [item['de'] for item in batch]

    print("Initializing a new BPE tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    tokenizer.pre_tokenizer = Whitespace()

    print("Training the tokenizer. This may take a few minutes...")
    trainer = BpeTrainer(vocab_size=32000, special_tokens=["<unk>", "<s>", "</s>", "<pad>"])

    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    pad_token_id = tokenizer.token_to_id("<pad>")
    if pad_token_id is not None:
        tokenizer.enable_padding(pad_id=pad_token_id, pad_token="<pad>")
    else:
        print("Warning: '<pad>' token not found after training!")

    tokenizer.save(TOKENIZER_FILE)
    print(f"Tokenizer successfully trained and saved to {TOKENIZER_FILE}")

if __name__ == '__main__':
    # Check if the tokenizer file already exists 
    if not os.path.exists(TOKENIZER_FILE):
        train_new_tokenizer()
    else:
        print(f"Tokenizer file '{TOKENIZER_FILE}' already exists. Skipping training.")