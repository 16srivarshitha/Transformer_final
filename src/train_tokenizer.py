import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_new_tokenizer(
    dataset_name, 
    lang_pair, 
    vocab_size, 
    output_file,
    lang_keys=('en', 'de')
):
    print(f"Loading '{dataset_name} ({lang_pair})' for tokenizer training...")
    # Load the dataset for the specified language pair
    dataset = load_dataset(dataset_name, lang_pair, split='train')

    def get_training_corpus(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            # Handle different dataset structures ('translation' key vs direct)
            if 'translation' in batch:
                yield [item[lang_keys[0]] for item in batch['translation']] + \
                      [item[lang_keys[1]] for item in batch['translation']]
            else:
                yield batch[lang_keys[0]] + batch[lang_keys[1]]
    
    print("Initializing a new BPE tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    print(f"Training the tokenizer with vocab size {vocab_size}. This may take a few minutes...")
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<s>", "</s>", "<pad>"])

    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer, length=len(dataset))

    pad_token_id = tokenizer.token_to_id("<pad>")
    if pad_token_id is not None:
        tokenizer.enable_padding(pad_id=pad_token_id, pad_token="<pad>")
    else:
        print("Warning: '<pad>' token not found after training!")

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
        
    tokenizer.save(output_file)
    print(f"Tokenizer successfully trained and saved to {output_file}")


if __name__ == '__main__':

    TOKENIZER_CONFIG = {
        "dataset_name": "multi30k",  
        "lang_pair": "de-en",
        "vocab_size": 32000,
        "output_file": "de-en-tokenizer.json"
    }

    # Check if the tokenizer file already exists
    if not os.path.exists(TOKENIZER_CONFIG["output_file"]):
        train_new_tokenizer(
            dataset_name=TOKENIZER_CONFIG["dataset_name"],
            lang_pair=TOKENIZER_CONFIG["lang_pair"],
            vocab_size=TOKENIZER_CONFIG["vocab_size"],
            output_file=TOKENIZER_CONFIG["output_file"]
        )
    else:
        print(f"Tokenizer file '{TOKENIZER_CONFIG['output_file']}' already exists. Skipping training.")