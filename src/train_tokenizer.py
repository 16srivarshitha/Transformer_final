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
    print(f"Loading '{dataset_name}' for tokenizer training...")
    
    try:
        dataset = load_dataset(dataset_name, lang_pair, split='train')
        print(f"Dataset loaded successfully with {len(dataset):,} samples")
        
        # Validate dataset structure
        sample = dataset[0]
        if 'translation' in sample:
            print("Dataset structure: translation key found")
            print(f"Sample EN: {sample['translation'][lang_keys[0]][:50]}...")
            print(f"Sample DE: {sample['translation'][lang_keys[1]][:50]}...")
        else:
            print("Dataset structure: direct keys")
            print(f"Sample EN: {sample[lang_keys[0]][:50]}...")
            print(f"Sample DE: {sample[lang_keys[1]][:50]}...")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

    # --- THIS IS THE CORRECTED FUNCTION ---
    def get_training_corpus(batch_size=1000):
        """Generator that yields batches of text for tokenizer training."""
        is_nested = 'translation' in dataset.features

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i: i + batch_size]
            
            if is_nested:
                translation_pairs = batch['translation']
                
                source_sentences = [pair[lang_keys[0]] for pair in translation_pairs]
                target_sentences = [pair[lang_keys[1]] for pair in translation_pairs]
                
                yield source_sentences + target_sentences
            else:
                source_sentences = batch[lang_keys[0]]
                target_sentences = batch[lang_keys[1]]

                yield source_sentences + target_sentences
    
    print("Initializing a new BPE tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    print(f"Training the tokenizer with vocab size {vocab_size}. This may take a few minutes...")
    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        show_progress=True
    )
    num_sentences = len(dataset) * 2 
    
    tokenizer.train_from_iterator(
        get_training_corpus(), 
        trainer=trainer, 
        length=num_sentences 
    )

    # Configure padding
    pad_token_id = tokenizer.token_to_id("<pad>")
    if pad_token_id is not None:
        tokenizer.enable_padding(pad_id=pad_token_id, pad_token="<pad>")
        print(f"Padding enabled with token ID: {pad_token_id}")
    else:
        print("Warning: '<pad>' token not found after training!")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    # Save the tokenizer
    tokenizer.save(output_file)
    print(f"Tokenizer successfully trained and saved to {output_file}")
    
    # Test the tokenizer
    print("\nTesting tokenizer...")
    test_sentences = [
        "Hello world, this is a test sentence.",
        "Hallo Welt, das ist ein Testsatz.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    for test_text in test_sentences:
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded.ids)
        print(f"Original: {test_text}")
        print(f"Tokens: {encoded.tokens[:10]}...")  # Show first 10 tokens
        print(f"IDs: {encoded.ids[:10]}...")  # Show first 10 IDs
        print(f"Decoded: {decoded}")
        print("-" * 50)
    
    # Print tokenizer stats
    print(f"\nTokenizer Statistics:")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens:")
    for token in ["<unk>", "<s>", "</s>", "<pad>"]:
        token_id = tokenizer.token_to_id(token)
        print(f"  {token}: {token_id}")
    
    return True


if __name__ == '__main__':
    
    

    TOKENIZER_CONFIG = {
        "dataset_name": "bentrevett/multi30k",  
        "lang_pair": None,
        "vocab_size": 32000,
        "output_file": "en-de-tokenizer.json"
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