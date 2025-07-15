import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast
from sacrebleu import corpus_bleu
import math
from tqdm import tqdm

class EvaluationMetrics:
    def __init__(self, tokenizer, max_length=150):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        
        print(f"EvaluationMetrics initialized. BOS: {self.bos_token_id}, EOS: {self.eos_token_id}, PAD: {self.pad_token_id}")
    
    def generate_translations(self, model, data_loader, device, debug=True):
        model.eval()
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Generating Translations")):
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                
                # Generate translations using greedy decoding
                generated = self.greedy_decode(model, src, device)
                
                # Convert to text
                for i in range(src.size(0)):
                    # Get reference (remove BOS token, stop at EOS)
                    ref_tokens = tgt[i].cpu().tolist()
                    if ref_tokens[0] == self.bos_token_id:
                        ref_tokens = ref_tokens[1:]  # Remove BOS
                    if self.eos_token_id in ref_tokens:
                        ref_tokens = ref_tokens[:ref_tokens.index(self.eos_token_id)]
                    
                    # Get prediction (stop at EOS, remove padding)
                    pred_tokens = generated[i].cpu().tolist()
                    if self.eos_token_id in pred_tokens:
                        pred_tokens = pred_tokens[:pred_tokens.index(self.eos_token_id)]
                    
                    # Decode to text
                    pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                    ref_text = self.tokenizer.decode(ref_tokens, skip_special_tokens=True)
                    
                    predictions.append(pred_text)
                    references.append(ref_text)
                    
                    # Debug output for first few samples
                    if debug and batch_idx == 0 and i < 3:
                        print(f"\nDEBUG Sample {i}:")
                        print(f"  Source tokens: {src[i].cpu().tolist()[:20]}...")
                        print(f"  Source text: {self.tokenizer.decode(src[i].cpu().tolist(), skip_special_tokens=True)[:100]}...")
                        print(f"  Pred tokens: {pred_tokens[:20]}...")
                        print(f"  Pred text: '{pred_text}'")
                        print(f"  Ref tokens: {ref_tokens[:20]}...")
                        print(f"  Ref text: '{ref_text}'")
                
                # Only process a subset for speed
                if batch_idx >= 49:  # Process 50 batches
                    break
        
        if debug:
            print(f"\nSample Prediction: {predictions[0]}")
            print(f"Sample Reference:  {references[0]}")
        
        return predictions, references
    
    def greedy_decode(self, model, src, device):
        batch_size = src.size(0)
        max_len = self.max_length
        
        # Create src_mask (assuming you have this in your model)
        src_mask = (src != self.pad_token_id)
        
        # Start with BOS token
        generated = torch.full((batch_size, 1), self.bos_token_id, device=device, dtype=torch.long)
        
        for _ in range(max_len - 1):
            # Create target mask for current sequence
            tgt_mask = (generated != self.pad_token_id)
            
            # Get model output
            with torch.no_grad():
                output = model(src, generated, src_mask=src_mask, tgt_mask=tgt_mask)
                
            # Get next token (greedy)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have generated EOS
            if (next_token == self.eos_token_id).all():
                break
        
        return generated
    
    def calculate_bleu(self, predictions, references):
        # Make sure we have strings, not lists
        if not predictions or not references:
            return 0.0
            
        try:
            # BLEU expects list of references for each prediction
            refs_for_bleu = [[ref] for ref in references]
            bleu = corpus_bleu(predictions, refs_for_bleu)
            return bleu.score
        except Exception as e:
            print(f"BLEU calculation failed: {e}")
            return 0.0
    
    def calculate_perplexity(self, model, data_loader, device):
        model.eval()
        total_loss = 0
        total_tokens = 0
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction='sum')
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Calculating Perplexity"):
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                
                tgt_input = tgt[:, :-1]  # Remove last token
                tgt_output = tgt[:, 1:]  # Remove first token (BOS)
                
                output = model(src, tgt_input)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                
                # Count non-padding tokens
                non_pad_tokens = (tgt_output != self.pad_token_id).sum().item()
                
                total_loss += loss.item()
                total_tokens += non_pad_tokens
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return perplexity