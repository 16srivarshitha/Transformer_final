import torch
import torch.nn as nn
import math
import time
from sacrebleu.metrics import BLEU

class EvaluationMetrics:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        print(f"BOS: {self.bos_token_id}, EOS: {self.eos_token_id}, PAD: {self.pad_token_id}")
        
    def generate_translations(self, model, dataloader, device, max_length=60):
        model.eval()
        predictions = []
        references = []
        
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        
        with torch.no_grad():
            for batch in dataloader:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                batch_size = src.size(0)
                
                decoder_input = torch.full(
                    (batch_size, 1), 
                    bos_token_id, 
                    dtype=torch.long, 
                    device=device
                )

                finished_sentences = torch.zeros(batch_size, dtype=torch.bool, device=device)

                for _ in range(max_length):
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        output = model(src, decoder_input)
                    
                    next_token_logits = output[:, -1, :]
                    next_token_logits = next_token_logits / 0.8  # temperature
                    next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1).squeeze(1)
                    
                    # Add the predicted token to decoder input
                    decoder_input = torch.cat(
                        [decoder_input, next_token.unsqueeze(1)], 
                        dim=1
                    )
                    
                    # Check if any sequences generated EOS token
                    finished_sentences |= (next_token == eos_token_id)
                    
                    # Break if all sequences are finished
                    if finished_sentences.all():
                        break
                
                # Decode predictions (skip BOS token)
                pred_text = self.tokenizer.batch_decode(decoder_input[:, 1:], skip_special_tokens=True)
                
                # Decode references
                ref_text = self.tokenizer.batch_decode(tgt, skip_special_tokens=True)
                print(f"Sample prediction: {pred_text[0]}")
                print(f"Sample reference: {ref_text[0]}")
                
                predictions.extend(pred_text)
                references.extend(ref_text)
                
        return predictions, references

    def calculate_bleu(self, predictions, references):
        bleu = BLEU()
        # sacrebleu expects references to be a list of lists
        return bleu.corpus_score(predictions, [[ref] for ref in references]).score

    def calculate_perplexity(self, model, dataloader, device):
        model.eval()
        total_loss = 0
        total_tokens = 0

        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction='sum')
        
        with torch.no_grad():
            for batch in dataloader:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(src, tgt_input)
                    # The loss for the entire batch, summed up (due to reduction='sum')
                    loss = criterion(output.view(-1, output.size(-1)), tgt_output.reshape(-1))
                
                total_loss += loss.item()
                total_tokens += (tgt_output != self.pad_token_id).sum().item()
        
        # Avoid division by zero
        if total_tokens == 0:
            return float('inf')
            
        avg_loss = total_loss / total_tokens
        return math.exp(avg_loss)