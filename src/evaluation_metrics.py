import torch
import torch.nn.functional as F
import math
import time
from sacrebleu.metrics import BLEU

class EvaluationMetrics:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def generate_translations(self, model, dataloader, device, max_len=60):
        model.eval()
        predictions = []
        references = []

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.token_to_id("</s>")
            
        with torch.no_grad():
            for batch in dataloader:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                
                decoder_input = torch.full(
                    (src.size(0), 1), 
                    self.tokenizer.bos_token_id, 
                    dtype=torch.long, 
                    device=device
                )

                for _ in range(max_len):
                    output = model(src, decoder_input)
                    next_token_logits = output[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1)
                    
                    decoder_input = torch.cat(
                        [decoder_input, next_token.unsqueeze(1)], 
                        dim=1
                    )
                    
                    if (next_token == eos_token_id).all():
                        break
                
                pred_text = self.tokenizer.batch_decode(decoder_input, skip_special_tokens=True)
                ref_text = self.tokenizer.batch_decode(tgt, skip_special_tokens=True)
                
                predictions.extend(pred_text)
                references.extend(ref_text)
                
        return predictions, references

    def calculate_bleu(self, predictions, references):
        bleu = BLEU()
        return bleu.corpus_score(predictions, [references]).score

    def calculate_perplexity(self, model, dataloader, device):
        model.eval()
        total_loss = 0
        total_tokens = 0
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        with torch.no_grad():
            for batch in dataloader:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                output = model(src, tgt_input)
                loss = criterion(output.view(-1, output.size(-1)), tgt_output.reshape(-1))
                
                num_non_pad_tokens = (tgt_output != self.tokenizer.pad_token_id).sum().item()
                total_loss += loss.item() * num_non_pad_tokens
                total_tokens += num_non_pad_tokens
        
        avg_loss = total_loss / total_tokens
        return math.exp(avg_loss)
        
    def measure_inference_speed(self, model, dataloader, device, num_batches=20):
        model.eval()
        total_time = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                
                start_time = time.time()
                _ = model(src, tgt[:, :-1])
                end_time = time.time()
                
                total_time += (end_time - start_time)
                total_tokens += src.numel() + tgt.numel()
        
        return total_tokens / total_time  # tokens per second