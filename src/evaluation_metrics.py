import torch
import torch.nn as nn
import math
from sacrebleu.metrics import BLEU
from tqdm import tqdm 

class EvaluationMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        print(f"EvaluationMetrics initialized. BOS: {self.bos_token_id}, EOS: {self.eos_token_id}, PAD: {self.pad_token_id}")

    def generate_translations(self, model, dataloader, device, max_length=60):
        model.eval()
        predictions = []
        references = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating Translations"):
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                batch_size = src.size(0)

                decoder_input = torch.full(
                    (batch_size, 1),
                    self.bos_token_id,
                    dtype=torch.long,
                    device=device
                )

                finished_sentences = torch.zeros(batch_size, dtype=torch.bool, device=device)

                for _ in range(max_length - 1): # `-1` to account for BOS token
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        output = model(src, decoder_input)

                    # greedy decoding for stable validation 
                    next_token_logits = output[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1)

                    decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
                    finished_sentences |= (next_token == self.eos_token_id)
                    if finished_sentences.all():
                        break

                pred_text = self.tokenizer.batch_decode(decoder_input, skip_special_tokens=True)
                ref_text = self.tokenizer.batch_decode(tgt, skip_special_tokens=True)

                predictions.extend(pred_text)
                references.extend(ref_text)

        if predictions and references:
            print(f"Sample Prediction: {predictions[0]}")
            print(f"Sample Reference:  {references[0]}")

        return predictions, references

    def calculate_bleu(self, predictions, references):
        bleu = BLEU()
        return bleu.corpus_score(predictions, [references]).score

    def calculate_perplexity(self, model, dataloader, device):
        model.eval()
        total_loss = 0
        total_tokens = 0
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction='sum')

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating Perplexity"):
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(src, tgt_input)
                    loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

                total_loss += loss.item()
                total_tokens += (tgt_output != self.pad_token_id).sum().item()

        if total_tokens == 0: 
            return float('inf')
        return math.exp(total_loss / total_tokens)