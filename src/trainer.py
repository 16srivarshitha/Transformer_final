import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import itertools
import math
from .evaluation_metrics import EvaluationMetrics 
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, tokenizer, config, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.pad_token_id = self.tokenizer.pad_token_id

        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay)

        def lr_lambda(current_step):
            step = current_step + 1
            warmup_steps = config.warmup_steps
            if step < warmup_steps:
                return step / warmup_steps
            return (warmup_steps / step) ** 0.5

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, label_smoothing=self.config.label_smoothing)
        self.scaler = GradScaler()
        self.evaluator = EvaluationMetrics(tokenizer)
        self.global_step = 0

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        accumulation_steps = self.config.accumulation_steps
        self.optimizer.zero_grad()

        # Track loss progression
        loss_values = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training Epoch")):
            src, tgt = batch['src'].to(self.device), batch['tgt'].to(self.device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]

            with autocast('cuda', enabled=True, dtype=torch.float16):
                output = self.model(src, tgt_input)
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                loss = loss / accumulation_steps
            
            # DEBUG: Enhanced logging for first batch and every 100 batches
            if batch_idx == 0 or batch_idx % 100 == 0:
                print(f"\n--- BATCH {batch_idx} DEBUG ---")
                print(f"Loss: {loss.item() * accumulation_steps:.4f}")
                
                # Check model predictions vs targets
                with torch.no_grad():
                    pred_tokens = output[0].argmax(dim=-1)[:10]  # First 10 tokens of first sample
                    target_tokens = tgt_output[0][:10]
                    
                    print(f"Predicted tokens: {pred_tokens.cpu().tolist()}")
                    print(f"Target tokens:    {target_tokens.cpu().tolist()}")
                    
                    # Show actual text
                    pred_text = self.tokenizer.decode(pred_tokens.cpu().tolist(), skip_special_tokens=True)
                    target_text = self.tokenizer.decode(target_tokens.cpu().tolist(), skip_special_tokens=True)
                    print(f"Predicted text: '{pred_text}'")
                    print(f"Target text:    '{target_text}'")
                    
                    # Check if model is learning variety
                    unique_pred_tokens = len(set(pred_tokens.cpu().tolist()))
                    print(f"Unique predicted tokens in sample: {unique_pred_tokens}/10")
                    
                    # Check logits distribution
                    logits_sample = output[0, 0, :]  # First token position
                    top_5_probs, top_5_indices = torch.topk(torch.softmax(logits_sample, dim=-1), 5)
                    print(f"Top 5 token probabilities: {top_5_probs.cpu().tolist()}")
                    print(f"Top 5 token indices: {top_5_indices.cpu().tolist()}")
                    
                    # Check gradient norms
                    total_norm = 0
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    print(f"Total gradient norm: {total_norm:.6f}")
                    
                print("--- END BATCH DEBUG ---\n")

            self.scaler.scale(loss).backward()
            
            # Track loss for analysis
            loss_values.append(loss.item() * accumulation_steps)
            
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * accumulation_steps

        if (len(train_loader)) % accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        # DEBUG: Loss progression analysis
        if len(loss_values) > 10:
            first_10_avg = sum(loss_values[:10]) / 10
            last_10_avg = sum(loss_values[-10:]) / 10
            print(f"\nTRAINING EPOCH SUMMARY:")
            print(f"  First 10 batches avg loss: {first_10_avg:.4f}")
            print(f"  Last 10 batches avg loss: {last_10_avg:.4f}")
            print(f"  Loss improvement: {((first_10_avg - last_10_avg) / first_10_avg * 100):.2f}%")
            
        return total_loss / len(train_loader.dataset)

    def validate(self, val_loader):
        self.model.eval()
        print("\n--- Running Validation ---")
        
        num_bleu_batches = 50
        val_subset_for_bleu = itertools.islice(val_loader, num_bleu_batches)

        perplexity = self.evaluator.calculate_perplexity(self.model, val_loader, self.device)
        predictions, references = self.evaluator.generate_translations(self.model, val_subset_for_bleu, self.device)
        
        # DEBUG: Enhanced validation analysis
        print(f"\nVALIDATION DEBUG:")
        print(f"Total predictions: {len(predictions)}")
        print(f"Total references: {len(references)}")
        
        # Analyze prediction patterns
        empty_preds = sum(1 for p in predictions if len(p.strip()) == 0)
        period_preds = sum(1 for p in predictions if p.strip() == ".")
        single_word_preds = sum(1 for p in predictions if len(p.strip().split()) == 1)
        multi_word_preds = sum(1 for p in predictions if len(p.strip().split()) > 1)
        
        print(f"Prediction Analysis:")
        print(f"  Empty predictions: {empty_preds}/{len(predictions)} ({empty_preds/len(predictions)*100:.1f}%)")
        print(f"  Period-only predictions: {period_preds}/{len(predictions)} ({period_preds/len(predictions)*100:.1f}%)")
        print(f"  Single word predictions: {single_word_preds}/{len(predictions)} ({single_word_preds/len(predictions)*100:.1f}%)")
        print(f"  Multi-word predictions: {multi_word_preds}/{len(predictions)} ({multi_word_preds/len(predictions)*100:.1f}%)")
        
        # Check vocabulary diversity
        all_pred_words = []
        for pred in predictions:
            all_pred_words.extend(pred.split())
        
        unique_words = len(set(all_pred_words))
        total_words = len(all_pred_words)
        
        print(f"Vocabulary Diversity:")
        print(f"  Total words generated: {total_words}")
        print(f"  Unique words: {unique_words}")
        print(f"  Vocabulary diversity ratio: {unique_words/max(total_words, 1):.3f}")
        
        # Show most common generated words
        if all_pred_words:
            from collections import Counter
            word_counts = Counter(all_pred_words)
            most_common = word_counts.most_common(10)
            print(f"  Most common words: {most_common}")
        
        # Sample predictions and references
        print(f"\nSample Predictions:")
        for i in range(min(5, len(predictions))):
            print(f"  Sample {i}:")
            print(f"    Prediction: '{predictions[i]}'")
            print(f"    Reference:  '{references[i]}'")
            print(f"    Pred length: {len(predictions[i].split())}, Ref length: {len(references[i].split())}")
        
        bleu_score = self.evaluator.calculate_bleu(predictions, references)
        
        return perplexity, bleu_score

    def train(self, train_loader, val_loader):
        print("Starting training...")
        best_perplexity = float('inf')

        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{self.config.num_epochs}")
            print(f"{'='*60}")
            
            train_loss = self.train_epoch(train_loader)
            
            with torch.no_grad():
                perplexity, bleu_score = self.validate(val_loader)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1} SUMMARY:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Validation Perplexity: {perplexity:.4f}")
            print(f"  BLEU Score: {bleu_score:.2f}") 
            print(f"  Learning Rate: {current_lr:.7f}")
            print(f"  Global Step: {self.global_step}")
            
            # Progress indicators
            if epoch > 0:
                print(f"  Perplexity change: {(perplexity - best_perplexity):+.4f}")
            
            if perplexity < best_perplexity:
                best_perplexity = perplexity
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"  ✓ New best model saved! (Perplexity: {best_perplexity:.4f})")
            else:
                print(f"  - No improvement (Best: {best_perplexity:.4f})")
            
            print(f"{'='*60}")
            
            # Early stopping check (optional)
            if epoch > 2 and bleu_score > 0.1:  # Some minimal threshold
                print(f"Model showing signs of learning (BLEU > 0.1)!")