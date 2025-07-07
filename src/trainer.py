import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from evaluation_metrics import EvaluationMetrics
import itertools

def get_lr(step, d_model, warmup_steps):
    arg1 = step ** -0.5
    arg2 = step * (warmup_steps ** -1.5)
    return (d_model ** -0.5) * min(arg1, arg2)
class Trainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.d_model = model.src_embedding.d_model

        # Optimizer
        self.optimizer = Adam(
            model.parameters(), 
            lr=1.0,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        
        self.step_num = 0
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=config.label_smoothing)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):

            self.step_num += 1
            lr = get_lr(self.step_num, self.d_model, self.config.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            # Prepare target input and output
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input)
            
            # Calculate loss
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % self.config.log_every == 0:
                print(f'Batch {batch_idx}, Step: {self.step_num}, Loss: {loss.item():.4f}, LR: {lr:.6f}')
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, tokenizer):
        self.model.eval()
        
        evaluator = EvaluationMetrics(tokenizer)
        
        print("\nRunning validation...")
        print("Calculating perplexity on the full validation set...")
        perplexity = evaluator.calculate_perplexity(self.model, val_loader, self.device)
        
        num_bleu_batches = 100 
        
        print(f"Generating translations for BLEU score on a subset of {num_bleu_batches} batches...")
        
        val_subset_for_bleu = itertools.islice(val_loader, num_bleu_batches)
        
        predictions, references = evaluator.generate_translations(
            self.model, 
            val_subset_for_bleu, 
            self.device
        )
        bleu_score = evaluator.calculate_bleu(predictions, references)
        
        return perplexity, bleu_score
    
    def train(self, train_loader, val_loader, tokenizer):
        print("Starting training...")
        best_perplexity = float('inf')
        
        for epoch in range(self.config.num_epochs):
            print(f"\n--- Epoch {epoch+1}/{self.config.num_epochs} ---")
            
            train_loss = self.train_epoch(train_loader)
            
            # Now this call is correct because the 'tokenizer' is available
            perplexity, bleu_score = self.validate(val_loader, tokenizer)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  - Current Learning Rate: {current_lr:.6f}")
            
            print("-" * 60)
            print(f"Epoch {epoch+1} Summary:")
            print(f"  - Train Loss: {train_loss:.4f}")
            print(f"  - Validation Perplexity: {perplexity:.4f}")
            print(f"  - BLEU Score: {bleu_score:.2f}")
            print(f"  - Current Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            print("-" * 60)
            
            if perplexity < best_perplexity:
                best_perplexity = perplexity
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"New best model saved with perplexity: {best_perplexity:.4f}")