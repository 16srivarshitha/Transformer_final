import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import itertools
from .evaluation_metrics import EvaluationMetrics

from torch.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, tokenizer, config, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.pad_token_id = self.tokenizer.pad_token_id

        self.optimizer = AdamW(
            model.parameters(), 
            lr=1.0, 
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )

        d_model = self.model.src_embedding.d_model

        def lr_lambda(current_step):
            step = current_step + 1
            warmup_steps = self.config.warmup_steps  # Should be 4000
            d_model = self.model.src_embedding.d_model
            
            # Original paper formula
            arg1 = step ** -0.5
            arg2 = step * (warmup_steps ** -1.5)
            
            
            base_lr = 1.0 
            result = base_lr * (d_model ** -0.5) * min(arg1, arg2)
            
            print(f"Step {step}: arg1={arg1:.8f}, arg2={arg2:.8f}, min={min(arg1, arg2):.8f}, final_lr={result:.8f}")
            return result

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_token_id, 
            label_smoothing=self.config.label_smoothing
        )
        
        self.scaler = GradScaler()
        self.global_step = 0

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        accumulation_steps = self.config.accumulation_steps
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            with autocast('cuda', enabled=True, dtype=torch.float16):
                output = self.model(src, tgt_input)
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                loss = loss / accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update learning rate
                self.scheduler.step()
                self.global_step += 1
                
                # Reset gradients for the next accumulation cycle
                self.optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            
            if (batch_idx + 1) % self.config.log_every == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Batch {batch_idx+1}/{len(train_loader)}, Step: {self.global_step}, Loss: {loss.item() * accumulation_steps:.4f}, LR: {current_lr:.7f}')
        
        return total_loss / len(train_loader.dataset)

    def validate(self, val_loader):
        self.model.eval()
 
        evaluator = EvaluationMetrics(self.tokenizer)
        print("\nRunning validation...")
        
 
        perplexity = evaluator.calculate_perplexity(self.model, val_loader, self.device)
        
 
        num_bleu_batches = 50 
        val_subset_for_bleu = itertools.islice(val_loader, num_bleu_batches)
        predictions, references = evaluator.generate_translations(self.model, val_subset_for_bleu, self.device, max_length=512)
        bleu_score = evaluator.calculate_bleu(predictions, references)
        
        return perplexity, bleu_score
    
    def train(self, train_loader, val_loader):
        print("Starting training...")
        best_perplexity = float('inf')

        for epoch in range(self.config.num_epochs):
            print(f"\n--- Epoch {epoch+1}/{self.config.num_epochs} ---")
            train_loss = self.train_epoch(train_loader)
            
            with torch.no_grad():
                perplexity, bleu_score = self.validate(val_loader)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print("-" * 60)
            print(f"Epoch {epoch+1} Summary:")
            print(f"  - Train Loss: {train_loss:.4f}")
            print(f"  - Validation Perplexity: {perplexity:.4f}")
            print(f"  - BLEU Score: {bleu_score:.2f}")
            print(f"  - Current Learning Rate: {current_lr:.7f}")
            print("-" * 60)
            
            if perplexity < best_perplexity:
                best_perplexity = perplexity
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"New best model saved with perplexity: {best_perplexity:.4f}")