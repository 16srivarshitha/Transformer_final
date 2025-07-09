import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import itertools
from .evaluation_metrics import EvaluationMetrics

from torch.amp import autocast
from torch.amp import GradScaler

class Trainer:
    def __init__(self, model, tokenizer, config, device='cuda', rank=0):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.rank = rank  # Store the process rank
        self.pad_token_id = self.tokenizer.pad_token_id

        self.optimizer = AdamW(
            model.parameters(), 
            lr=1.0,  # set to 1.0 for Noam scheduler
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        
        d_model = self.model.module.src_embedding.d_model

        def lr_lambda(current_step):
            step = current_step + 1
            warmup_steps = self.config.warmup_steps
            return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda, last_epoch=-1)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_token_id, 
            label_smoothing=config.label_smoothing
        )
        self.scaler = GradScaler()
        self.global_step = 0
        self.current_epoch = 0

    def train_epoch(self, train_loader):
        self.model.train()
        train_loader.sampler.set_epoch(self.current_epoch)
        
        total_loss = 0
        accumulation_steps = self.config.accumulation_steps
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]

            with autocast(device_type='cuda', dtype=torch.float16):
                output = self.model(src, tgt_input)
                loss = self.criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
                loss = loss / accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.scaler.step(self.optimizer) 
                self.scaler.update()
                
                self.scheduler.step()
                self.global_step += 1
                self.optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            
            if self.rank == 0 and (batch_idx + 1) % self.config.log_every == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {self.current_epoch+1}, Batch {batch_idx+1}, Step: {self.global_step}, Loss: {loss.item() * accumulation_steps:.4f}, LR: {current_lr:.6f}')
        
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        eval_model = self.model.module
        
        if self.rank == 0:
            evaluator = EvaluationMetrics(self.tokenizer)
            print("\nRunning validation...")
            perplexity = evaluator.calculate_perplexity(eval_model, val_loader, self.device)
            
            num_bleu_batches = 100 
            val_subset_for_bleu = itertools.islice(val_loader, num_bleu_batches)
            predictions, references = evaluator.generate_translations(eval_model, val_subset_for_bleu, self.device, max_length=self.config.max_seq_len)
            bleu_score = evaluator.calculate_bleu(predictions, references)
            return perplexity, bleu_score
        return None, None 

    def train(self, train_loader, val_loader):
        if self.rank == 0:
            print("Starting training...")
        best_perplexity = float('inf')

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            if self.rank == 0:
                print(f"\n--- Epoch {epoch+1}/{self.config.num_epochs} ---")
            
            train_loss = self.train_epoch(train_loader)
            
            if self.rank == 0:
                perplexity, bleu_score = self.validate(val_loader)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                print("-" * 60)
                print(f"Epoch {epoch+1} Summary:")
                print(f"  - Train Loss: {train_loss:.4f} (from all processes)")
                print(f"  - Validation Perplexity: {perplexity:.4f}")
                print(f"  - BLEU Score: {bleu_score:.2f}")
                print(f"  - Current Learning Rate: {current_lr:.6f}")
                print("-" * 60)
                
                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    model_state = self.model.module.state_dict()
                    torch.save(model_state, 'best_model.pth')
                    print(f"New best model saved with perplexity: {best_perplexity:.4f}")
            
            torch.distributed.barrier()