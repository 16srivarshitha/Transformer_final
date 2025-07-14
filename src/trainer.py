
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

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training Epoch")):
            src, tgt = batch['src'].to(self.device), batch['tgt'].to(self.device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]

            with autocast('cuda', enabled=True, dtype=torch.float16):
                output = self.model(src, tgt_input)
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                loss = loss / accumulation_steps
            
            if batch_idx == 0:
                print("... (your debug output) ...")

            self.scaler.scale(loss).backward()
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
            
        return total_loss / len(train_loader.dataset)

    def validate(self, val_loader):
        self.model.eval()
        print("\n--- Running Validation ---")
        
        num_bleu_batches = 50
        val_subset_for_bleu = itertools.islice(val_loader, num_bleu_batches)

        perplexity = self.evaluator.calculate_perplexity(self.model, val_loader, self.device)
        predictions, references = self.evaluator.generate_translations(self.model, val_subset_for_bleu, self.device)
        bleu_score = self.evaluator.calculate_bleu(predictions, references)
        
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