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
            lr=1e-4,  # Fixed learning rate
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )

        # def lr_lambda(current_step):
        #     step = current_step + 1
        #     warmup_steps = config.warmup_steps  
            
        #     if step < warmup_steps:
        #         # Linear warmup from 0 to 1
        #         result = step / warmup_steps
        #     else:
        #         # Square root decay after warmup
        #         result = (warmup_steps / step) ** 0.5
            
        #     print(f"Step {step}: warmup_progress={step/warmup_steps:.4f}, final_lr={result * config.learning_rate:.8f}")
        #     return result

        # self.scheduler = LambdaLR(self.optimizer, lr_lambda)

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
            
            # ADD DEBUG CODE HERE
            if batch_idx == 0:  # First batch of each epoch
                print("=== TRAINING DEBUG ===")
                print(f"Encoder input shape: {src.shape}")
                print(f"Decoder input shape: {tgt_input.shape}")
                print(f"Target shape: {tgt_output.shape}")
                
                # Print first sequence
                print(f"German (encoder): {self.tokenizer.decode(src[0].tolist())}")
                print(f"English (decoder input): {self.tokenizer.decode(tgt_input[0].tolist())}")
                print(f"Target: {self.tokenizer.decode(tgt_output[0].tolist())}")
                
                # Check logits
                print(f"Output logits shape: {output.shape}")
                print(f"Max logit index: {output[0, 0].argmax().item()}")
                print(f"Loss: {loss.item() * accumulation_steps:.4f}")
                
                # Check what the model would predict (greedy decoding)
                predicted_ids = output[0].argmax(dim=-1)
                print(f"Predicted tokens: {predicted_ids.tolist()[:10]}...")  # First 10 tokens
                print(f"Predicted text: {self.tokenizer.decode(predicted_ids.tolist())}")
                print("=" * 50)
            
            self.scaler.scale(loss).backward()

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