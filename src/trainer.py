import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import itertools
import math
import random
from .evaluation_metrics import EvaluationMetrics 
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import logging
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, tokenizer, config, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.pad_token_id = self.tokenizer.pad_token_id

        self.optimizer = AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            betas=(config.beta1, config.beta2), 
            eps=config.eps, 
            weight_decay=config.weight_decay
        )

        def lr_lambda(current_step):
            step = current_step + 1
            warmup_steps = config.warmup_steps
            if step < warmup_steps:
                return step / warmup_steps
            return (warmup_steps / step) ** 0.5

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_token_id, 
            label_smoothing=self.config.label_smoothing
        )
        self.scaler = GradScaler()
        self.evaluator = EvaluationMetrics(tokenizer)
        self.global_step = 0
        
        # Early stopping parameters
        self.best_perplexity = float('inf')
        self.patience = getattr(config, 'patience', 5)
        self.patience_counter = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.history = {
            'train_loss': [],
            'perplexity': [],
            'bleu_score': []
        }

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        accumulation_steps = self.config.accumulation_steps
        self.optimizer.zero_grad()

        loss_values = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training Epoch")):
            try:
                src, tgt = batch['src'].to(self.device), batch['tgt'].to(self.device)
                tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]

                with autocast('cuda', enabled=True, dtype=torch.float16):
                    output = self.model(src, tgt_input)
                    
                    # Create mask for non-padding tokens
                    mask = (tgt_output != self.pad_token_id)
                    
                    # Calculate loss only on non-padding tokens
                    loss = self.criterion(
                        output.reshape(-1, output.size(-1)), 
                        tgt_output.reshape(-1)
                    )

                    # 1. Scale the loss and compute gradients
                    self.scaler.scale(loss).backward()
                    
                    # Track loss for analysis
                    loss_values.append(loss.item() * accumulation_steps)
                    

                if (batch_idx + 1) % accumulation_steps == 0:
                # 2. Unscale gradients before clipping and stepping
                    self.scaler.unscale_(self.optimizer)
                    
                    if batch_idx == 0 or (batch_idx % 200 == 0 and batch_idx > 0):
                        self._debug_batch(batch_idx, loss, output, tgt_output, accumulation_steps)

                    # 4. Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # 5. Step the optimizer and scheduler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                total_loss += loss.item() * accumulation_steps
                
                # Periodic memory cleanup
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.warning(f"CUDA OOM at batch {batch_idx}, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        # Handle remaining gradients
        if len(train_loader) % accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        # Loss progression analysis
        if len(loss_values) > 10:
            self._analyze_loss_progression(loss_values)
            
        return total_loss / len(train_loader)  # Average loss per batch

    def _debug_batch(self, batch_idx, loss, output, tgt_output, accumulation_steps):
        """Enhanced debugging with better memory management"""
        print(f"\n--- BATCH {batch_idx} DEBUG ---")
        print(f"Loss: {loss.item() * accumulation_steps:.4f}")
        
        with torch.no_grad():
            # Sample only first sequence to save memory
            pred_tokens = output[0].argmax(dim=-1)[:10]
            target_tokens = tgt_output[0][:10]
            
            print(f"Predicted tokens: {pred_tokens.cpu().tolist()}")
            print(f"Target tokens:    {target_tokens.cpu().tolist()}")
            
            # Text comparison
            pred_text = self.tokenizer.decode(pred_tokens.cpu().tolist(), skip_special_tokens=True)
            target_text = self.tokenizer.decode(target_tokens.cpu().tolist(), skip_special_tokens=True)
            print(f"Predicted: '{pred_text}'")
            print(f"Target:    '{target_text}'")
            
            # Token diversity
            unique_pred_tokens = len(set(pred_tokens.cpu().tolist()))
            print(f"Unique predicted tokens: {unique_pred_tokens}/10")
            
            # Top predictions analysis
            logits_sample = output[0, 0, :]
            top_5_probs, top_5_indices = torch.topk(torch.softmax(logits_sample, dim=-1), 5)
            top_5_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_5_indices]
            print(f"Top 5 predictions: {list(zip(top_5_tokens, top_5_probs.cpu().tolist()))}")
            
            # Gradient norms
            total_norm = sum(p.grad.data.norm(2).item() ** 2 
                           for p in self.model.parameters() if p.grad is not None)
            total_norm = total_norm ** 0.5
            print(f"Total gradient norm: {total_norm:.6f}")
            
        print("--- END BATCH DEBUG ---\n")

    def _analyze_loss_progression(self, loss_values):
        """Analyze loss progression throughout the epoch"""
        first_10_avg = sum(loss_values[:10]) / 10
        last_10_avg = sum(loss_values[-10:]) / 10
        improvement = ((first_10_avg - last_10_avg) / first_10_avg * 100) if first_10_avg > 0 else 0
        
        print(f"\nTRAINING EPOCH SUMMARY:")
        print(f"  First 10 batches avg loss: {first_10_avg:.4f}")
        print(f"  Last 10 batches avg loss: {last_10_avg:.4f}")
        print(f"  Loss improvement: {improvement:.2f}%")
        
        # Check for training instability
        if improvement < -10:  # Loss increased significantly
            self.logger.warning("Training may be unstable - loss increased significantly during epoch")

    def validate(self, val_loader):
        self.model.eval()
        print("\n--- Running Validation ---")
        
        # Random sampling for BLEU evaluation
        num_bleu_batches = min(50, len(val_loader))
        val_indices = random.sample(range(len(val_loader)), num_bleu_batches)
        val_subset_for_bleu = [batch for i, batch in enumerate(val_loader) if i in val_indices]

        try:
            perplexity = self.evaluator.calculate_perplexity(self.model, val_loader, self.device)
            predictions, references = self.evaluator.generate_translations(
                self.model, val_subset_for_bleu, self.device
            )
            
            # Enhanced validation analysis
            self._analyze_predictions(predictions, references)
            bleu_score = self.evaluator.calculate_bleu(predictions, references)
            
            return perplexity, bleu_score
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return float('inf'), 0.0

    def _analyze_predictions(self, predictions, references):
        """Analyze prediction quality and patterns"""
        print(f"\nVALIDATION ANALYSIS:")
        print(f"Total predictions: {len(predictions)}")
        
        # Prediction pattern analysis
        empty_preds = sum(1 for p in predictions if len(p.strip()) == 0)
        period_preds = sum(1 for p in predictions if p.strip() == ".")
        single_word_preds = sum(1 for p in predictions if len(p.strip().split()) == 1)
        multi_word_preds = sum(1 for p in predictions if len(p.strip().split()) > 1)
        
        total_preds = len(predictions)
        print(f"Prediction Patterns:")
        print(f"  Empty: {empty_preds}/{total_preds} ({empty_preds/total_preds*100:.1f}%)")
        print(f"  Period-only: {period_preds}/{total_preds} ({period_preds/total_preds*100:.1f}%)")
        print(f"  Single word: {single_word_preds}/{total_preds} ({single_word_preds/total_preds*100:.1f}%)")
        print(f"  Multi-word: {multi_word_preds}/{total_preds} ({multi_word_preds/total_preds*100:.1f}%)")
        
        # Vocabulary diversity
        all_pred_words = [word for pred in predictions for word in pred.split()]
        if all_pred_words:
            unique_words = len(set(all_pred_words))
            diversity_ratio = unique_words / len(all_pred_words)
            print(f"Vocabulary: {unique_words} unique / {len(all_pred_words)} total (ratio: {diversity_ratio:.3f})")
        
        # Sample outputs
        print(f"\nSample Predictions:")
        for i in range(min(3, len(predictions))):
            print(f"  {i+1}. Pred: '{predictions[i][:50]}{'...' if len(predictions[i]) > 50 else ''}'")
            print(f"     Ref:  '{references[i][:50]}{'...' if len(references[i]) > 50 else ''}'")

    def train(self, train_loader, val_loader):
        print("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{self.config.num_epochs}")
            print(f"{'='*60}")
            
            train_loss = self.train_epoch(train_loader)
            
            with torch.no_grad():
                perplexity, bleu_score = self.validate(val_loader)
                self.history['train_loss'].append(train_loss)
                self.history['perplexity'].append(perplexity)
                self.history['bleu_score'].append(bleu_score)
                print(f"DEBUG: Perplexity from validate: {perplexity}") 
                print(f"DEBUG: BLEU Score from validate: {bleu_score}") 

                
            current_lr = self.optimizer.param_groups[0]['lr']

            
            # Comprehensive epoch summary
            self._print_epoch_summary(epoch, train_loss, perplexity, bleu_score, current_lr)
            
            # Model saving and early stopping
            if self._should_save_model(perplexity):
                self._save_best_model(perplexity)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            # Early stopping check
            if self._should_early_stop():
                print(f"Early stopping triggered after {self.patience} epochs without improvement")
                break
            
            # Learning indicators
            if epoch > 0 and bleu_score > 0.1:
                print(f"✓ Model showing learning progress (BLEU > 0.1)")

    def _print_epoch_summary(self, epoch, train_loss, perplexity, bleu_score, current_lr):
        """Print comprehensive epoch summary"""
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1} SUMMARY:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Validation Perplexity: {perplexity:.4f}")
        print(f"  BLEU Score: {bleu_score:.4f}")
        print(f"  Learning Rate: {current_lr:.7f}")
        print(f"  Global Step: {self.global_step}")
        print(f"  Patience Counter: {self.patience_counter}/{self.patience}")
        
        if epoch > 0:
            perplexity_change = perplexity - self.best_perplexity
            print(f"  Perplexity change: {perplexity_change:+.4f}")
        
        print(f"{'='*60}")

    def _should_save_model(self, perplexity):
        """Check if current model should be saved"""
        return perplexity < self.best_perplexity

    def _save_best_model(self, perplexity):
        """Save the best model checkpoint"""
        self.best_perplexity = perplexity
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'perplexity': perplexity,
            'global_step': self.global_step,
            'config': self.config
        }
        torch.save(checkpoint, 'best_model.pth')
        print(f" New best model saved! (Perplexity: {self.best_perplexity:.4f})")

    def _should_early_stop(self):
        """Check if training should be stopped early"""
        return self.patience_counter >= self.patience
    def plot_history(self, save_path="training_plots.png"):
        """Plots the training history and saves it to a file."""
        num_epochs = len(self.history['train_loss'])
        if num_epochs == 0:
            print("No history to plot.")
            return
            
        epochs = range(1, num_epochs + 1)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        # Plot Training Loss
        ax1.plot(epochs, self.history['train_loss'], 'bo-', label='Training Loss')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot Perplexity
        ax2.plot(epochs, self.history['perplexity'], 'go-', label='Validation Perplexity')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Validation Perplexity')
        ax2.legend()
        ax2.grid(True)
        
        # Plot BLEU Score
        ax3.plot(epochs, self.history['bleu_score'], 'ro-', label='Validation BLEU Score')
        ax3.set_ylabel('BLEU Score')
        ax3.set_xlabel('Epoch')
        ax3.set_title('Validation BLEU Score')
        ax3.legend()
        ax3.grid(True)
        
        fig.tight_layout()
        fig.savefig(save_path)
        print(f"Training plots saved to {save_path}")
        plt.show()