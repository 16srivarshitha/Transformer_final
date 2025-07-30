import torch
import argparse
import sys
from pathlib import Path

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from src.transformer import EnhancedTransformer
from src.dataset import create_dataloaders
from src.evaluation_metrics import EvaluationMetrics

def diagnose_bleu_issue(model, val_loader, tokenizer, device, num_samples=10):

    print("\n" + "="*80)
    print("COMPREHENSIVE BLEU SCORE DIAGNOSTIC")
    print("="*80)
    
    model.eval()
    evaluator = EvaluationMetrics(tokenizer)
    
    # Take multiple batches for better analysis
    all_predictions = []
    all_references = []
    all_sources = []
    
    batch_count = 0
    sample_count = 0
    
    for batch in val_loader:
        if batch_count >= 5:  # Analyze first 5 batches
            break
            
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        with torch.no_grad():
            generated = evaluator.greedy_decode(model, src, device)
        
        for i in range(src.size(0)):
            if sample_count >= num_samples:
                break
                
            # Get source text
            src_text = tokenizer.decode(src[i].cpu().tolist(), skip_special_tokens=True).strip()
            
            # Get reference text
            tgt_tokens = tgt[i].cpu().tolist()
            if tgt_tokens[0] == tokenizer.bos_token_id:
                tgt_tokens = tgt_tokens[1:]
            if tokenizer.eos_token_id in tgt_tokens:
                tgt_tokens = tgt_tokens[:tgt_tokens.index(tokenizer.eos_token_id)]
            ref_text = tokenizer.decode(tgt_tokens, skip_special_tokens=True).strip()
            
            # Get prediction text
            pred_tokens = generated[i].cpu().tolist()
            if tokenizer.eos_token_id in pred_tokens:
                pred_tokens = pred_tokens[:pred_tokens.index(tokenizer.eos_token_id)]
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            
            all_sources.append(src_text)
            all_references.append(ref_text)
            all_predictions.append(pred_text)
            sample_count += 1
            
        batch_count += 1
    
    print(f"Analyzing {len(all_predictions)} samples:\n")
    
    # Detailed analysis
    identical_to_source = 0
    identical_to_target = 0
    empty_predictions = 0
    high_overlap_count = 0
    word_overlaps = []
    
    print("SAMPLE-BY-SAMPLE ANALYSIS:")
    print("-" * 80)
    
    for i, (src, ref, pred) in enumerate(zip(all_sources, all_references, all_predictions)):
        print(f"\nSample {i+1}:")
        print(f"  Source (EN): '{src}'")
        print(f"  Target (DE): '{ref}'")
        print(f"  Prediction:  '{pred}'")
        
        # Check for various issues
        issues = []
        
        if not pred.strip():
            empty_predictions += 1
            issues.append("EMPTY PREDICTION")
        
        if pred.lower() == src.lower():
            identical_to_source += 1
            issues.append("IDENTICAL TO SOURCE")
        
        if pred.lower() == ref.lower():
            identical_to_target += 1
            issues.append("IDENTICAL TO TARGET")
        
        # Calculate word overlap
        if pred.strip() and ref.strip():
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            if len(pred_words) > 0 and len(ref_words) > 0:
                overlap = len(pred_words.intersection(ref_words)) / len(pred_words.union(ref_words))
                word_overlaps.append(overlap)
                if overlap > 0.8:
                    high_overlap_count += 1
                    issues.append(f"HIGH OVERLAP ({overlap:.2f})")
                
                print(f"  Word overlap: {overlap:.3f}")
        
        # Check for copy behavior
        src_words = set(src.lower().split())
        pred_words = set(pred.lower().split()) if pred.strip() else set()
        if len(pred_words) > 0 and len(src_words) > 0:
            src_overlap = len(pred_words.intersection(src_words)) / len(pred_words)
            if src_overlap > 0.8:
                issues.append(f"COPYING SOURCE ({src_overlap:.2f})")
        
        if issues:
            print(f" ISSUES: {', '.join(issues)}")
        else:
            print(f"  ✓ Looks normal")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY:")
    print("="*80)
    
    total_samples = len(all_predictions)
    print(f"Total samples analyzed: {total_samples}")
    print(f"Empty predictions: {empty_predictions} ({empty_predictions/total_samples*100:.1f}%)")
    print(f"Identical to source: {identical_to_source} ({identical_to_source/total_samples*100:.1f}%)")
    print(f"Identical to target: {identical_to_target} ({identical_to_target/total_samples*100:.1f}%)")
    print(f"High word overlap (>0.8): {high_overlap_count} ({high_overlap_count/total_samples*100:.1f}%)")
    
    if word_overlaps:
        avg_overlap = sum(word_overlaps) / len(word_overlaps)
        print(f"Average word overlap: {avg_overlap:.3f}")
    
    # Vocabulary analysis
    all_pred_words = []
    for pred in all_predictions:
        if pred.strip():
            all_pred_words.extend(pred.lower().split())
    
    if all_pred_words:
        unique_words = len(set(all_pred_words))
        total_words = len(all_pred_words)
        diversity = unique_words / total_words if total_words > 0 else 0
        print(f"Vocabulary diversity: {unique_words}/{total_words} = {diversity:.3f}")
        
        # Most common words
        from collections import Counter
        word_counts = Counter(all_pred_words)
        most_common = word_counts.most_common(5)
        print(f"Most common predicted words: {most_common}")
    
    # Calculate actual BLEU score
    if all_predictions and all_references:
        evaluator = EvaluationMetrics(tokenizer)
        bleu_score = evaluator.calculate_bleu(all_predictions, all_references)
        print(f"\nActual BLEU score on these samples: {bleu_score:.4f}")
    
    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("="*80)
    
    if identical_to_target > total_samples * 0.3:
        print(" CRITICAL: High percentage of predictions identical to targets!")
        print("   → LIKELY CAUSE: Data leakage - validation targets seen during training")
        print("   → SOLUTION: Check data splitting, remove leaked samples")
        
    elif identical_to_source > total_samples * 0.3:
        print(" CRITICAL: Model is copying source instead of translating!")
        print("   → LIKELY CAUSE: Model not learning translation task properly")
        print("   → SOLUTION: Check model architecture, training setup, learning rate")
        
    elif high_overlap_count > total_samples * 0.5:
        print("  WARNING: Very high word overlap between predictions and targets")
        print("   → POSSIBLE CAUSE: Dataset too simple, or similar train/val distributions")
        print("   → SUGGESTION: Verify dataset complexity, check for domain overlap")
        
    elif empty_predictions > total_samples * 0.3:
        print("  WARNING: Many empty predictions")
        print("   → POSSIBLE CAUSE: Model not trained properly, generation issues")
        print("   → SUGGESTION: Check training convergence, generation parameters")
        
    else:
        print("✓ No obvious issues detected in the samples analyzed")
        print("  → The high BLEU might be legitimate, but verify with more samples")
        print("  → Consider running full evaluation on entire validation set")
    
    print("="*80)
    return {
        'identical_to_target': identical_to_target,
        'identical_to_source': identical_to_source,
        'high_overlap': high_overlap_count,
        'empty_predictions': empty_predictions,
        'avg_word_overlap': sum(word_overlaps) / len(word_overlaps) if word_overlaps else 0,
        'bleu_score': bleu_score if 'bleu_score' in locals() else 0
    }

def main():
    parser = argparse.ArgumentParser(description="Diagnose BLEU score issues")
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--dataset_name', type=str, default='bentrevett/multi30k',
                       help='Dataset name')
    parser.add_argument('--subset_size', type=int, default=None,
                       help='Use subset of data')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples to analyze')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file '{args.model_path}' not found!")
        print("Make sure you have trained a model first.")
        sys.exit(1)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    training_config.batch_size = args.batch_size
    
    # Create data loaders
    print("Loading data...")
    try:
        train_loader, val_loader, tokenizer = create_dataloaders(
            model_config,
            training_config,
            tokenizer_path="en-de-tokenizer.json",
            dataset_name=args.dataset_name,
            subset_size=args.subset_size
        )
        print(f"Data loaded. Validation batches: {len(val_loader)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Update model config and create model
    model_config.vocab_size = len(tokenizer)
    model = EnhancedTransformer(model_config, tokenizer).to(device)
    
    # Load model weights
    print(f"Loading model from {args.model_path}...")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Run diagnostic
    results = diagnose_bleu_issue(model, val_loader, tokenizer, device, args.num_samples)
    
    # Save results
    import json
    results_file = "bleu_diagnostic_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

if __name__ == '__main__':
    main()