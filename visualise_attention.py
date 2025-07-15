import torch
import matplotlib.pyplot as plt
import seaborn as sns
from src.evaluation_metrics import EvaluationMetrics

def visualize_attention(model, tokenizer, sentence, device):
    """
    Visualizes the encoder-decoder attention for a given sentence.
    """
    model.eval()
    
    src_tokens = tokenizer.encode(sentence)
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(device)

    attention_weights = []
    
    def hook_fn(module, input, output):
        attention_weights.append(output[1])

    hook = model.decoder.layers[-1].cross_attention.register_forward_hook(hook_fn)

    evaluator = EvaluationMetrics(tokenizer)
    generated_tensor = evaluator.greedy_decode(model, src_tensor, device)

    hook.remove()
    
    if not attention_weights:
        print("Could not capture attention weights. Check the layer path for the hook.")
        return

    attn = attention_weights[0][0, 0, :, :].cpu().numpy()

    src_labels = tokenizer.decode(src_tokens, skip_special_tokens=False).split()
    tgt_labels = tokenizer.decode(generated_tensor[0, 1:].cpu().tolist(), skip_special_tokens=False).split()

    # Plotting
    plt.figure(figsize=(12, 12))
    sns.heatmap(attn, xticklabels=src_labels, yticklabels=tgt_labels, cmap='viridis')
    plt.xlabel("Source Tokens (German)")
    plt.ylabel("Generated Tokens (English)")
    plt.title("Encoder-Decoder Attention")
    plt.show()

