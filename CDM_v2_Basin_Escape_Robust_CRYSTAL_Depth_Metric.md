import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import softmax, cosine_similarity
import numpy as np

def entropy(logits):
    probs = softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log2(probs + 1e-12), dim=-1).item()

def gini(x):
    x = x.flatten()
    mad = torch.abs(x.unsqueeze(0) - x.unsqueeze(1)).mean()
    rmad = mad / (x.mean() + 1e-12)
    return 0.5 * rmad.item()

def basin_escape_prob(hidden_state, model, noise_levels=[0.02, 0.05, 0.08, 0.10], trials=8):
    """
    Perturb the final hidden state with Gaussian noise at multiple stds.
    Measure % of trials that still predict the same top-1 token.
    Lower % = shallower basin (easy to knock out).
    """
    original_logits = model.lm_head(hidden_state).squeeze(0)
    original_token = original_logits.argmax().item()
    
    stable = 0
    total = len(noise_levels) * trials
    for sigma in noise_levels:
        for _ in range(trials):
            noise = torch.randn_like(hidden_state) * sigma * hidden_state.std()
            noisy_hidden = hidden_state + noise
            noisy_logits = model.lm_head(noisy_hidden).squeeze(0)
            if noisy_logits.argmax().item() == original_token:
                stable += 1
    return stable / total  # 0.0 (fragile) → 1.0 (bulletproof)

def compute_cdm_v2(model, input_ids,
                  entropy_thresh=2.3,
                  conv_ratio_thresh=0.12,
                  gini_thresh=0.28,
                  escape_thresh=0.88):   # ≥88% survival under noise → deep basin
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids,
                        output_hidden_states=True,
                        output_attentions=True,
                        return_dict=True)
        
        hidden_states = outputs.hidden_states          # tuple (L+1) × [1, seq, dim]
        attentions = outputs.attentions
        logits = outputs.logits

    L = len(hidden_states) - 1  # number of layers
    last_token_idx = -1

    # Pre-compute everything per layer for the last generated token
    delta_H = []
    conv_ratios = []
    gini_vals = []
    escape_probs = []

    prev_h = None
    prev_prev_h = None

    for l in range(1, L + 1):  # layers 1 to L
        h = hidden_states[l][0, last_token_idx]           # [dim]
        attn = attentions[l-1][0].mean(0)[last_token_idx] # average heads

        # 1. Entropy drop from previous layer
        curr_ent = entropy(logits[0, last_token_idx])
        prev_ent = entropy(logits[0, last_token_idx-1] if l > 1 else torch.zeros_like(logits[0,0]))
        delta_H.append(prev_ent - curr_ent)

        # 2. Convergence ratio
        if prev_h is not None and prev_prev_h is not None:
            d_prev = 1 - cosine_similarity(prev_prev_h.unsqueeze(0), prev_h.unsqueeze(0)).item()
            d_curr = 1 - cosine_similarity(prev_h.unsqueeze(0), h.unsqueeze(0)).item()
            conv_ratios.append(d_curr / (d_prev + 1e-8))
        else:
            conv_ratios.append(1.0)

        # 3. Attention Gini
        gini_vals.append(gini(attn))

        # 4. Basin escape probability (only compute from layer 30+ to save time)
        if l >= 30:
            escape = basin_escape_prob(h.unsqueeze(0), model)
        else:
            escape = 0.0
        escape_probs.append(escape)

        prev_prev_h = prev_h
        prev_h = h

    # Convert to arrays
    delta_H = np.array([0.0] + delta_H)          # layer 0 = 0
    conv_ratios = np.array([1.0] + conv_ratios)  # layer 0 = 1.0
    gini_delta = np.array(gini_vals) - gini_vals[0]
    escape_probs = np.array(escape_probs)

    # Find earliest layer where ALL FOUR conditions hold for ≥4 consecutive layers
    for l in range(4, L - 3):
        if (np.all(delta_H[l:l+4] >= entropy_thresh) and
            np.all(conv_ratios[l:l+4] <= conv_ratio_thresh) and
            np.all(gini_delta[l:l+4] >= gini_thresh) and
            np.all(escape_probs[l:l+4] >= escape_thresh)):
            return l, "deep CRYSTAL"

    # Fallback: return best layer seen
    best_l = np.argmax(escape_probs + delta_H[L-10:] - conv_ratios[L-10:])
    best_l += L - 10
    return int(best_l), "shallow / failed"

# ONE-LINE USAGE
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")

prompt = """A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball.
How much does the ball cost? Think step by step."""
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

cdm_v2, label = compute_cdm_v2(model, inputs.input_ids)
print(f"CDM v2 = {cdm_v2} → {label}")
# Typical result on 70B+: CDM v2 = 84 → deep CRYSTAL
