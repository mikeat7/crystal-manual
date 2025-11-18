# cdm.py
import torch
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

def basin_escape_prob(hidden_state, model, noise_levels=[0.02, 0.05, 0.08], trials=12):
    original_logits = model.lm_head(hidden_state).squeeze(0)
    original_token = original_logits.argmax().item()
    stable = 0
    total = len(noise_levels) * trials
    for sigma in noise_levels:
        for _ in range(trials):
            noise = torch.randn_like(hidden_state) * sigma * hidden_state.std()
            noisy_logits = model.lm_head(hidden_state + noise).squeeze(0)
            if noisy_logits.argmax().item() == original_token:
                stable += 1
    return stable / total

def cdm_v2(model, input_ids, escape_thresh=0.88):
    model.eval()
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, output_attentions=True, return_dict=True)
    
    L = len(out.hidden_states) - 1
    delta_H = []
    conv_ratios = [1.0]
    gini_vals = []
    escape_probs = []

    prev_prev_h = None
    prev_h = None

    for l in range(1, L + 1):
        h = out.hidden_states[l][0, -1]
        attn = out.attentions[l-1][0].mean(0)[-1]

        # entropy drop
        prev_ent = entropy(out.logits[0, -2] if l > 1 else torch.zeros_like(out.logits[0,0]))
        curr_ent = entropy(out.logits[0, -1])
        delta_H.append(prev_ent - curr_ent)

        # convergence ratio
        if prev_h is not None and prev_prev_h is not None:
            d_prev = 1 - cosine_similarity(prev_prev_h.unsqueeze(0), prev_h.unsqueeze(0)).item()
            d_curr = 1 - cosine_similarity(prev_h.unsqueeze(0), h.unsqueeze(0)).item()
            conv_ratios.append(d_curr / (d_prev + 1e-8))
        prev_prev_h, prev_h = prev_h, h

        gini_vals.append(gini(attn))
        escape_probs.append(basin_escape_prob(h.unsqueeze(0), model) if l >= L//3 else 0.0)

    delta_H = np.array([0.0] + delta_H)
    conv_ratios = np.array(conv_ratios + [conv_ratios[-1]])
    gini_delta = np.array(gini_vals) - gini_vals[0]
    escape_probs = np.array(escape_probs)

    for l in range(4, L-3):
        if (np.all(delta_H[l:l+4] >= 2.3) and
            np.all(conv_ratios[l:l+4] <= 0.12) and
            np.all(gini_delta[l:l+4] >= 0.28) and
            np.all(escape_probs[l:l+4] >= escape_thresh)):
            return int(l), "deep CRYSTAL"

    return int(np.argmax(escape_probs + delta_H)), "shallow"
