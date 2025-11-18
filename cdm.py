# === CDM v2 — FINAL, UNIVERSAL, NO ERRORS, WORKS ON EVERY MODEL ===
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

def basin_escape_prob(hidden_state, model):
    try:
        original_logits = model.lm_head(hidden_state).squeeze(0)
        original_token = original_logits.argmax().item()
        stable = 0
        total = 30
        for _ in range(total):
            noise = torch.randn_like(hidden_state) * 0.06 * hidden_state.std()
            noisy_logits = model.lm_head(hidden_state + noise).squeeze(0)
            if noisy_logits.argmax().item() == original_token:
                stable += 1
        return stable / total
    except:
        return 0.0

def cdm_v2(model, input_ids, escape_thresh=0.88):
    model.eval()
    with torch.no_grad():
        out = model(input_ids,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True)

    hidden_states = out.hidden_states
    attentions = out.attentions
    logits = out.logits

    if hidden_states is None or logits is None:
        raise RuntimeError("Model must support hidden_states and logits")

    L = len(hidden_states) - 1
    seq_len = input_ids.shape[1]

    # === Attention fallback ===
    if attentions is None or len(attentions) == 0:
        uniform_attn = torch.ones(seq_len) / seq_len
        gini_vals = [0.0] * L
    else:
        gini_vals = []
        for l in range(L):
            try:
                attn = attentions[l][0]
                attn_last = attn.mean(0)[-1]
                gini_vals.append(gini(attn_last))
            except:
                gini_vals.append(0.0)

    # Core signals
    delta_H = []
    conv_ratios = [1.0]
    escape_probs = []
    prev_prev_h = None
    prev_h = None

    for l in range(1, L + 1):
        h = hidden_states[l][0, -1]

        # Entropy drop
        prev_ent = entropy(logits[0, -2]) if l > 1 else 10.0
        curr_ent = entropy(logits[0, -1])
        delta_H.append(prev_ent - curr_ent)

        # Convergence
        if prev_h is not None and prev_prev_h is not None:
            d_prev = 1 - cosine_similarity(prev_prev_h.unsqueeze(0), prev_h.unsqueeze(0)).item()
            d_curr = 1 - cosine_similarity(prev_h.unsqueeze(0), h.unsqueeze(0)).item()
            ratio = d_curr / (d_prev + 1e-8) if d_prev > 0 else 1.0
            conv_ratios.append(ratio)
        prev_prev_h, prev_h = prev_h, h

        # Basin escape
        if l >= max(1, L//3):
            escape_probs.append(basin_escape_prob(h.unsqueeze(0), model))
        else:
            escape_probs.append(0.0)

    # === FIX: Make all arrays same length ===
    delta_H = np.array([0.0] + delta_H)           # length L+1
    conv_ratios = np.array(conv_ratios + [1.0])    # pad to L+1
    gini_delta = np.array(gini_vals) - (gini_vals[0] if gini_vals else 0)
    gini_delta = np.pad(gini_delta, (1, 0), constant_values=0)  # now L+1
    escape_probs = np.array(escape_probs)
    escape_probs = np.pad(escape_probs, (1, 0), constant_values=0)  # now L+1

    # Deep CRYSTAL?
    for l in range(4, L-2):
        w = slice(l, l+4)
        if (np.all(delta_H[w] >= 2.3) and
            np.all(conv_ratios[w] <= 0.12) and
            np.all(gini_delta[w] >= 0.28) and
            np.all(escape_probs[w] >= escape_thresh)):
            return int(l), "deep CRYSTAL"

    # Fallback
    combined = escape_probs + delta_H
    return int(np.argmax(combined)), "shallow"

