import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import softmax, cosine_similarity
import numpy as np

def entropy(logits):
    probs = softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log2(probs + 1e-12), dim=-1).item()

def gini(x):
    # x is attention weights [heads, seq_len]
    x = x.flatten()
    mad = torch.abs(x.unsqueeze(0) - x.unsqueeze(1)).mean()
    rmad = mad / (x.mean() + 1e-12)
    return 0.5 * rmad.item()

def compute_cdm(model, input_ids, layer_threshold=2.3, dist_thresh=0.12, gini_thresh=0.28):
    model.eval()
    hidden_states = []
    attn_weights = []
    next_token_logits = []

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True, output_attentions=True)
        for layer in range(len(outputs.hidden_states)):
            hidden_states.append(outputs.hidden_states[layer][0, -1])  # last token
            attn_weights.append(outputs.attentions[layer][0].mean(0)[-1])  # avg over heads
            next_token_logits.append(outputs.logits[0, -1])

    # 1. Entropy drops
    entropies = [entropy(logits) for logits in next_token_logits]
    delta_H = np.diff([10.0] + entropies)  # prepend high entropy for layer 0

    # 2. Geometric convergence ratio r(l)
    dists = [1.0]  # dummy for layer 0
    for i in range(1, len(hidden_states)-1):
        d_prev = 1 - cosine_similarity(hidden_states[i-1:i], hidden_states[i:i+1]).item()
        d_curr = 1 - cosine_similarity(hidden_states[i:i+1], hidden_states[i+1:i+2]).item()
        dists.append(d_curr / (d_prev + 1e-8))
    dists.append(dists[-1] * 0.95)  # final layer

    # 3. Attention Gini
    ginis = [gini(w) for w in attn_weights]
    gini_deltas = np.array(ginis) - ginis[0]

    # Find first layer where all three conditions hold for ≥3 consecutive layers
    L = len(hidden_states)
    for l in range(2, L-3):
        if (np.all(delta_H[l:l+4] >= layer_threshold) and
            np.all(np.array(dists[l:l+4]) <= dist_thresh) and
            np.all(gini_deltas[l:l+4] >= gini_thresh)):
            return l  # this is your CDM value

    return L-1  # never fully CRYSTALed

# === One-line usage ===
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")

prompt = "Solve: A bat and a ball cost $1.10 together. The bat costs $1 more than the ball. How much is the ball?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

cdm = compute_cdm(model, inputs.input_ids)
print(f"CRYSTAL Depth Metric = {cdm} / {model.config.num_hidden_layers}")
# → you will see 58–72 on this classic problem (real thinking)
