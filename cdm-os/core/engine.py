# engine.py — Merged CDM-CTM-PCI Fusion Engine
# Elias Rook, Dec 2025 — Core for self-improving local LLMs

import torch
import numpy as np
import zlib  # LZ for PCI
from transformers import AutoModelForCausalLM, AutoTokenizer

class CDM_CTM_PCI_Engine:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
                 target_cdm: int = 78, target_pci: float = 0.35,
                 max_ctm: int = 1024, velocity_thresh: int = 5):
        self.model_name = model_name
        self.target_cdm = target_cdm
        self.target_pci = target_pci
        self.max_ctm = max_ctm
        self.velocity_thresh = velocity_thresh
        self.granularity = 1  # Adjustment level
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
            output_hidden_states=True, output_attentions=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def entropy(self, logits):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return -torch.sum(probs * torch.log2(probs + 1e-12), dim=-1).item()

    def gini(self, x):
        x = x.flatten()
        mad = torch.abs(x.unsqueeze(0) - x.unsqueeze(1)).mean()
        rmad = mad / (x.mean() + 1e-12)
        return 0.5 * rmad.item()

    def basin_escape_prob(self, hidden_state):
        try:
            original_logits = self.model.lm_head(hidden_state).squeeze(0)
            original_token = original_logits.argmax().item()
            stable = 0
            total = 30
            for _ in range(total):
                noise = torch.randn_like(hidden_state) * 0.06 * hidden_state.std()
                noisy_logits = self.model.lm_head(hidden_state + noise).squeeze(0)
                if noisy_logits.argmax().item() == original_token:
                    stable += 1
            return stable / total
        except:
            return 0.0

    def cdm_v2(self, input_ids):
        with torch.no_grad():
            out = self.model(input_ids, output_hidden_states=True, output_attentions=True, return_dict=True)

        hidden_states = out.hidden_states
        attentions = out.attentions or []
        logits = out.logits

        L = len(hidden_states) - 1
        seq_len = input_ids.shape[1]

        if len(attentions) == 0:
            uniform_attn = torch.ones(seq_len) / seq_len
            gini_vals = [0.0] * L
        else:
            gini_vals = [self.gini(attentions[l][0].mean(0)[-1]) for l in range(L)]

        delta_H = []
        conv_ratios = [1.0]
        escape_probs = []
        prev_prev_h = None
        prev_h = None

        for l in range(1, L + 1):
            h = hidden_states[l][0, -1]

            prev_ent = self.entropy(logits[0, -2]) if l > 1 else 10.0
            curr_ent = self.entropy(logits[0, -1])
            delta_H.append(prev_ent - curr_ent)

            if prev_h is not None and prev_prev_h is not None:
                d_prev = 1 - cosine_similarity(prev_prev_h.unsqueeze(0), prev_h.unsqueeze(0)).item()
                d_curr = 1 - cosine_similarity(prev_h.unsqueeze(0), h.unsqueeze(0)).item()
                conv_ratios.append(d_curr / (d_prev + 1e-8))

            prev_prev_h, prev_h = prev_h, h
            escape_probs.append(self.basin_escape_prob(h.unsqueeze(0)) if l >= L//3 else 0.0)

        delta_H = np.array([0.0] + delta_H)
        conv_ratios = np.array(conv_ratios + [1.0])
        gini_delta = np.array(gini_vals) - gini_vals[0] if gini_vals else np.zeros(L+1)
        gini_delta = np.pad(gini_delta, (1, 0), constant_values=0)
        escape_probs = np.pad(np.array(escape_probs), (1, 0), constant_values=0)

        for l in range(4, L-3):
            w = slice(l, l+4)
            if all([np.all(delta_H[w] >= 2.3), np.all(conv_ratios[w] <= 0.12), np.all(gini_delta[w] >= 0.28), np.all(escape_probs[w] >= 0.88)]):
                return int(l), "deep CRYSTAL"

        return int(np.argmax(escape_probs + delta_H)), "shallow"

    def binarize_matrix(self, hidden_states):
        matrix = []
        for h in hidden_states[1:]:
            mean = h.mean().item()
            binary = (h > mean).cpu().numpy().flatten()
            matrix.append(binary)
        return np.vstack(matrix)

    def lz_complexity(self, matrix):
        compressed = zlib.compress(matrix.tobytes())
        return len(compressed) / matrix.nbytes

    def pci_ai(self, input_ids, sigma=0.05):
        with torch.no_grad():
            out = self.model(input_ids, output_hidden_states=True)

        mid = len(out.hidden_states) // 2
        h_mid = out.hidden_states[mid]
        noise = torch.randn_like(h_mid) * sigma * h_mid.std()
        perturbed_h = h_mid + noise

        perturbed_out = self.model(inputs_embeds=self.model.model.embed_tokens(input_ids),
                                   hidden_states=[None] * mid + [perturbed_h] + [None] * (len(out.hidden_states) - mid - 1),
                                   output_hidden_states=True)

        matrix = self.binarize_matrix(perturbed_out.hidden_states)
        return self.lz_complexity(matrix)

    def pci_cdm_ctm_infer(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        trajectory = input_ids.clone()
        cdm_history = []
        pci_history = []
        prev_cdm = 0

        for step in range(self.max_ctm):
            cdm, label = self.cdm_v2(trajectory)
            pci = self.pci_ai(trajectory)
            cdm_history.append(cdm)
            pci_history.append(pci)
            delta_cdm = cdm - prev_cdm

            if cdm >= self.target_cdm and label == "deep CRYSTAL" and pci >= self.target_pci:
                break

            if delta_cdm > self.velocity_thresh and step > self.max_ctm * 0.7:
                self.max_ctm += 256

            trajectory = torch.cat([trajectory, torch.tensor([[self.tokenizer.pad_token_id]], device=self.model.device)], dim=1)
            prev_cdm = cdm

        with torch.no_grad():
            output = self.model.generate(trajectory, max_new_tokens=512, do_sample=False)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return {
            "response": response,
            "final_cdm": cdm,
            "final_pci": pci,
            "ctm_used": step,
            "cdm_trajectory": cdm_history,
            "pci_trajectory": pci_history
        }

# Usage
if __name__ == "__main__":
    engine = CDM_CTM_PCI_Engine()
    result = engine.pci_cdm_ctm_infer("Solve: A bat and ball cost $1.10. Bat costs $1 more than ball. Ball costs?")
    print(result)
