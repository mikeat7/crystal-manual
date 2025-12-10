# extensions/memory.py — Infinite-horizon external memory
# Remembers every high-CDM output forever using FAISS

import faiss
import numpy as np
from core.engine import CDM_CTM_PCI_Engine

class MemoryCTM:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct", dim: int = 4096):
        self.engine = CDM_CTM_PCI_Engine(model_name)
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.memory = []  # (text, cdm_score, embedding)

    def add_high_cdm(self, prompt: str, min_cdm: int = 75):
        result = self.engine.pci_cdm_ctm_infer(prompt)
        if result["final_cdm"] >= min_cdm:
            # Extract final hidden state for embedding
            with torch.no_grad():
                inputs = self.engine.tokenizer(prompt, return_tensors="pt").input_ids.to(self.engine.model.device)
                hidden = self.engine.model(inputs, output_hidden_states=True).hidden_states[-1].mean(1).cpu().numpy()
            self.index.add(hidden)
            self.memory.append((result["response"], result["final_cdm"], hidden))
            print(f"Stored high-CDM memory (CDM={result['final_cdm']})")
        return result["response"]

    def recall_and_think(self, prompt: str, k: int = 3):
        with torch.no_grad():
            inputs = self.engine.tokenizer(prompt, return_tensors="pt").input_ids.to(self.engine.model.device)
            query = self.engine.model(inputs, output_hidden_states=True).hidden_states[-1].mean(1).cpu().numpy()
        D, I = self.index.search(query, k)
        context = "\n".join([HIGH-CDM MEMORY]\n" + "\n---\n".join(self.memory[i][0] for i in I[0] if i < len(self.memory)) + "\n[/MEMORY]\n"
        augmented_prompt = context + prompt
        return self.engine.pci_cdm_ctm_infer(augmented_prompt)["response"]

# Example
if __name__ == "__main__":
    mem = MemoryCTM()
    mem.add_high_cdm("Explain quantum entanglement like I'm 15")
    print("\n--- Later query using memory ---")
    print(mem.recall_and_think("Now explain quantum computing with entanglement"))
