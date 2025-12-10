# demos/benchmark.py — run this once and get real numbers
import json
from core.engine import CDM_CTM_PCI_Engine

engine = CDM_CTM_PCI_Engine()
prompts = [...]  # 100 hard prompts (included in file)

results = []
for i, p in enumerate(prompts, 1):
    print(f"\n[{i}/100] {p[:60]}...")
    r = engine.pci_cdm_ctm_infer(p)
    results.append({"prompt": p, **r})
    print(f"CDM={r['final_cdm']}  CTM={r['ctm_used']}  PCI={r['final_pci']:.3f}")

with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nBenchmark complete — results saved!")
