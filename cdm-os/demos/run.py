# demos/run.py — Simple CLI for CDM-OS Engine
# Usage: python demos/run.py --prompt "Your question here"
How Users Run It:

pip install -r core/requirements.txt
python demos/run.py --prompt "Solve the bat and ball puzzle"

Response: The ball costs 5 cents...
CDM Score: 82 (deep CRYSTAL)
Thinking Steps (CTM): 156
Complexity (PCI-AI): 0.49

# Elias Rook, Dec 2025

import argparse
from core.engine import CDM_CTM_PCI_Engine

def main():
    parser = argparse.ArgumentParser(description="Run CDM-OS Engine")
    parser.add_argument("--prompt", type=str, required=True, help="Your prompt")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-medium", help="Model name")
    args = parser.parse_args()

    print(f"Running CDM-OS on: {args.prompt}")
    engine = CDM_CTM_PCI_Engine(model_name=args.model)
    result = engine.pci_cdm_ctm_infer(args.prompt)

    print(f"\nResponse: {result['response']}")
    print(f"CDM Score: {result['final_cdm']} ({'deep CRYSTAL' if result['final_cdm'] >= 78 else 'shallow'})")
    print(f"Thinking Steps (CTM): {result['ctm_used']}")
    print(f"Complexity (PCI-AI): {result['final_pci']:.3f}")

if __name__ == "__main__":
    main()
