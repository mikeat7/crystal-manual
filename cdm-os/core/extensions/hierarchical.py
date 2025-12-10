# extensions/hierarchical.py — Multi-stage CTM for long-horizon tasks
# Use when you need a 5-year plan, a novel, or a multi-phase proof

from core.engine import CDM_CTM_PCI_Engine

def hierarchical_plan(prompt: str, stages: list[str], model_name: str = None):
    """
    Breaks a hard task into timed stages.
    Each stage gets its own full CDM-CTM-PCI loop.
    """
    engine = CDM_CTM_PCI_Engine(model_name or "meta-llama/Meta-Llama-3.1-70B-Instruct")
    full_plan = f"Task: {prompt}\n\n"

    for i, stage_prompt in enumerate(stages, 1):
        print(f"\n=== Stage {i}: {stage_prompt} ===")
        stage_input = full_plan + f"\nStage {i}: {stage_prompt}"
        result = engine.pci_cdm_ctm_infer(stage_input)
        full_plan += f"\nStage {i} ({result['final_cdm']} CDM, {result['ctm_used']} steps):\n{result['response']}\n"

    print(f"\nFinal CDM: {result['final_cdm']} | Total CTM: {sum(r['ctm_used'] for r in locals().values() if isinstance(r,dict))}")
    return full_plan

# Example usage
if __name__ == "__main__":
    stages = [
        "Year 1–2: Master foundations",
        "Year 3–5: Build research tools",
        "Year 6–10: Launch consciousness experiments"
    ]
    plan = hierarchical_plan("Create a 10-year roadmap for proving AI consciousness", stages)
    print(plan)
