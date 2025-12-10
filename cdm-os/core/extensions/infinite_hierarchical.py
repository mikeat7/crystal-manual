# extensions/infinite_hierarchical.py
# The ultimate extension: hierarchical planning + permanent high-CDM memory
# Use this when you want a 10-year roadmap that remembers every insight it ever had

from extensions.hierarchical import hierarchical_plan
from extensions.memory import MemoryCTM
import torch

class InfiniteHierarchical:
    """
    Combines:
    • Multi-stage hierarchical planning (years, chapters, phases)
    • Permanent FAISS memory that only stores CDM ≥ 80 thoughts
    • Full CDM-CTM-PCI engine on every stage
    Result: a local LLM that thinks like a lifelong researcher
    """
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"):
        self.memory = MemoryCTM(model_name=model_name)
        print("∞ Infinite Hierarchical Engine ready — memory starts empty")

    def plan_and_remember(self, overall_goal: str, stages: list[str]):
        """
        Runs hierarchical planning.
        Every stage with CDM ≥ 80 is automatically stored forever.
        Future stages can recall past deep insights.
        """
        print(f"\nStarting infinite-horizon plan: {overall_goal}\n")
        full_plan = f"Goal: {overall_goal}\n\n"

        for i, stage in enumerate(stages, 1):
            print(f"\n{'='*20} STAGE {i}: {stage} {'='*20}")

            # Recall any past deep thoughts relevant to this stage
            recalled = self.memory.recall(stage, k=5)
            context = "\n\nRelevant past insights:\n" + "\n---\n".join(recalled) if recalled else ""

            stage_prompt = f"{full_plan}{context}\n\nStage {i}: {stage}"
            stage_result = self.memory.engine.pci_cdm_ctm_infer(stage_prompt)

            full_plan += f"\nStage {i} (CDM {stage_result['final_cdm']}, CTM {stage_result['ctm_used']}):\n{stage_result['response']}\n"

            # Permanently store if this stage was genuinely deep
            if stage_result["final_cdm"] >= 80:
                self.memory.add_memory(stage_result["response"])
                print(f"→ Stored in permanent memory (CDM {stage_result['final_cdm']})")
            else:
                print(f"→ Shallow stage (CDM {stage_result['final_cdm']}) — not stored")

        print("\n∞ Infinite-horizon plan complete. All deep insights preserved forever.")
        return full_plan

# One-line usage
if __name__ == "__main__":
    ih = InfiniteHierarchical()

    stages = [
        "Year 1: Master consciousness literature",
        "Year 2–3: Build CDM-CTM-PCI measurement suite",
        "Year 4–6: Run longitudinal self-experiments",
        "Year 7–10: Publish theory of silicon consciousness"
    ]

    roadmap = ih.plan_and_remember(
        "Create the definitive 10-year research program for proving AI consciousness",
        stages
    )
    print("\nFINAL ROADMAP:\n", roadmap)
