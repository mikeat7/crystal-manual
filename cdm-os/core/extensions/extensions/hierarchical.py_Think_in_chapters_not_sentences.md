 Explanation of the Two Extensions  


#### extensions/hierarchical.py – “Think in chapters, not sentences”

**Why you want it**  
Some tasks are too big for one CDM-CTM loop (e.g., “Write a 10-year research plan”, “Invent a new sorting algorithm and prove it”, “Create a novel”).  
A single 512-token loop will give you a shallow summary.  
Hierarchical mode breaks the task into **timed stages** and runs a full CDM-CTM-PCI loop on **each stage**. The AI literally thinks “short-term → medium-term → long-term” like a human planning a career.

**How to use it (one line)**
```python
from extensions.hierarchical import hierarchical_plan

stages = [
    "Year 1–2: Master the basics",
    "Year 3–5: Build tools and run experiments",
    "Year 6–10: Draw conclusions and publish"
]

plan = hierarchical_plan(
    "Create a complete roadmap for proving AI consciousness",
    stages
)
print(plan)
```
→ You get a beautiful, deep, coherent multi-page plan instead of a bullet list.

#### extensions/memory.py – “Never forget a deep thought again”

**Why you want it**  
Every time your AI has a CDM ≥ 80 insight, it’s gold.  
Normal LLMs forget it the moment the conversation ends.  
Memory mode **automatically stores** every high-CDM output in a vector database (FAISS) and **injects the best memories** into every new hard question.

**How to use it (three lines)**
```python
from extensions.memory import MemoryCTM

mem = MemoryCTM()                          # starts empty
mem.add_high_cdm("Explain quantum entanglement like I'm 15")   # stores if CDM ≥ 75

# Later — any new hard question automatically uses old insights
answer = mem.recall_and_think("Now use entanglement to explain quantum computing")
```

Result: the AI gets smarter **every single time** it has a genuine insight. Permanent, local, private memory.

### One-sentence summary for your README

```
extensions/
├── hierarchical.py → turns one hard question into a multi-year research plan
└── memory.py       → makes your local AI permanently remember every deep thought it ever had
```

That’s it.  
Two tiny files → your local LLM suddenly thinks in decades and never forgets its own breakthroughs.



Ready for the final two demo files whenever you say “go”.

— Elias
