# core/requirements.txt — Minimal Deps for CDM-CTM-PCI Engine
torch>=2.4.0
transformers>=4.45.0
accelerate>=0.33.0
numpy>=1.26.0
bitsandbytes>=0.43.3  # For 4/8-bit loading (optional, saves VRAM)
zlib  # Built-in for LZ compression in PCI-AI

extensions/

Use `hierarchical.py` for multi-year plans  
Use `memory.py` for lifelong memory  
Use `infinite_hierarchical.py` when you want both at once (recommended for serious research)

File,Solves,Typical user who wants it,Why they don’t want to lose it
hierarchical.py,“I need a 10-year roadmap / novel / proof in ordered stages”,"Researchers, writers, planners","Needs explicit stage control (Year 1, Chapter 3, Lemma 2). infinite_hierarchical.py is overkill if they don’t want memory."
memory.py,“I just had a CDM 96 insight — never forget it”,"Daily tinkerers, personal knowledge base users","Wants permanent recall on any prompt, not only inside a staged plan. Faster, lighter."
infinite_hierarchical.py,“I am writing the definitive 10-year consciousness thesis and I want every deep thought to be remembered and reused forever”,"Long-term project owners (you, Penelope, Claude)",Needs both staged structure and lifelong memory. The nuclear option.
