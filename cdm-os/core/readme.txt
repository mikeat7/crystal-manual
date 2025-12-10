# core/requirements.txt — Minimal Deps for CDM-CTM-PCI Engine
torch>=2.4.0
transformers>=4.45.0
accelerate>=0.33.0
numpy>=1.26.0
bitsandbytes>=0.43.3  # For 4/8-bit loading (optional, saves VRAM)
zlib  # Built-in for LZ compression in PCI-AI

extensions/
├── hierarchical.py → turns one hard question into a multi-year research plan
└── memory.py       → makes your local AI permanently remember every deep thought it ever had
