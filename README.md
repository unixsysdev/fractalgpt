# Nano-Fractal 4B

> Emergent Fractal Intelligence: A hybrid Mamba-Transformer with dynamic Matryoshka scaling.

**Fork of [karpathy/nanochat](https://github.com/karpathy/nanochat)** — Extended with:

- 🧬 **Hybrid Architecture**: 32 Attention + 32 Mamba layers (64 total, ~4B params)
- 📐 **Matryoshka Dimensions**: Ghost (128) → God (4096) dynamic scaling
- 🧠 **Neural Confidence Probes**: Per-layer topological signals for capacity allocation
- ⚡ **Energy Penalty Loss**: Forces compute minimization ("lazy but correct")

## Architecture

```
nanochat-d32 (1.9B) → Surgery → Nano-Fractal 4B
├── 32 Attention layers (pretrained, expanded 2048→4096)
├── 32 Mamba layers (new, interleaved)
├── MatryoshkaMLP: [128, 512, 1024, 2048, 4096]
├── MatryoshkaKV: [32, 64, 128, 256]
└── ConfidenceProbe per layer (variance, agreement, spread)
```

## New Files

| File | Purpose |
|------|---------|
| `nanochat/mamba_block.py` | Mamba layer with SSM fallback |
| `nanochat/confidence_probe.py` | Neural probes using topological signals |
| `nanochat/matryoshka.py` | Dimension slicing + energy penalty |
| `nanochat/hybrid_gpt.py` | HybridGPT (interleaved Mamba+Attention) |
| `scripts/surgery.py` | Convert nanochat-d32 → hybrid |
| `scripts/fractal_train.py` | Training with Matryoshka loss |

## Quick Start

```bash
# 1. Download base model
huggingface-cli download karpathy/nanochat-d32 --local-dir ~/.cache/nanochat/chatsft_checkpoints/d32

# 2. Run surgery (creates 4B hybrid)
python -m scripts.surgery

# 3. Train on 8×H100
torchrun --nproc_per_node=8 -m scripts.fractal_train \
    --checkpoint ~/.cache/nanochat/hybrid_checkpoints/d32/model_surgery.pt \
    --matryoshka --sample-dim \
    --num-iterations=5000
```

## Ghost → God Spectrum

| Mode | MLP Dim | KV Dim | Compute |
|------|---------|--------|---------|
| Ghost | 128 | 32 | ~0.1% |
| Whisper | 512 | 64 | ~1.5% |
| Normal | 2048 | 128 | ~25% |
| Think | 4096 | 256 | ~100% |

The model learns *when* to scale up based on task difficulty.

## Training Cost

~$200 total on 8×H100 (~13 hours):
- Phase 1: Expand + Initialize (~2 hrs)
- Phase 2: Matryoshka Training (~6 hrs)  
- Phase 3: Emergent Think Training (~3 hrs)

---

*Based on [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy*
