# Nano-Fractal

> Emergent Fractal Intelligence: Hybrid Mamba-Transformer with dynamic Matryoshka scaling.

**Fork of [karpathy/nanochat](https://github.com/karpathy/nanochat)**

## What's New

- 🧬 **Hybrid Architecture**: 32 Attention + 32 Mamba layers (64 total)
- 📐 **Matryoshka Dimensions**: Ghost (128) → God (4096) dynamic scaling
- 🧠 **Neural Confidence Probes**: Per-layer capacity allocation
- ⚡ **Energy Penalty Loss**: "Lazy but correct" compute minimization
- 🪆 **Progressive Training**: 4B → 9B → 20B staged expansion

## Architecture

```
nanochat-d32 (1.9B, 32 layers, dim=2048)
    ↓ Surgery (add 32 Mamba layers)
Stage 1: 4B  (dim=2048)  ← No expansion, just add Mamba
    ↓ Progressive expand
Stage 2: 9B  (dim=2560)  
    ↓ Progressive expand
Stage 3: 20B (dim=4096)
```

## Training Cost Estimates

| Stage | Model Size | New Params | Tokens | Hours (8×H100) | Cost |
|-------|------------|------------|--------|----------------|------|
| 1 | 4B | ~2B | 40B | 25h | $600 |
| 2 | 9B | ~5B | 100B | 60h | $1,500 |
| 3 | 20B | ~11B | 220B | 130h | $3,200 |
| **Total** | **20B** | | | **215h** | **~$5,300** |

*Or stop at any stage - 4B alone costs ~$600*

## Quick Start

```bash
# 1. Download nanochat-d32 base
huggingface-cli download karpathy/nanochat-d32 \
    --local-dir ~/.cache/nanochat/chatsft_checkpoints/d32

# 2. Stage 1: Create 4B hybrid (no dim expansion)
python -m scripts.surgery --new-dim=2048

# 3. Train Stage 1
torchrun --nproc_per_node=8 -m scripts.fractal_train \
    --checkpoint ~/.cache/nanochat/hybrid_checkpoints/d32_2048/model.pt \
    --expanded-dim=2048 --matryoshka --sample-dim

# 4. Stage 2: Expand trained 4B → 9B
python -m scripts.surgery --expand-from=2048 --new-dim=2560

# 5. Train Stage 2, then expand to 20B...
```

## New Files

| File | Purpose |
|------|---------|
| `nanochat/hybrid_gpt.py` | HybridGPT (Mamba+Attention) |
| `nanochat/mamba_block.py` | Mamba with SSM fallback |
| `nanochat/matryoshka.py` | Dimension slicing + energy loss |
| `nanochat/confidence_probe.py` | Neural probes |
| `scripts/surgery.py` | Create/expand hybrid checkpoints |
| `scripts/fractal_train.py` | Matryoshka training |

## Matryoshka Levels

| Mode | Dim | Compute | Use Case |
|------|-----|---------|----------|
| Ghost | 128 | ~0.4% | Trivial tasks |
| Whisper | 512 | ~6% | Simple Q&A |
| Normal | 1024 | ~25% | General use |
| Think | 2048+ | 100% | Complex reasoning |

---

*Based on [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy*
