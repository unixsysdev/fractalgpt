# Nano-Fractal

> Emergent Fractal Intelligence: Hybrid Mamba-Transformer with dynamic Matryoshka scaling.

**Fork of [karpathy/nanochat](https://github.com/karpathy/nanochat)**

## What's New

- 🧬 **Hybrid Architecture**: 32 Attention + 32 Mamba layers (64 total)
- 📐 **Matryoshka Dimensions**: Ghost (128) → God (4096) dynamic scaling
- 🧠 **Neural Confidence Probes**: Per-layer capacity allocation
- ⚡ **Energy Penalty Loss**: "Lazy but correct" compute minimization
- 🪆 **Progressive Training**: 6B → 10B → 20B staged expansion

## Architecture

```
nanochat-d32 (1.9B)
    ↓ Surgery (expand + add Mamba)
Stage 1: 6B  (dim=2560)  ← Train ~$1,200
    ↓ Progressive expand
Stage 2: 10B (dim=3072)  ← Train ~$700  
    ↓ Progressive expand
Stage 3: 20B (dim=4096)  ← Train ~$1,500
```

## Training Cost Estimates

| Stage | Model Size | New Params | Tokens | Hours (8×H100) | Cost |
|-------|------------|------------|--------|----------------|------|
| 1 | 6B | ~4B | 80B | 50h | $1,200 |
| 2 | 10B | ~2B | 40B | 30h | $700 |
| 3 | 20B | ~5B | 100B | 60h | $1,500 |
| **Total** | **20B** | | | **140h** | **~$3,400** |

*Still 1000× cheaper than training 20B from scratch (~$10M+)*

## Quick Start

```bash
# 1. Download nanochat-d32 base
huggingface-cli download karpathy/nanochat-d32 \
    --local-dir ~/.cache/nanochat/chatsft_checkpoints/d32

# 2. Stage 1: Create 6B hybrid
python -m scripts.surgery --new-dim=2560

# 3. Train Stage 1
torchrun --nproc_per_node=8 -m scripts.fractal_train \
    --checkpoint ~/.cache/nanochat/hybrid_checkpoints/d32_2560/model.pt \
    --expanded-dim=2560 --matryoshka --sample-dim

# 4. Stage 2: Expand trained 6B → 10B
python -m scripts.surgery --expand-from=2560 --new-dim=3072

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
| Ghost | 128 | ~0.1% | Trivial tasks |
| Whisper | 512 | ~4% | Simple Q&A |
| Normal | 2048 | ~25% | General use |
| Think | 4096 | 100% | Complex reasoning |

## Mamba:Attention Ratio

We use 1:1 interleaved (32 Mamba + 32 Attention). Research suggests:
- 4:1 Mamba:Attention common (Jamba)
- 1:1 simpler, works well for our scale
- Mamba provides O(n) efficiency for long context (128K feasible)

---

*Based on [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy*
