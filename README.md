# Nano-Fractal V2

> Emergent Fractal Intelligence: Hybrid Mamba-Transformer with dynamic Matryoshka scaling.

**Fork of [karpathy/nanochat](https://github.com/karpathy/nanochat)**

## V2 Architecture ✨

**NEW** - GPU-friendly adaptive compute:
- 🎯 **LayerDimPredictor**: Predicts per-layer dims upfront (no graph breaks)
- 🚪 **ConfidenceGate**: Early exit (71% savings) + dim expansion
- 📦 **MatryoshkaKVCache**: Slice-down cache strategy
- 🧠 **Static Mamba**: Uses efficient CUDA kernel (no padding)

```
┌─────────────────────────────────────────────────┐
│  PROMPT → LayerDimPredictor → [dim per layer]   │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│  Layer 0:  Attention(256) + MLP(256)           │
│            Gate: 0.48                           │
│  Layer 16: Gate > 0.95 → EXIT EARLY ✓           │
│       OR   Gate < 0.5  → EXPAND remaining       │
└─────────────────────────────────────────────────┘
```

## What's New

- 🧬 **Hybrid Architecture**: 32 Attention + 32 Mamba layers (64 total)
- 📐 **Matryoshka Dimensions**: Ghost (128) → God (4096) dynamic scaling
- 🧠 **V2 Confidence Gate**: Unified early exit + expansion (replaces per-layer probes)
- ⚡ **Energy Penalty Loss**: "Lazy but correct" compute minimization
- 🪆 **Progressive Training**: 6B → 9B → 20B staged expansion

## Validated in tiny_experiment

```
Early exits: 71.5%  (37.5% compute saved!)
Gate loss:   0.28 → 0.03  (learned when to exit)
Hard tasks get more dims than easy tasks ✓
```

## Architecture

```
nanochat-d32 (1.9B, 32 layers, dim=2048)
    ↓ Surgery (add 32 Mamba layers)
Stage 1: 6.4B  (dim=2048)  ← Add Mamba, no expansion
    ↓ Progressive expand
Stage 2: 9.3B  (dim=2560)  
    ↓ Progressive expand
Stage 3: 20B   (dim=4096)
```

## Training Cost Estimates

| Stage | Model Size | Dim | Hours (8×H100) | Est. Cost |
|-------|------------|-----|----------------|-----------|
| 1 | 6.4B | 2048 | 40h | $1,000 |
| 2 | 9.3B | 2560 | 50h | $1,200 |
| 3 | 20B | 4096 | 100h | $2,400 |
| **Total** | | | **190h** | **~$4,600** |

*Or stop at any stage - 6.4B alone costs ~$1,000*

## Quick Start

```bash
# 1. Download nanochat-d32 base
huggingface-cli download karpathy/nanochat-d32 \
    --local-dir ~/.cache/nanochat/chatsft_checkpoints/d32

# 2. Stage 1: Create 6.4B hybrid
python -m scripts.surgery --new-dim=2048

# 3. Train Stage 1
torchrun --nproc_per_node=8 -m scripts.fractal_train \
    --checkpoint ~/.cache/nanochat/hybrid_checkpoints/d32_2048/model.pt \
    --expanded-dim=2048 --matryoshka --sample-dim

# 4. Stage 2: Expand trained 6.4B → 9.3B
python -m scripts.surgery --expand-from=2048 --new-dim=2560
```

## New Files

| File | Purpose |
|------|---------|
| `nanochat/hybrid_gpt.py` | HybridGPT V2 (Mamba+Attention+Gate) |
| `nanochat/mamba_block.py` | Static Mamba with SSM fallback |
| `nanochat/matryoshka.py` | Dimension slicing + energy loss |
| `nanochat/confidence_probe.py` | V2: LayerDimPredictor, ConfidenceGate |
| `scripts/surgery.py` | Create/expand hybrid checkpoints |
| `scripts/fractal_train.py` | Matryoshka training |
| `tiny_experiment/` | Local validation suite |

## Matryoshka Levels

| Mode | Dim | Compute | Use Case |
|------|-----|---------|----------|
| Ghost | 128 | ~0.4% | Trivial tasks |
| Whisper | 512 | ~6% | Simple Q&A |
| Normal | 1024 | ~25% | General use |
| Think | 2048+ | 100% | Complex reasoning |

---

*Based on [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy*
