# Adamba

> **Ad**aptive **Mamba**: Elastic compute with dynamic Matryoshka scaling

**Fork of [karpathy/nanochat](https://github.com/karpathy/nanochat)**

*Also known as: ElasticGPT • Accordion-Net • Dynamic Compute Budget Model*

## Architecture Overview

Adamba combines three efficiency techniques into a unified pipeline:

| Technique | Implementation | Purpose |
|-----------|----------------|---------|
| **Matryoshka (MRL)** | Width: 128 → 4096 per layer | Elastic compute |
| **Early Exit** | ConfidenceGate | Skip layers when confident |
| **Static SSM** | Mamba at full dim | Stable memory backbone |

```
┌─────────────────────────────────────────────────┐
│  PROMPT → LayerDimPredictor → [dim per layer]   │
│                                                 │
│  Attention + MLP: Dynamic (sliced)              │
│  Mamba:           Static (full dim)             │
│                                                 │
│  Gate > 0.95 → EXIT EARLY                       │
│  Gate < 0.50 → EXPAND remaining layers          │
└─────────────────────────────────────────────────┘
```

## V2 Architecture ✨

- 🎯 **LayerDimPredictor**: Predicts per-layer dims upfront (no graph breaks)
- 🚪 **ConfidenceGate**: Unified early exit + dim expansion
- 📦 **MatryoshkaKVCache**: Slice-down cache strategy
- 🧠 **Static Mamba**: Uses efficient CUDA kernel (no SSM state resizing)

**Key insight**: Resizing SSM states on the fly is mathematically messy. Resizing Attention heads is trivial. Keep Mamba static, make Attention/MLP dynamic.

## Validated Results

```
tiny_experiment validation:
  Early exits: 71.5%  (37.5% compute saved!)
  Gate loss:   0.28 → 0.03  (self-supervised difficulty learning)
  Hard tasks get more dims than easy tasks ✓
```

The gate trains itself using **shadow loss**: comparing what loss *would be* at each layer to teach the gate when it's safe to exit.

## Architecture

```
nanochat-d32 (1.9B, 32 layers, dim=2048)
    ↓ Surgery (add 32 Mamba layers)
Stage 1: 6.4B  (dim=2048)  ← Hybrid, no expansion
    ↓ Progressive expand
Stage 2: 9.3B  (dim=2560)  
    ↓ Progressive expand
Stage 3: 20B   (dim=4096)
```

## Training Cost

| Stage | Model Size | Dim | Hours (8×H100) | Est. Cost |
|-------|------------|-----|----------------|-----------|
| 1 | 6.4B | 2048 | 40h | $1,000 |
| 2 | 9.3B | 2560 | 50h | $1,200 |
| 3 | 20B | 4096 | 100h | $2,400 |
| **Total** | | | **190h** | **~$4,600** |

## Quick Start

```bash
# 1. Download nanochat-d32 base
huggingface-cli download karpathy/nanochat-d32 \
    --local-dir ~/.cache/nanochat/chatsft_checkpoints/d32

# 2. Create 6.4B hybrid
python -m scripts.surgery --new-dim=2048

# 3. Train Stage 1
torchrun --nproc_per_node=8 -m scripts.fractal_train \
    --checkpoint ~/.cache/nanochat/hybrid_checkpoints/d32_2048/model.pt \
    --expanded-dim=2048 --matryoshka --sample-dim

# 4. Expand → Stage 2
python -m scripts.surgery --expand-from=2048 --new-dim=2560
```

## Files

| File | Purpose |
|------|---------|
| `nanochat/hybrid_gpt.py` | Adamba model (Mamba+Attention+Gate) |
| `nanochat/mamba_block.py` | Static Mamba with SSM fallback |
| `nanochat/matryoshka.py` | Dimension slicing + energy loss |
| `nanochat/confidence_probe.py` | V2: LayerDimPredictor, ConfidenceGate |
| `scripts/surgery.py` | Create/expand hybrid checkpoints |
| `scripts/fractal_train.py` | Matryoshka training |
| `tiny_experiment/` | Local validation suite |

## Compute Modes

| Mode | Dim | Compute | Use Case |
|------|-----|---------|----------|
| Ghost | 128 | ~0.4% | Trivial tasks |
| Whisper | 512 | ~6% | Simple Q&A |
| Normal | 1024 | ~25% | General use |
| Think | 2048+ | 100% | Complex reasoning |

## Related Work

- **Matryoshka Embeddings** (OpenAI/Harvard): MRL applied to model weights
- **FastBERT / DeeBERT**: Confidence-based early exit
- **Mixture of Depths** (Google DeepMind): Dynamic FLOP allocation

---

## TODO Before Real Training

### 1. Stage-Dependent Freeze/Unfreeze

**Stage 1 (Train Mamba):**
- Freeze: All attention layers (even blocks)
- Train: Mamba layers (odd blocks) + Gates + DifficultyEstimator
- Why: Saves 50% optimizer VRAM → bigger batch → more tokens per $

**Stage 2+ (Expansion):**
- Freeze: Original dims (first 2048)
- Train: NEW expanded dims only + Mamba
- Later: Unfreeze all with lower LR for fine-tuning

**Implementation:** Add `--stage` flag to `scripts/fractal_train.py`:
```python
if args.stage == 1:
    freeze_even_blocks()  # Attention
    train_odd_blocks()    # Mamba
elif args.stage == 2:
    freeze_first_n_dims(2048)
    train_expanded_dims()
```

### 2. Smarter Expansion Initialization

**Current (Stage 1):** Mamba uses `zero-init` ✓ (correct, nothing to copy)

**For Stage 2/3 Expansion:** Don't just use zeros. Options:

| Method | Description | Quality |
|--------|-------------|---------|
| **LoRA-style** | `new = A @ B` (low-rank, small init) | ⭐⭐⭐ |
| Copy+Scale | Copy first 512 dims, scale by 0.1 | ⭐⭐ |
| SVD Extension | Extrapolate singular values | ⭐⭐⭐ |

**Recommended:** LoRA-style for expanded dims (retains structure, gradients flow well)

---

*Based on [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy*
