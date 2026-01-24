# Adamba

> **Ad**aptive **Mamba**: Elastic compute with dynamic Matryoshka scaling

🤗 **[Original Model](https://huggingface.co/datasysdev/adamba)** | 🤗 **[GPT-OSS 20B MoE](https://huggingface.co/datasysdev/gptoss-adamba)** | 📂 **[GitHub](https://github.com/unixsysdev/adamba)**

*Also known as: ElasticGPT • Accordion-Net • Dynamic Compute Budget Model*

## GPT-OSS 20B MoE Integration 🚀

**NEW:** Adamba now supports GPT-OSS 20B MoE as a base model!

```bash
# 1. Run surgery to create hybrid checkpoint
python scripts/surgery_moe.py --dst=/path/to/gptoss_hybrid.pt

# 2. Train Phase 1 (Mamba integration)
torchrun --nproc_per_node=8 -m scripts.fractal_train \
    --phase=1 --model-type=gptoss --depth=24 \
    --checkpoint=/path/to/gptoss_hybrid.pt \
    --gradient-checkpointing \
    --device-batch-size=20
```

### Architecture (GPT-OSS + Mamba)

| Component | Count | Details |
|-----------|-------|---------|
| Attention+MoE Blocks | 24 | Frozen in Phase 1 |
| Mamba Blocks | 12 | Trainable (9% of params) |
| Total Parameters | 22.7B | |
| MoE Experts | 32 per layer | Top-2 routing |

### Phase 1 Training

- **Goal:** Train Mamba to pass-through (minimal interference)
- **Frozen:** Attention + MoE layers (original GPT-OSS knowledge preserved)
- **Trainable:** Mamba layers only (~2B params)
- **Hardware:** 8×H200 (143GB each), ~80GB/GPU utilization
- **Throughput:** ~77k tok/s

---

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

## Files

| File | Purpose |
|------|---------|
| `nanochat/hybrid_gpt.py` | Adamba model (Mamba+Attention+Gate) |
| `nanochat/mamba_block.py` | Static Mamba with SSM fallback |
| `nanochat/moe_block.py` | MoE layer for GPT-OSS |
| `scripts/surgery_moe.py` | GPT-OSS → Adamba hybrid conversion |
| `scripts/fractal_train.py` | Distributed training with FSDP |

## Training Pipeline

### Phase-Aware Training

| Phase | Command | What Trains | Matryoshka |
|-------|---------|-------------|------------|
| **1** | `--phase=1` | Mamba only (frozen attn/MoE) | ✗ Off |
| **2** | `--phase=2` | All + Gates | ✓ On |
| **3** | `--phase=3` | Expanded weights | ✓ On |

```bash
# Phase 1: Integrate Mamba (freeze attention)
torchrun -m scripts.fractal_train --phase=1 --model-type=gptoss --checkpoint=phase1.pt

# Phase 2: Matryoshka + Gates (unfreeze all)
torchrun -m scripts.fractal_train --phase=2 --model-type=gptoss --checkpoint=phase2.pt
```

## Compute Modes

| Mode | Dim | Compute | Use Case |
|------|-----|---------|----------|
| Ghost | 128 | ~0.4% | Trivial tasks |
| Whisper | 512 | ~6% | Simple Q&A |
| Normal | 1024 | ~25% | General use |
| Think | 2048+ | 100% | Complex reasoning |

---

*Based on [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy*
