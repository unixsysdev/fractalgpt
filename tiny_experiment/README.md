# Adamba Tiny Experiment

Self-contained architecture validation for **Adamba V2** — the elastic compute model.

## 🎉 Results: All Systems Working!

| Metric | Value | Meaning |
|--------|-------|---------|
| **Early exits** | 71.5% | Model exits at layer 5/8 (37.5% compute saved) |
| **Loss** | 32.4 → 0.11 | Learned the task well |
| **Gate loss** | 0.28 → 0.03 | Gate learned when to exit |
| **Expansions** | 3% | Correctly rare (hard cases only) |

## Architecture: The 3-Axis Adaptive Model

```
┌─────────────────────────────────────────────────┐
│  PROMPT → LayerDimPredictor → [dim per layer]   │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│  Layer 0: Attention(64) + MLP(64)              │
│           Gate: 0.48                            │
│                                                 │
│  Layer 4: Attention(128) + MLP(128)            │
│           Gate: 1.0 → EXIT EARLY ✓              │
│                                                 │
│  (Layers 5-7 skipped!)                         │
└─────────────────────────────────────────────────┘
```

### Three Axes of Adaptation

| Axis | Mechanism | Control |
|------|-----------|---------|
| **Width** | Matryoshka dims (32→256) | LayerDimPredictor |
| **Depth** | Early exit | ConfidenceGate |
| **Time** | Context length | Standard attention |

## Key Components

### 1. LayerDimPredictor
Predicts per-layer dimensions **once, upfront** from prompt:
```
easy  task: [64, 32, 32,  32, 64, 128, 256, 256]
hard  task: [64, 32, 32, 256, 128, 128, 256, 256]
                      ↑
              Hard gets 256 earlier!
```

### 2. ConfidenceGate (Unified Control)
One gate, two actions:
- **High confidence (>0.95)** → Exit early
- **Low confidence (<0.5) near end** → Expand remaining layers

```
Confidence per layer: [0.48, 1.0, 1.0, 1.0, 1.0]
                        ↑         ↑
                   Uncertain   Confident → Exit!
```

### 3. MatryoshkaKVCache
Slice-down cache strategy:
- Store at max dim seen so far
- Slice down to match query dim
- Never pad up (Matryoshka property: first dims most important)

### 4. Static Mamba
Mamba runs at full dim (uses efficient CUDA kernel). Only Attention + MLP are dynamic.

## Files

| File | Purpose |
|------|---------|
| `model_v2.py` | Complete architecture (9M params) |
| `train_v2.py` | Training with gate learning |
| `model.py` | Original V1 (for comparison) |
| `train.py` | Original V1 training |

## Usage

```bash
# Quick architecture test
python -m tiny_experiment.model_v2

# Full training (2000 steps, ~6 min on GPU)
python -m tiny_experiment.train_v2
```

## Training Output

```
Step    0 | loss=32.4364 | gate_loss=0.2789 | avg_exit_layer=8.0
Step  200 | loss=0.0957  | gate_loss=0.0179 | avg_exit_layer=6.0
...
Step 1999 | loss=0.1149  | gate_loss=0.0258 | avg_exit_layer=5.0

Early exits: 1429/2000 (71.5%)
Expansions: 61/2000 (3.0%)

🎉 NANO-FRACTAL V2 TRAINING SUCCESSFUL!
```

## Validated Properties

| Property | Status |
|----------|--------|
| Per-layer dim control | ✅ Different dims per layer |
| Early exit | ✅ 71.5% exit before final layer |
| Gate learning | ✅ Converges well |
| Matryoshka cache | ✅ Variable dims work |
| Difficulty awareness | ✅ Hard tasks get more dims |

## Next Steps

1. Port to main `nanochat/` codebase
2. Train on real data (6.4B → 9.3B → 20B)
3. Add think tokens for emergent reasoning
