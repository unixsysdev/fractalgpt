# Tiny Fractal Experiment

Self-contained architecture validation for Nano-Fractal.

## Purpose

Validate the architecture works before spending $1000+ on cloud training.

## Results (2000 steps, 10 seconds on CUDA)

### ✅ All Components Are Dynamic

| Dim | Attention | MLP Hidden | Mamba d_inner |
|-----|-----------|------------|---------------|
| 16 | 1 head × 16d | 64 | 32 |
| 32 | 1 head × 32d | 128 | 64 |
| 64 | 2 heads × 32d | 256 | 128 |
| 128 | 4 heads × 32d | 512 | 256 |

### ✅ Loss Scales with Dimension

```
dim= 16: loss=0.89  (limited capacity)
dim= 32: loss=0.33
dim= 64: loss=0.25
dim=128: loss=0.20  (full capacity)
```

### ✅ ASCII Visualization

```
Active dimension: 64
  Attention:
    dims:    ███████████████░░░░░░░░░░░░░░░  64/128
    heads:   ███████████████░░░░░░░░░░░░░░░   2/4
  MLP 1:
    hidden:  ███████████████░░░░░░░░░░░░░░░ 256/512
  Mamba:
    d_inner: ███████████████░░░░░░░░░░░░░░░ 128/256
```

## Files

| File | Purpose |
|------|---------|
| `model.py` | TinyFractal (2 layers, 128 dim, 536K params) |
| `data.py` | Easy/Medium/Hard contrived tasks |
| `train.py` | Basic training (500 steps) |
| `detailed_train.py` | Detailed training with component visualization |

## Usage

```bash
# Quick test (3 seconds)
python -m tiny_experiment.train

# Detailed test with visualizations (10 seconds)
python -m tiny_experiment.detailed_train
```

## Conclusions

1. **Matryoshka works** - Model operates at any dimension level
2. **All components scale** - Attention, MLP, AND Mamba are dynamic
3. **Energy penalty works** - Lower dims have higher loss but use less compute
4. **Ready for real training** - Architecture validated ✓
