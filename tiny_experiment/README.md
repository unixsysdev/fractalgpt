# Tiny Fractal Experiment

Self-contained architecture validation for Nano-Fractal.

## Purpose

Before spending $1000+ on cloud training, validate:
1. ✅ Matryoshka MLP dimension scaling works
2. ✅ Matryoshka KV cache scaling works
3. ✅ Mamba layer integration works
4. ✅ Energy penalty reduces compute usage
5. ✅ Model learns to use minimum dims for easy tasks

## Files

| File | Purpose |
|------|---------|
| `model.py` | TinyFractal model (2 layers, 128 dim) |
| `data.py` | Easy/Medium/Hard contrived tasks |
| `train.py` | Training + validation tests |

## Usage

```bash
# Run validation (takes ~2 minutes on CPU)
python -m tiny_experiment.train
```

## Expected Output

```
✓ PASS: Matryoshka works at all dims
✓ PASS: Loss decreases
✓ PASS: Energy penalty works
✓ PASS: Hard > Medium > Easy dims

🎉 ALL TESTS PASSED - Architecture validated!
```

## What It Tests

### Task Types
- **Easy**: Copy tasks ("copy: abc" → "abc")
- **Medium**: Counting, simple arithmetic
- **Hard**: Multi-step problems (reverse + add)

### Validation
- Model should use ~16-32 dims for easy tasks
- Model should use ~64-128 dims for hard tasks
- Energy usage should decrease over training
