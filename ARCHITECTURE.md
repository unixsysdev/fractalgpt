# Adamba: Adaptive Mamba Architecture

> **Ad**aptive **Mamba**: Elastic compute with dynamic Matryoshka scaling

## Overview

Adamba is a **Compute-Adaptive Language Model** that modulates both:
- **Width** (via Matryoshka slicing) — variable dimensions per layer
- **Depth** (via Early Exit) — skip layers when confident

This creates an architecture that "stretches and shrinks to fit the problem."

## Key Components

### 1. DifficultyEstimator (Trainable Dimension Predictor)

```python
class DifficultyEstimator(nn.Module):
    """
    Predicts per-layer difficulty scores (0-1) via REGRESSION.
    Maps to dimensions via lookup table.
    
    CRITICAL: Uses sigmoid (not argmax) to remain differentiable!
    """
```

**Training Signal:**
```python
difficulty_target = torch.sigmoid(loss.detach() - 1.0)
difficulty_loss = F.mse_loss(difficulty_pred, difficulty_target)
```
- High loss → High difficulty → More dimensions
- Low loss → Low difficulty → Fewer dimensions

### 2. ConfidenceGate (Early Exit Controller)

```python
class ConfidenceGate(nn.Module):
    """Returns confidence in [0, 1]. High = safe to exit."""
```

**Training Signal (Shadow Loss):**
```python
exit_loss = cross_entropy(lm_head(ln_f(hidden_at_layer_i)), targets)
target = torch.sigmoid(2.0 - exit_loss)  # Low loss → High confidence
gate_loss = mse_loss(confidence, target)
```

### 3. ExpansionGate (Dim Expansion Controller)

If stuck near the end with low confidence, expand remaining layer dimensions.

### 4. Static Mamba + Dynamic Attention

**Key engineering insight:** 
- **Mamba**: Always full dimension (SSM state resizing is mathematically messy)
- **Attention/MLP**: Dynamic dimensions (trivial to slice)

This maintains a stable "memory backbone" while compressing "reasoning" compute.

## Causality Safety

**Problem:** Using future tokens to predict dims for past tokens leaks information.

**Solution:** Separate forward passes:
- `forward_train()`: Random dims (Matryoshka dropout), no predictor
- `forward_inference()`: Predictor + gates (safe at inference)

## Training Efficiency

**Single Forward Pass Optimization:**
```python
loss, metrics = model.forward_train(x, y, return_hidden_states=True)
# Reuse hidden_states for gate training
for i, hidden in enumerate(metrics['hidden_states']):
    exit_loss = compute_exit_loss(hidden)
    gate_loss += train_gate(hidden, exit_loss)
```

**Warmup Period:**
- Steps 0-200: Train main model only
- Steps 200+: Add gate and difficulty training

---

## Connection to Existing Research

### Matryoshka Embeddings (OpenAI/Harvard, 2022)
Our implementation of slicing MLP and Attention weights is a direct application of **Matryoshka Representation Learning (MRL)** to model weights. Often called "Nested Dropout" — the first dimensions contain the most important information.

**Reference:** Kusupati et al., "Matryoshka Representation Learning"

### Early Exits (FastBERT / DeeBERT, 2020)
Our confidence-based exit mechanism is the standard implementation for early-exit transformers:
- Attach classifier at each layer
- Exit when confidence exceeds threshold

**References:**
- Liu et al., "FastBERT: a Self-distilling BERT with Adaptive Inference Time"
- Xin et al., "DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference"

### Mixture of Depths (Google DeepMind, April 2024)
They dynamically allocate FLOPs by skipping blocks for **specific tokens**. Our approach differs:
- **MoD:** Skip blocks for specific tokens (per-token routing)
- **Adamba:** Skip layers for whole sequence (per-sequence routing)

Same goal: **iso-FLOP efficiency** — same compute, better quality.

**Reference:** Raposo et al., "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"

### Speculative Decoding / Early Exit LLMs
Related work on making inference more efficient:
- SkipDecode (2023)
- Calm (Schuster et al., 2022)
- LayerSkip (Meta, 2024)

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│  PROMPT                                                     │
│     ↓                                                       │
│  DifficultyEstimator → difficulty_scores [0.3, 0.4, ...]   │
│     ↓                                                       │
│  Map to dims → [64, 128, 128, 256, ...]                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 0: Attention(64) + MLP(64) + Mamba(256)             │
│           Gate: 0.48                                        │
│                                                             │
│  Layer 4: Gate: 0.78                                        │
│                                                             │
│  Layer 6: Gate: 0.95 → EXIT EARLY ✓                        │
│       OR  Gate: 0.3  → EXPAND remaining layers              │
└─────────────────────────────────────────────────────────────┘
```

## Training Pipeline

| Phase | Model Size | Dim | What Trains |
|-------|------------|-----|-------------|
| Stage 1 | 6.4B | 2048 | Weights + Gates + Predictor |
| Stage 2 | 9.3B | 2560 | Fine-tune expanded weights |
| Stage 3 | 20B | 4096 | Final fine-tune |

## Validation Results (Tiny Experiment)

```
Config: 8 layers, 256 dim, 9M params

Training:
  Main loss: 31.89 → 0.12 ✓
  Gate loss: 0.46 → 0.0003 ✓
  Confidences: [0.54, 0.65, 0.65, 0.64, 0.65, 0.79, 0.79, 0.86]
  
Observation: Gate learned to increase confidence through layers.
The toy task requires all 8 layers — no early exit opportunities.
A real LLM with varied difficulty inputs would show early exits.
```

## Files

| File | Purpose |
|------|---------|
| `nanochat/hybrid_gpt.py` | Main Adamba model |
| `nanochat/confidence_probe.py` | DifficultyEstimator, Gates |
| `nanochat/mamba_block.py` | Static Mamba implementation |
| `nanochat/matryoshka.py` | Dimension slicing utilities |
| `tiny_experiment/model_v2.py` | Standalone validation model |
| `tiny_experiment/train_v2.py` | Validation training script |

---

*Adamba: Elastic intelligence that expands when challenged, contracts when confident.*
