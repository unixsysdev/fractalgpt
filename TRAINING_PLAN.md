# Adamba Agent Training Plan

## Overview

Complete training pipeline for Adamba Agent with progressive expansion, CoT, function calling, and RL.

**Provider:** HPC-AI.COM (8× H200 @ $1.96/GPU/hr = $15.68/hr)

## Training Phases

| Phase | Type | Dim | Data | Hours | Cost |
|-------|------|-----|------|-------|------|
| **1** | Mamba Integration | 2048 | FineWeb | ~15 | ~$235 |
| **2** | Matryoshka + Gates | 2048 | FineWeb + SFT | ~12 | ~$188 |
| **3a** | Expansion Surgery | 2048→2560 | - | 0 | $0 |
| **3b** | Train Expanded | 2560 | FineWeb + SFT | ~10 | ~$157 |
| **4a** | Expansion Surgery | 2560→4096 | - | 0 | $0 |
| **4b** | Train Expanded | 4096 | FineWeb + SFT | ~8 | ~$125 |
| **5** | CoT + Function Call | 4096 | CoT + Tools | ~10 | ~$157 |
| **6** | RL (GRPO) | 4096 | Trajectories | ~5 | ~$78 |
| | | | **Total** | **~60 hrs** | **~$940** |

**Fits within $1,000 budget with ~$60 buffer for reruns.**

---

## Phase 1: Mamba Integration

**Goal:** Mamba learns to work with frozen attention/MLP  
**Dim:** 2048 (original)

```bash
# Data
python -m nanochat.dataset --dataset=fineweb --num-files=100

# Training
torchrun --nproc_per_node=8 -m scripts.fractal_train \
    --phase=1 \
    --checkpoint=~/.cache/nanochat/hybrid_checkpoints/d32_2048/model.pt
```

---

## Phase 2: Matryoshka + Gates

**Goal:** All layers learn variable dimensions + early exit  
**Dim:** 2048

```bash
# Data (mix)
python -m nanochat.dataset --dataset=fineweb --num-files=50
python -m nanochat.dataset --dataset=openorca

# Training
torchrun --nproc_per_node=8 -m scripts.fractal_train \
    --phase=2 \
    --checkpoint=phase1_trained.pt
```

---

## Phase 3: First Expansion (2048 → 2560)

### 3a. Surgery
```bash
python -m scripts.surgery --expand-from=2048 --new-dim=2560
```

### 3b. Train Expanded Weights
```bash
torchrun --nproc_per_node=8 -m scripts.fractal_train \
    --phase=3 \
    --checkpoint=~/.cache/nanochat/hybrid_checkpoints/d32_2560/model.pt \
    --expanded-dim=2560
```

---

## Phase 4: Second Expansion (2560 → 4096)

### 4a. Surgery
```bash
python -m scripts.surgery --expand-from=2560 --new-dim=4096
```

### 4b. Train Expanded Weights
```bash
torchrun --nproc_per_node=8 -m scripts.fractal_train \
    --phase=3 \
    --checkpoint=~/.cache/nanochat/hybrid_checkpoints/d32_4096/model.pt \
    --expanded-dim=4096
```

---

## Phase 5: CoT + Function Calling

**Goal:** Model learns reasoning and tool use  
**Dim:** 4096 (final)

**Datasets:**

| Dataset | Source | Size | Focus |
|---------|--------|------|-------|
| OpenOrca | `Open-Orca/OpenOrca` | 4M | General reasoning |
| FLAN-CoT | `conceptofmind/FLAN_CoT` | 1M | Reasoning traces |
| MetaMathQA | `meta-math/MetaMathQA` | 400K | Math CoT |
| Glaive-FC | `glaiveai/glaive-function-calling-v2` | 113K | Function schemas |

```bash
# Download datasets
python -m nanochat.dataset --dataset=openorca
python -m nanochat.dataset --dataset=flan_cot
python -m nanochat.dataset --dataset=metamath
python -m nanochat.dataset --dataset=glaive_fc

# Training (SFT mode)
torchrun --nproc_per_node=8 -m scripts.fractal_train \
    --phase=2 \
    --checkpoint=phase4b_trained.pt \
    --sft-mode
```

---

## Phase 6: RL (GRPO/DPO)

**Goal:** Optimize for task completion, not token likelihood

**Method:** GRPO (simpler than PPO) or DPO (offline)

**Data:**
- Agent trajectories with success/failure labels
- Tool execution outcomes
- User preference pairs

```bash
torchrun --nproc_per_node=8 -m scripts.rl_train \
    --method=grpo \
    --checkpoint=phase5_trained.pt
```

---

## Adamba-Specific Benefits

**CoT + Early Exit synergy:**
- Easy questions → short reasoning → early exit at layer 4-6
- Hard questions → long CoT → full 64 layers

**Function calling + Matryoshka:**
- Simple `get_time()` → 128 dims (fast)
- Complex multi-tool chains → 4096 dims (accurate)

---

## Dataset Registry

| ID | HuggingFace Path | Type |
|----|------------------|------|
| `fineweb` | `karpathy/fineweb-edu-100b-shuffle` | Pre-train |
| `openorca` | `Open-Orca/OpenOrca` | SFT |
| `flan_cot` | `conceptofmind/FLAN2021_CoT` | CoT |
| `metamath` | `meta-math/MetaMathQA` | Math |
| `glaive_fc` | `glaiveai/glaive-function-calling-v2` | Tools |
| `toolbench` | `ToolBench/ToolBench` | Tools |
| `sharegpt` | `anon8231489123/ShareGPT_Vicuna_unfiltered` | Chat |

---

## Budget Summary (HPC-AI.COM, 8× H200)

**Rate:** $1.96/GPU/hr × 8 = $15.68/hr

| Phase | Description | Hours | Cost |
|-------|-------------|-------|------|
| 1 | Mamba Integration | 15 | $235 |
| 2 | Matryoshka + Gates | 12 | $188 |
| 3b | Train 2560 | 10 | $157 |
| 4b | Train 4096 | 8 | $125 |
| 5 | CoT + Tools | 10 | $157 |
| 6 | RL | 5 | $78 |
| **Total** | | **60** | **$940** |

**Alternative (Single Expansion to 2560):** Skip Phase 4, save ~$125
