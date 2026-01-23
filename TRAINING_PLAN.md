# Adamba Agent Training Plan

## Overview

Complete training pipeline for Adamba Agent with CoT, function calling, and RL.

## Training Phases

| Phase | Type | Data | Duration | Cost |
|-------|------|------|----------|------|
| **1** | Pre-train | FineWeb-Edu | ~20 hrs | ~$240 |
| **2** | Pre-train + SFT | FineWeb + OpenOrca | ~20 hrs | ~$240 |
| **3** | Expansion | Surgery only | - | - |
| **4** | SFT | CoT datasets | ~10 hrs | ~$120 |
| **5** | SFT | Function calling | ~10 hrs | ~$120 |
| **6** | RL | Agent trajectories | ~5 hrs | ~$80 |

**Total: ~65 hours, ~$800** (buffer for reruns)

---

## Phase 1: Mamba Integration

**Goal:** Mamba learns to work with frozen attention/MLP

**Data:** FineWeb-Edu (pre-training data)
```bash
python -m nanochat.dataset --dataset=fineweb --num-files=100
```

**Training:**
```bash
torchrun -m scripts.fractal_train --phase=1 --checkpoint=...
```

---

## Phase 2: Matryoshka + Gates

**Goal:** All layers learn variable dimensions + early exit

**Data:** Mix of pre-train + instruction
```bash
python -m nanochat.dataset --dataset=fineweb --num-files=50
python -m nanochat.dataset --dataset=openorca --num-files=10
```

**Training:**
```bash
torchrun -m scripts.fractal_train --phase=2 --checkpoint=...
```

---

## Phase 3: Expansion

**Goal:** Increase model capacity (2048 → 2560)

**Surgery:**
```bash
python -m scripts.surgery --expand-from=2048 --new-dim=2560
```

**Training:**
```bash
torchrun -m scripts.fractal_train --phase=3 --checkpoint=...
```

---

## Phase 4: Chain-of-Thought

**Goal:** Model learns step-by-step reasoning

**Datasets:**

| Dataset | Source | Size | Focus |
|---------|--------|------|-------|
| OpenOrca | `Open-Orca/OpenOrca` | 4M | General reasoning |
| FLAN-CoT | `conceptofmind/FLAN_CoT` | 1M | Reasoning traces |
| GSM8K | `gsm8k` | 8K | Math reasoning |
| MetaMathQA | `meta-math/MetaMathQA` | 400K | Math CoT |

```bash
python -m nanochat.dataset --dataset=openorca
python -m nanochat.dataset --dataset=flan_cot
python -m nanochat.dataset --dataset=metamath
```

---

## Phase 5: Function Calling

**Goal:** Model learns to output structured tool calls

**Datasets:**

| Dataset | Source | Size | Focus |
|---------|--------|------|-------|
| Glaive-FC | `glaiveai/glaive-function-calling-v2` | 113K | Function schemas |
| ToolBench | `ToolBench/ToolBench` | 126K | Multi-tool |
| Gorilla | `gorilla-llm/APIBench` | 17K | API calls |

```bash
python -m nanochat.dataset --dataset=glaive_fc
python -m nanochat.dataset --dataset=toolbench
```

**Format:**
```json
{"messages": [
  {"role": "user", "content": "What's the weather?"},
  {"role": "assistant", "content": null, "function_call": {"name": "get_weather", "arguments": "{\"location\": \"NYC\"}"}}
]}
```

---

## Phase 6: RL (GRPO/DPO)

**Goal:** Optimize for task completion, not token likelihood

**Method:** GRPO (simpler than PPO) or DPO (offline)

**Data:**
- Agent trajectories with success/failure labels
- Tool execution outcomes
- User preference pairs

**Training:**
```bash
torchrun -m scripts.rl_train --method=grpo --checkpoint=...
```

---

## Adamba-Specific Benefits

**CoT + Early Exit synergy:**
- Easy questions → short reasoning → early exit at layer 4-6
- Hard questions → long CoT → full 64 layers

**Function calling + Matryoshka:**
- Simple `get_time()` → 128 dims (fast)
- Complex multi-tool chains → 2048 dims (accurate)

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

## Budget Summary

**At Hyperbolic ($1.49/GPU/hr × 8 GPUs = $11.92/hr):**

| Phase | Hours | Cost |
|-------|-------|------|
| 1-2 | 40 | $477 |
| 3 | 10 | $119 |
| 4-5 | 20 | $238 |
| 6 | 5 | $60 |
| **Total** | **75** | **$894** |

**Fits within $1,000 budget with ~$100 buffer.**
