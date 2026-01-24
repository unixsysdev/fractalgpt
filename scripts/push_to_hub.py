#!/usr/bin/env python3
"""
Push Adamba checkpoint to HuggingFace Hub.

Supports two repos:
  - datasysdev/adamba: Original nanochat-based (6B-20B)
  - datasysdev/gptoss-adamba: GPT-OSS 20B MoE based (22B+)

Usage:
    # Original Adamba
    python scripts/push_to_hub.py --checkpoint path/model.pt --hf-token TOKEN --variant phase1_6b_base
    
    # GPT-OSS Adamba (MoE)
    python scripts/push_to_hub.py --checkpoint path/model.pt --hf-token TOKEN --variant gptoss_phase1 --repo-id gptoss-adamba
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

# Model variants for the README
VARIANTS = {
    # Original Adamba (nanochat-based)
    "phase1_6b_base": {"params": "6.4B", "dim": 2048, "features": ["mamba_integration"], "status": "training", "base": "nanochat"},
    "phase2_6b_matryoshka": {"params": "6.4B", "dim": 2048, "features": ["matryoshka", "early_exit"], "status": "pending", "base": "nanochat"},
    "phase3_9b_matryoshka": {"params": "9.3B", "dim": 2560, "features": ["matryoshka", "early_exit"], "status": "pending", "base": "nanochat"},
    "phase3_20b_matryoshka": {"params": "20B", "dim": 4096, "features": ["matryoshka", "early_exit"], "status": "pending", "base": "nanochat"},
    "sft_20b": {"params": "20B", "dim": 4096, "features": ["matryoshka", "early_exit", "sft"], "status": "pending", "base": "nanochat"},
    "rl_20b": {"params": "20B", "dim": 4096, "features": ["matryoshka", "early_exit", "rl_agent"], "status": "pending", "base": "nanochat"},
    
    # GPT-OSS Adamba (MoE-based, 22B)
    "gptoss_phase1": {"params": "21.9B", "dim": 2880, "features": ["mamba_integration", "moe_32experts"], "status": "training", "base": "gpt-oss-20b"},
    "gptoss_phase2": {"params": "21.9B", "dim": 2880, "features": ["matryoshka", "early_exit", "moe_32experts"], "status": "pending", "base": "gpt-oss-20b"},
    "gptoss_phase3": {"params": "30B+", "dim": 4096, "features": ["matryoshka", "early_exit", "moe_32experts", "expansion"], "status": "pending", "base": "gpt-oss-20b"},
    "gptoss_sft": {"params": "21.9B", "dim": 2880, "features": ["matryoshka", "moe_32experts", "sft"], "status": "pending", "base": "gpt-oss-20b"},
}

def main():
    parser = argparse.ArgumentParser(description="Push Adamba model to HuggingFace Hub")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--hf-token", type=str, required=True, help="HuggingFace API token")
    parser.add_argument("--repo-id", type=str, default="adamba", help="HuggingFace repo name (adamba or gptoss-adamba)")
    parser.add_argument("--org", type=str, default="datasysdev", help="HuggingFace organization/username")
    parser.add_argument("--variant", type=str, default="phase1_6b_base", 
                        choices=list(VARIANTS.keys()),
                        help="Model variant name (use gptoss_* variants for gptoss-adamba repo)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--update-readme", action="store_true", help="Also update README with new variant")
    args = parser.parse_args()
    
    # Check dependencies
    try:
        from huggingface_hub import HfApi, create_repo, upload_file, hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import HfApi, create_repo, upload_file, hf_hub_download
    
    # Verify checkpoint exists
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Setup repo ID
    repo_id = f"{args.org}/{args.repo_id}"
    variant_info = VARIANTS[args.variant]
    
    print(f"📦 Preparing to push {args.variant} to: {repo_id}")
    
    api = HfApi(token=args.hf_token)
    
    # Create repo if needed
    try:
        create_repo(repo_id, token=args.hf_token, private=args.private, exist_ok=True)
        print(f"✅ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"⚠️  Repo note: {e}")
    
    # Upload checkpoint with variant name
    checkpoint_name = f"{args.variant}.pt"
    print(f"📤 Uploading {checkpoint_name}...")
    api.upload_file(
        path_or_fileobj=str(args.checkpoint),
        path_in_repo=f"checkpoints/{checkpoint_name}",
        repo_id=repo_id,
        token=args.hf_token,
    )
    
    # Upload config for this variant
    is_gptoss = args.variant.startswith("gptoss_")
    config = {
        "variant": args.variant,
        "model_type": "adamba-moe" if is_gptoss else "adamba-hybrid",
        "architecture": "HybridMoEGPT (Attention + MoE + Mamba)" if is_gptoss else "HybridGPT (Attention + Mamba)",
        "base_model": variant_info.get("base", "unknown"),
        "parameters": variant_info["params"],
        "n_embd": variant_info["dim"],
        "features": variant_info["features"],
        "n_layers": 36 if is_gptoss else 64,  # 24 Attn + 12 Mamba for GPT-OSS
        "vocab_size": 201088 if is_gptoss else 65536,
        "num_experts": 32 if is_gptoss else None,
        "experts_per_token": 4 if is_gptoss else None,
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        config_path = f.name
    
    api.upload_file(
        path_or_fileobj=config_path,
        path_in_repo=f"checkpoints/{args.variant}_config.json",
        repo_id=repo_id,
        token=args.hf_token,
    )
    os.unlink(config_path)
    
    # Update README
    readme = generate_readme(args.variant)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(readme)
        readme_path = f.name
    
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        token=args.hf_token,
    )
    os.unlink(readme_path)
    
    print(f"🎉 Successfully pushed {args.variant} to: https://huggingface.co/{repo_id}")
    print(f"   Checkpoint: checkpoints/{checkpoint_name}")


def generate_readme(current_variant):
    is_gptoss = current_variant.startswith("gptoss_")
    
    # Filter variants for the current repo type
    if is_gptoss:
        filtered_variants = {k: v for k, v in VARIANTS.items() if k.startswith("gptoss_")}
    else:
        filtered_variants = {k: v for k, v in VARIANTS.items() if not k.startswith("gptoss_")}
    
    # Build variant table
    variant_rows = []
    for name, info in filtered_variants.items():
        status = "✅" if name == current_variant else ("🔄" if info["status"] == "training" else "⏳")
        features = ", ".join(info["features"])
        link = f"[Download](./checkpoints/{name}.pt)" if name == current_variant else "—"
        variant_rows.append(f"| {name} | {info['params']} | {info['dim']} | {features} | {status} | {link} |")
    
    variant_table = "\n".join(variant_rows)
    
    if is_gptoss:
        return f'''---
license: apache-2.0
tags:
- pytorch
- transformer
- mamba
- moe
- hybrid
- matryoshka
- gpt-oss
- adaptive-compute
pipeline_tag: text-generation
---

# 🌀 GPT-OSS Adamba: Hybrid MoE + Mamba

> **21.9B** parameters | **32 experts** | **Mamba-enhanced** reasoning backbone

📂 **[GitHub](https://github.com/unixsysdev/adamba)** | 🤗 **[Original Adamba](https://huggingface.co/datasysdev/adamba)**

## Available Checkpoints

| Variant | Parameters | Dim | Features | Status | Download |
|---------|------------|-----|----------|--------|----------|
{variant_table}

## Architecture

Built on [OpenAI GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b) with Mamba integration:

| Component | Spec |
|-----------|------|
| **Base Model** | GPT-OSS 20B MoE |
| **Hidden Dim** | 2880 |
| **Attention** | 24 layers (sliding + full alternating) |
| **Mamba** | 12 layers (interleaved 2:1) |
| **MoE** | 32 experts, top-4 routing |
| **Vocab** | 201,088 tokens |
| **Total Blocks** | 36 (24 Attn + 12 Mamba) |

```
┌─────────────────────────────────────────────────────┐
│  GPT-OSS 20B (Attention + MoE)                       │
│       ↓ Surgery (inject 12 Mamba layers)             │
│  Hybrid: A-A-M-A-A-M-... pattern                     │
│       ↓ Phase 1 (train Mamba only)                   │
│  Mamba learns to "speak GPT-OSS language"            │
│       ↓ Phase 2 (enable Matryoshka)                  │
│  Adaptive compute: 128 → 2880 dim per layer          │
└─────────────────────────────────────────────────────┘
```

## Training Status

**Phase 1**: Mamba integration (freeze Attention+MoE, train Mamba)

## Usage

```python
# Coming soon - inference code
# See: https://github.com/unixsysdev/adamba
```

## License

Apache 2.0 (same as GPT-OSS)
'''
    else:
        return f'''---
license: apache-2.0
tags:
- pytorch
- transformer
- mamba
- hybrid
- matryoshka
- nanochat
- adaptive-compute
pipeline_tag: text-generation
---

# 🌀 Adamba: Adaptive Mamba

> **Ad**aptive **Mamba**: Elastic compute with dynamic Matryoshka scaling

📂 **[GitHub](https://github.com/unixsysdev/adamba)** | 🤗 **[GPT-OSS Adamba](https://huggingface.co/datasysdev/gptoss-adamba)**

## Available Checkpoints

| Variant | Parameters | Dim | Features | Status | Download |
|---------|------------|-----|----------|--------|----------|
{variant_table}

## Architecture Overview

Adamba combines three efficiency techniques:

| Technique | Implementation | Purpose |
|-----------|----------------|---------|
| **Matryoshka (MRL)** | Width: 128 → 4096 per layer | Elastic compute |
| **Early Exit** | ConfidenceGate per layer | Skip when confident |
| **Static SSM** | Mamba at full dim | Stable memory backbone |

## Training Pipeline

```
nanochat-d32 (1.9B)
    ↓ Surgery (add 32 Mamba layers)
Phase 1: 6.4B  (dim=2048)  ← Mamba integration
    ↓ Enable Matryoshka
Phase 2: 6.4B  (dim=2048)  ← Full training
    ↓ Progressive expand
Phase 3: 9.3B → 20B (dim=4096)
    ↓ Fine-tuning
SFT: Instruction tuning
RL:  Agent capabilities
```

## Model Details

- **Base**: [karpathy/nanochat-d32](https://huggingface.co/karpathy/nanochat-d32)
- **Architecture**: 64 blocks (32 Attention + 32 Mamba interleaved)
- **Vocabulary**: 65,536 tokens  

## Usage

```python
# Coming soon - inference code
# See: https://github.com/unixsysdev/adamba
```

## License

Apache 2.0
'''


if __name__ == "__main__":
    main()
