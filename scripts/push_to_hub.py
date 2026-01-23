#!/usr/bin/env python3
"""
Push Adamba checkpoint to HuggingFace Hub.

All versions go to the same repo with descriptive filenames:
  - phase1_6b_base.pt
  - phase2_6b_matryoshka.pt  
  - phase3_9b_matryoshka.pt
  - phase3_20b_matryoshka.pt
  etc.

Usage:
    python scripts/push_to_hub.py --checkpoint path/to/model.pt --hf-token TOKEN --variant phase1_6b_base
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

# Model variants for the README
VARIANTS = {
    "phase1_6b_base": {"params": "6.4B", "dim": 2048, "features": ["mamba_integration"], "status": "training"},
    "phase2_6b_matryoshka": {"params": "6.4B", "dim": 2048, "features": ["matryoshka", "early_exit"], "status": "pending"},
    "phase3_9b_matryoshka": {"params": "9.3B", "dim": 2560, "features": ["matryoshka", "early_exit"], "status": "pending"},
    "phase3_20b_matryoshka": {"params": "20B", "dim": 4096, "features": ["matryoshka", "early_exit"], "status": "pending"},
    "sft_20b": {"params": "20B", "dim": 4096, "features": ["matryoshka", "early_exit", "sft"], "status": "pending"},
    "rl_20b": {"params": "20B", "dim": 4096, "features": ["matryoshka", "early_exit", "rl_agent"], "status": "pending"},
}

def main():
    parser = argparse.ArgumentParser(description="Push Adamba model to HuggingFace Hub")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--hf-token", type=str, required=True, help="HuggingFace API token")
    parser.add_argument("--repo-id", type=str, default="adamba", help="HuggingFace repo name")
    parser.add_argument("--org", type=str, default="datasysdev", help="HuggingFace organization/username")
    parser.add_argument("--variant", type=str, default="phase1_6b_base", 
                        choices=list(VARIANTS.keys()),
                        help="Model variant name")
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
    config = {
        "variant": args.variant,
        "model_type": "adamba-hybrid",
        "architecture": "HybridGPT (Attention + Mamba)",
        "parameters": variant_info["params"],
        "n_embd": variant_info["dim"],
        "features": variant_info["features"],
        "n_layers": 64,
        "vocab_size": 65536,
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
    # Build variant table
    variant_rows = []
    for name, info in VARIANTS.items():
        status = "✅" if name == current_variant else ("🔄" if info["status"] == "training" else "⏳")
        features = ", ".join(info["features"])
        link = f"[Download](./checkpoints/{name}.pt)" if name == current_variant else "—"
        variant_rows.append(f"| {name} | {info['params']} | {info['dim']} | {features} | {status} | {link} |")
    
    variant_table = "\n".join(variant_rows)
    
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

📂 **[GitHub](https://github.com/unixsysdev/adamba)** | 🤗 **[HuggingFace](https://huggingface.co/datasysdev/adamba)**

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

```
┌─────────────────────────────────────────────────┐
│  PROMPT → LayerDimPredictor → [dim per layer]   │
│                                                 │
│  Attention + MLP: Dynamic (Matryoshka sliced)   │
│  Mamba:           Static (full dim)             │
│                                                 │
│  Gate > 0.95 → EXIT EARLY                       │
│  Gate < 0.50 → EXPAND remaining layers          │
└─────────────────────────────────────────────────┘
```

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
- **Matryoshka Dims**: [128, 256, 512, 1024, 2048, 4096]

## Usage

```python
# Coming soon - inference code
# See: https://github.com/unixsysdev/adamba
```

## Links

- 📂 **GitHub**: [unixsysdev/adamba](https://github.com/unixsysdev/adamba)
- 📊 **Training**: [WandB](https://wandb.ai/dalletest123/nano-fractal)

## License

Apache 2.0
'''


if __name__ == "__main__":
    main()
