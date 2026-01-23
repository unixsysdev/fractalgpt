#!/usr/bin/env python3
"""
Push Adamba checkpoint to HuggingFace Hub.

Usage:
    python scripts/push_to_hub.py --checkpoint path/to/model.pt --hf-token YOUR_TOKEN
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Push Adamba model to HuggingFace Hub")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--hf-token", type=str, required=True, help="HuggingFace API token")
    parser.add_argument("--repo-id", type=str, default="adamba-hybrid-5b", help="HuggingFace repo name")
    parser.add_argument("--org", type=str, default="datasysdev", help="HuggingFace organization/username")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--phase", type=int, default=1, help="Training phase (1, 2, or 3)")
    args = parser.parse_args()
    
    # Check dependencies
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import HfApi, create_repo, upload_folder
    
    # Verify checkpoint exists
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Setup repo ID
    repo_id = f"{args.org}/{args.repo_id}"
    
    print(f"📦 Preparing to push to: {repo_id}")
    
    # Create temp directory with model files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Copy checkpoint
        shutil.copy(args.checkpoint, tmpdir / "model.pt")
        
        # Create config
        config = {
            "model_type": "adamba-hybrid",
            "architecture": "HybridGPT (Attention + Mamba)",
            "base_model": "nanochat-d32-2048",
            "training_phase": args.phase,
            "description": "Adamba: Adaptive Mamba with Matryoshka scaling",
            "parameters": "~5B",
            "vocab_size": 65536,
            "n_layers": 64,
            "n_embd": 2048,
            "status": f"phase_{args.phase}_checkpoint"
        }
        with open(tmpdir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Create README matching GitHub style
        phase_status = []
        phase_status.append("✅ Complete" if args.phase > 1 else "🔄 Current")
        phase_status.append("✅ Complete" if args.phase > 2 else "⏳ Pending")
        phase_status.append("⏳ Pending")
        
        readme = f'''---
license: apache-2.0
tags:
- pytorch
- transformer
- mamba
- hybrid
- matryoshka
- nanochat
pipeline_tag: text-generation
---

# 🌀 Adamba: Adaptive Mamba

> **Ad**aptive **Mamba**: Elastic compute with dynamic Matryoshka scaling

**Fork of [karpathy/nanochat](https://github.com/karpathy/nanochat)**

## Architecture Overview

Adamba combines three efficiency techniques into a unified pipeline:

| Technique | Implementation | Purpose |
|-----------|----------------|---------|
| **Matryoshka (MRL)** | Width: 128 → 2048 per layer | Elastic compute |
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

## 🚧 Training Status: Phase {args.phase}

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | {phase_status[0]} | Mamba integration (freeze attention) |
| 2 | {phase_status[1]} | Full training with Matryoshka |
| 3 | {phase_status[2]} | Progressive expansion to 20B |

## Model Details

- **Parameters**: ~5B
- **Architecture**: 64 interleaved blocks (32 Attention + 32 Mamba)
- **Vocabulary**: 65,536 tokens
- **Base Dimension**: 2048
- **Matryoshka Dims**: [128, 256, 512, 1024, 2048]

## Training Pipeline

```
nanochat-d32 (1.9B, 32 layers, dim=2048)
    ↓ Surgery (add 32 Mamba layers)
Stage 1: 6.4B  (dim=2048)  ← Hybrid, no expansion
    ↓ Progressive expand
Stage 2: 9.3B  (dim=2560)  
    ↓ Progressive expand
Stage 3: 20B   (dim=4096)
```

## Usage

```python
# Coming soon - inference code
# See: https://github.com/unixsysdev/adamba
```

## Links

- 📂 **GitHub**: [unixsysdev/adamba](https://github.com/unixsysdev/adamba)
- 📊 **Training Logs**: [WandB](https://wandb.ai/dalletest123/nano-fractal)

## License

Apache 2.0

---
*This is an intermediate checkpoint from active development.*
'''
        with open(tmpdir / "README.md", "w") as f:
            f.write(readme)
        
        # Create/update repo and upload
        api = HfApi(token=args.hf_token)
        
        try:
            create_repo(repo_id, token=args.hf_token, private=args.private, exist_ok=True)
            print(f"✅ Repository created/verified: {repo_id}")
        except Exception as e:
            print(f"⚠️  Repo creation note: {e}")
        
        # Upload
        print("📤 Uploading files...")
        api.upload_folder(
            folder_path=str(tmpdir),
            repo_id=repo_id,
            token=args.hf_token,
        )
        
        print(f"🎉 Successfully pushed to: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
