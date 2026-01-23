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
    parser.add_argument("--repo-id", type=str, default="adamba-hybrid", help="HuggingFace repo name")
    parser.add_argument("--org", type=str, default=None, help="HuggingFace organization (optional)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
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
    repo_id = f"{args.org}/{args.repo_id}" if args.org else args.repo_id
    
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
            "training_phase": 1,
            "description": "Adamba: Attention-Mamba hybrid with Matryoshka embedding support",
            "parameters": "~5B",
            "vocab_size": 65536,
            "n_layers": 64,
            "n_embd": 2048,
            "status": "in_development"
        }
        with open(tmpdir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Create README
        readme = """---
license: apache-2.0
tags:
- pytorch
- transformer
- mamba
- hybrid
- matryoshka
---

# 🌀 Adamba: Attention-Mamba Hybrid

**Adamba** is an experimental hybrid architecture combining:
- **Attention layers** (from nanochat) for precise pattern matching
- **Mamba SSM layers** for efficient long-range dependencies
- **Matryoshka embeddings** for adaptive compute

## 🚧 Status: In Development

This is an **intermediate checkpoint** from Phase 1 training:
- ✅ Base attention/MLP weights loaded from nanochat-d32
- ✅ Mamba layers partially trained (Phase 1)
- ⏳ Full training with Matryoshka (Phase 2) pending

## Architecture

```
┌─────────────────────────────────────┐
│          Adamba HybridGPT           │
├─────────────────────────────────────┤
│  64 interleaved blocks:             │
│    - Even: Attention + MLP          │
│    - Odd:  Mamba SSM + MLP          │
│  + Matryoshka dim levels            │
│  + Per-layer probes for early exit  │
└─────────────────────────────────────┘
```

## Parameters

- **Total**: ~5B parameters
- **Vocab size**: 65,536
- **Embedding dim**: 2048
- **Layers**: 64 (32 attention + 32 mamba)

## Usage

```python
# Coming soon - inference code
```

## Training

See: [github.com/unixsysdev/adamba](https://github.com/unixsysdev/adamba)

## License

Apache 2.0
"""
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
