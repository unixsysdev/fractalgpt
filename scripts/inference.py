#!/usr/bin/env python3
"""
Adamba inference script - works on CPU, CUDA, and ROCm.

Usage:
    python scripts/inference.py \
        --checkpoint path/to/model.pt \
        --prompt "Hello, I am Adamba"
        
For AMD Strix Halo with ROCm:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/inference.py ...
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def load_model(checkpoint_path: Path, device: str = "auto"):
    """Load Adamba model from checkpoint."""
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from nanochat.hybrid_gpt import HybridGPT, HybridConfig
    from nanochat.tokenizer import get_tokenizer
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, 'hip') or torch.cuda.is_available():
            # ROCm shows up as cuda
            device = "cuda"
        else:
            device = "cpu"
    
    print(f"🔧 Using device: {device}")
    
    # Load checkpoint to get config
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    # Infer config from state dict
    vocab_size, n_embd = state_dict["transformer.wte.weight"].shape
    
    # Count layers
    n_layer = 0
    for key in state_dict.keys():
        if key.startswith("blocks."):
            layer_num = int(key.split(".")[1])
            n_layer = max(n_layer, layer_num + 1)
    
    # Infer Mamba dt_rank from checkpoint (look at dt_proj.weight shape)
    dt_rank = 128  # default
    for key, tensor in state_dict.items():
        if "mamba.mamba.dt_proj.weight" in key:
            dt_rank = tensor.shape[1]  # [d_inner, dt_rank]
            break
    
    # Infer d_state from x_proj shape
    d_state = 128  # default
    for key, tensor in state_dict.items():
        if "mamba.mamba.x_proj.weight" in key:
            # x_proj.weight shape: [dt_rank + d_state*2, d_inner]
            # So d_state*2 = shape[0] - dt_rank
            d_state = (tensor.shape[0] - dt_rank) // 2
            break
    
    print(f"  Config: vocab={vocab_size}, n_embd={n_embd}, n_layer={n_layer}")
    
    # Create config
    config = HybridConfig(
        vocab_size=vocab_size,
        n_layer=n_layer // 2,  # Attention layers
        n_mamba_layer=n_layer // 2,  # Mamba layers
        n_embd=n_embd,
        n_embd_expanded=n_embd,
        n_head=n_embd // 128,  # Standard head dim of 128
        sequence_len=2048,
    )
    
    # Create model
    print("🏗️  Creating model...")
    with torch.device("meta"):
        model = HybridGPT(config)
    
    model = model.to_empty(device="cpu")
    
    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  ⚠️ Missing keys: {len(missing)}")
    if unexpected:
        print(f"  ⚠️ Unexpected keys: {len(unexpected)}")
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    return model, tokenizer, device


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: str = "cuda",
):
    """Generate text from prompt."""
    # Encode prompt
    tokens = tokenizer.encode(prompt, bos=True, eos=False)
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)
    
    print(f"📝 Prompt: {prompt}")
    print(f"🔢 Tokens: {tokens.shape[1]}")
    print()
    print("=" * 50)
    print(prompt, end="", flush=True)
    
    # Generate
    for _ in range(max_new_tokens):
        # Forward pass
        with torch.amp.autocast(device_type=device if device != "cpu" else "cpu", dtype=torch.bfloat16):
            logits = model(tokens)
        
        # Get next token logits
        next_logits = logits[0, -1, :] / temperature
        
        # Top-p sampling
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        
        # Remove tokens with cumulative probability above top_p
        remove_indices = cumsum > top_p
        remove_indices[1:] = remove_indices[:-1].clone()
        remove_indices[0] = False
        sorted_logits[remove_indices] = float('-inf')
        
        # Sample
        probs = F.softmax(sorted_logits, dim=-1)
        next_token_idx = torch.multinomial(probs, num_samples=1)
        next_token = sorted_indices[next_token_idx]
        
        # Append to sequence
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        
        # Decode and print
        token_str = tokenizer.decode([next_token.item()])
        print(token_str, end="", flush=True)
        
        # Stop on EOS
        if next_token.item() == tokenizer.eos_id:
            break
    
    print()
    print("=" * 50)
    
    # Return full generated text
    return tokenizer.decode(tokens[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Adamba inference")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--prompt", type=str, default="Hello, I am Adamba", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling threshold")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device")
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_model(args.checkpoint, args.device)
    
    # Generate
    output = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )
    
    print(f"\n✅ Generated {len(output)} characters")


if __name__ == "__main__":
    main()
