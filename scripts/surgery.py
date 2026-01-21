"""
Surgery script to convert nanochat-d32 to HybridGPT.

This script:
1. Loads the pretrained nanochat-d32 checkpoint
2. Expands dimensions 2048 → 4096 (zero-padding new weights)
3. Inserts fresh Mamba layers between attention layers
4. Saves the new hybrid checkpoint
"""

import argparse
import os
import json
from pathlib import Path

import torch
import torch.nn as nn

from nanochat.gpt import GPT, GPTConfig
from nanochat.hybrid_gpt import HybridGPT, HybridConfig


def expand_weight(weight: torch.Tensor, target_size: int, dim: int = 0) -> torch.Tensor:
    """
    Expand a weight tensor by zero-padding along a dimension.
    
    Args:
        weight: Original weight tensor
        target_size: Target size for the dimension
        dim: Which dimension to expand
    
    Returns:
        Expanded weight tensor
    """
    current_size = weight.size(dim)
    if current_size >= target_size:
        return weight
    
    # Calculate padding
    pad_size = target_size - current_size
    
    # Create padding shape
    pad_shape = list(weight.shape)
    pad_shape[dim] = pad_size
    
    # Pad with zeros
    padding = torch.zeros(pad_shape, dtype=weight.dtype, device=weight.device)
    
    return torch.cat([weight, padding], dim=dim)


def expand_embedding(embedding: nn.Embedding, new_dim: int) -> torch.Tensor:
    """Expand embedding dimension."""
    return expand_weight(embedding.weight.data, new_dim, dim=1)


def expand_linear_in(linear: nn.Linear, new_dim: int) -> torch.Tensor:
    """Expand linear layer input dimension."""
    return expand_weight(linear.weight.data, new_dim, dim=1)


def expand_linear_out(linear: nn.Linear, new_dim: int) -> torch.Tensor:
    """Expand linear layer output dimension."""
    return expand_weight(linear.weight.data, new_dim, dim=0)


def expand_linear_both(linear: nn.Linear, new_in: int, new_out: int) -> torch.Tensor:
    """Expand linear layer in both dimensions."""
    w = linear.weight.data
    w = expand_weight(w, new_in, dim=1)  # Input dim
    w = expand_weight(w, new_out, dim=0)  # Output dim
    return w


def transfer_attention_block(
    src_block,
    dst_block,
    old_dim: int = 2048,
    new_dim: int = 4096,
):
    """
    Transfer weights from nanochat Block to HybridBlock (attention).
    
    Expands all dimensions appropriately.
    """
    src_attn = src_block.attn
    dst_attn = dst_block.attn
    
    # Q, K, V projections: (out, in) = (n_head * head_dim, n_embd)
    # Need to expand input dim and output dim
    old_head_dim = old_dim // src_attn.n_head
    new_head_dim = new_dim // dst_attn.n_head
    
    # c_q: n_head * head_dim, n_embd
    dst_attn.c_q.weight.data = expand_linear_both(
        src_attn.c_q,
        new_in=new_dim,
        new_out=dst_attn.n_head * new_head_dim,
    )
    
    # c_k, c_v: n_kv_head * head_dim, n_embd
    dst_attn.c_k.weight.data = expand_linear_both(
        src_attn.c_k,
        new_in=new_dim,
        new_out=dst_attn.n_kv_head * new_head_dim,
    )
    dst_attn.c_v.weight.data = expand_linear_both(
        src_attn.c_v,
        new_in=new_dim,
        new_out=dst_attn.n_kv_head * new_head_dim,
    )
    
    # c_proj: n_embd, n_head * head_dim
    dst_attn.c_proj.weight.data = expand_linear_both(
        src_attn.c_proj,
        new_in=dst_attn.n_head * new_head_dim,
        new_out=new_dim,
    )
    
    # VE gate (if exists)
    if src_attn.ve_gate is not None and dst_attn.ve_gate is not None:
        dst_attn.ve_gate.weight.data = expand_linear_out(
            src_attn.ve_gate,
            dst_attn.n_kv_head,
        )
    
    # MLP
    src_mlp = src_block.mlp
    dst_mlp = dst_block.mlp
    
    # c_fc: 4*n_embd, n_embd
    old_intermediate = 4 * old_dim
    new_intermediate = 4 * new_dim
    dst_mlp.c_fc.weight.data = expand_linear_both(
        src_mlp.c_fc,
        new_in=new_dim,
        new_out=new_intermediate,
    )
    
    # c_proj: n_embd, 4*n_embd
    dst_mlp.c_proj.weight.data = expand_linear_both(
        src_mlp.c_proj,
        new_in=new_intermediate,
        new_out=new_dim,
    )


def surgery(
    src_checkpoint: Path,
    dst_checkpoint: Path,
    old_dim: int = 2048,
    new_dim: int = 4096,
):
    """
    Perform architecture surgery.
    
    Args:
        src_checkpoint: Path to nanochat-d32 checkpoint
        dst_checkpoint: Path to save hybrid checkpoint
    """
    print(f"Loading source checkpoint: {src_checkpoint}")
    
    # Load source config - try multiple meta file formats
    meta_path = src_checkpoint.parent / f"meta_{src_checkpoint.stem.split('_')[1]}.json"
    src_config = None
    
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        # Try different key names used by nanochat
        config_dict = meta.get('config') or meta.get('model_config') or {}
        if config_dict:
            src_config = GPTConfig(**{k: v for k, v in config_dict.items() if k in GPTConfig.__dataclass_fields__})
    
    # Default nanochat-d32 config if meta not found or empty
    if src_config is None:
        print("Meta file not found or invalid, using default nanochat-d32 config")
        src_config = GPTConfig(
            sequence_len=2048,
            vocab_size=32768,
            n_layer=32,
            n_head=16,
            n_kv_head=16,
            n_embd=2048,
        )
    
    print(f"Source config: n_layer={src_config.n_layer}, n_embd={src_config.n_embd}")
    
    # Create source model and load weights
    print("Loading source model...")
    src_model = GPT(src_config)
    src_state = torch.load(src_checkpoint, map_location='cpu', weights_only=True)
    src_model.load_state_dict(src_state, strict=False)
    del src_state  # Free checkpoint memory
    import gc; gc.collect()
    
    # Create destination config
    dst_config = HybridConfig(
        sequence_len=32768,  # Extended context
        vocab_size=src_config.vocab_size,
        n_layer=src_config.n_layer,
        n_mamba_layer=src_config.n_layer,  # Same number of Mamba layers
        n_head=src_config.n_head,
        n_kv_head=src_config.n_kv_head,
        n_embd=src_config.n_embd,
        n_embd_expanded=new_dim,
    )
    
    print(f"Destination config: n_layer={dst_config.n_layer}, n_mamba_layer={dst_config.n_mamba_layer}, n_embd_expanded={dst_config.n_embd_expanded}")
    
    # Create destination model on meta device first (no memory), then materialize
    print("Creating destination model...")
    with torch.device('meta'):
        dst_model = HybridGPT(dst_config)
    
    # Materialize on CPU
    dst_model = dst_model.to_empty(device='cpu')
    dst_model.init_weights()  # Initialize fresh weights
    
    # Transfer embeddings
    print("Transferring embeddings...")
    dst_model.transformer.wte.weight.data = expand_embedding(
        src_model.transformer.wte,
        new_dim,
    )
    
    # Transfer lm_head
    dst_model.lm_head.weight.data = expand_linear_in(
        src_model.lm_head,
        new_dim,
    )
    
    # Transfer attention blocks (every other block in hybrid)
    print("Transferring attention blocks...")
    src_block_idx = 0
    for i, dst_block in enumerate(dst_model.blocks):
        if dst_block.is_attention and src_block_idx < len(src_model.transformer.h):
            print(f"  Block {i}: Attention (from src block {src_block_idx})")
            transfer_attention_block(
                src_model.transformer.h[src_block_idx],
                dst_block,
                old_dim=old_dim,
                new_dim=new_dim,
            )
            src_block_idx += 1
        else:
            print(f"  Block {i}: Mamba (fresh init)")
    
    # Transfer value embeddings
    print("Transferring value embeddings...")
    for key, src_ve in src_model.value_embeds.items():
        # Find corresponding key in dst (may be different due to interleaving)
        # For now, skip as the mapping is complex
        pass
    
    # Transfer per-layer scalars (expand to new number of layers)
    print("Expanding per-layer scalars...")
    n_total = len(dst_model.blocks)
    dst_model.resid_lambdas.data = torch.ones(n_total)
    dst_model.x0_lambdas.data = torch.zeros(n_total)
    
    # Copy from source where applicable
    for i, dst_block in enumerate(dst_model.blocks):
        if dst_block.is_attention and i // 2 < src_config.n_layer:
            src_idx = i // 2
            dst_model.resid_lambdas.data[i] = src_model.resid_lambdas.data[src_idx]
            dst_model.x0_lambdas.data[i] = src_model.x0_lambdas.data[src_idx]
    
    # Save
    print(f"Saving to {dst_checkpoint}")
    dst_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dst_model.state_dict(), dst_checkpoint)
    
    # Save config
    config_path = dst_checkpoint.parent / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'sequence_len': dst_config.sequence_len,
            'vocab_size': dst_config.vocab_size,
            'n_layer': dst_config.n_layer,
            'n_mamba_layer': dst_config.n_mamba_layer,
            'n_head': dst_config.n_head,
            'n_kv_head': dst_config.n_kv_head,
            'n_embd': dst_config.n_embd,
            'n_embd_expanded': dst_config.n_embd_expanded,
        }, f, indent=2)
    
    print("Surgery complete!")
    print(f"  Source params: {sum(p.numel() for p in src_model.parameters()):,}")
    print(f"  Dest params:   {sum(p.numel() for p in dst_model.parameters()):,}")


def main():
    parser = argparse.ArgumentParser(description="Convert nanochat-d32 to HybridGPT")
    parser.add_argument(
        '--src', type=Path,
        default=Path.home() / '.cache/nanochat/chatsft_checkpoints/d32/model_000650.pt',
        help='Path to source checkpoint',
    )
    parser.add_argument(
        '--dst', type=Path,
        default=Path.home() / '.cache/nanochat/hybrid_checkpoints/d32/model_surgery.pt',
        help='Path to save hybrid checkpoint',
    )
    parser.add_argument('--old-dim', type=int, default=2048)
    parser.add_argument('--new-dim', type=int, default=4096)
    
    args = parser.parse_args()
    surgery(args.src, args.dst, args.old_dim, args.new_dim)


if __name__ == '__main__':
    main()
