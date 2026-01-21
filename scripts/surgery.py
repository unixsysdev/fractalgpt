"""
Surgery script to convert nanochat-d32 to HybridGPT.

This script:
1. Loads the pretrained nanochat-d32 checkpoint (as state_dict, not model)
2. Creates expanded weight tensors layer-by-layer
3. Saves the hybrid state dict directly (memory efficient)
"""

import argparse
import os
import json
import gc
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from nanochat.gpt import GPTConfig


def expand_weight(weight: torch.Tensor, target_size: int, dim: int = 0) -> torch.Tensor:
    """Expand a weight tensor by zero-padding along a dimension."""
    current_size = weight.size(dim)
    if current_size >= target_size:
        return weight
    
    pad_size = target_size - current_size
    pad_shape = list(weight.shape)
    pad_shape[dim] = pad_size
    padding = torch.zeros(pad_shape, dtype=weight.dtype, device=weight.device)
    
    return torch.cat([weight, padding], dim=dim)


def build_hybrid_state_dict(
    src_state: Dict[str, torch.Tensor],
    src_config: GPTConfig,
    old_dim: int = 2048,
    new_dim: int = 4096,
) -> Dict[str, torch.Tensor]:
    """
    Build hybrid state dict from source state dict.
    
    Memory efficient: works with tensors, not full models.
    """
    dst_state = {}
    
    n_layer = src_config.n_layer
    n_mamba = n_layer  # Same number of mamba layers
    n_head = src_config.n_head
    n_kv_head = src_config.n_kv_head
    old_head_dim = old_dim // n_head
    new_head_dim = new_dim // n_head
    vocab_size = 32768
    padded_vocab = 32768
    
    print(f"Building hybrid state dict: {n_layer} attn + {n_mamba} mamba layers")
    
    # Embedding: expand dim
    print("  Expanding embedding...")
    wte = src_state['transformer.wte.weight']
    dst_state['transformer.wte.weight'] = expand_weight(wte, new_dim, dim=1)
    del wte; gc.collect()
    
    # LM head: expand input dim
    print("  Expanding lm_head...")
    lm_head = src_state['lm_head.weight']
    dst_state['lm_head.weight'] = expand_weight(lm_head, new_dim, dim=1)
    del lm_head; gc.collect()
    
    # Per-layer scalars: expand to n_layer + n_mamba
    total_layers = n_layer + n_mamba
    dst_state['resid_lambdas'] = torch.ones(total_layers)
    dst_state['x0_lambdas'] = torch.zeros(total_layers)
    
    # Copy scalars from source where applicable
    src_resid = src_state.get('resid_lambdas', torch.ones(n_layer))
    src_x0 = src_state.get('x0_lambdas', torch.zeros(n_layer))
    
    # Interleave: attention at even indices
    attn_idx = 0
    for i in range(total_layers):
        if i % 2 == 0 and attn_idx < n_layer:
            dst_state['resid_lambdas'][i] = src_resid[attn_idx]
            dst_state['x0_lambdas'][i] = src_x0[attn_idx]
            attn_idx += 1
    
    # Process blocks
    src_block_idx = 0
    for block_idx in range(total_layers):
        is_attention = (block_idx % 2 == 0) and (src_block_idx < n_layer)
        prefix = f'blocks.{block_idx}'
        
        if is_attention:
            print(f"  Block {block_idx}: Attention (from src {src_block_idx})")
            src_prefix = f'transformer.h.{src_block_idx}'
            
            # Attention Q, K, V, proj
            for name, out_mult in [('c_q', n_head), ('c_k', n_kv_head), ('c_v', n_kv_head)]:
                src_key = f'{src_prefix}.attn.{name}.weight'
                if src_key in src_state:
                    w = src_state[src_key]
                    # Expand in and out dimensions
                    w = expand_weight(w, new_dim, dim=1)  # input
                    w = expand_weight(w, out_mult * new_head_dim, dim=0)  # output
                    dst_state[f'{prefix}.attn.{name}.weight'] = w
            
            # c_proj: (n_embd, n_head * head_dim) 
            src_key = f'{src_prefix}.attn.c_proj.weight'
            if src_key in src_state:
                w = src_state[src_key]
                w = expand_weight(w, n_head * new_head_dim, dim=1)  # input
                w = expand_weight(w, new_dim, dim=0)  # output
                dst_state[f'{prefix}.attn.c_proj.weight'] = w
            
            # VE gate (if exists) - small, just copy
            src_key = f'{src_prefix}.attn.ve_gate.weight'
            if src_key in src_state:
                dst_state[f'{prefix}.attn.ve_gate.weight'] = src_state[src_key].clone()
            
            # MLP: c_fc (4*n_embd, n_embd) and c_proj (n_embd, 4*n_embd)
            src_key = f'{src_prefix}.mlp.c_fc.weight'
            if src_key in src_state:
                w = src_state[src_key]
                w = expand_weight(w, new_dim, dim=1)  # input
                w = expand_weight(w, 4 * new_dim, dim=0)  # output
                dst_state[f'{prefix}.mlp.c_fc.weight'] = w
            
            src_key = f'{src_prefix}.mlp.c_proj.weight'
            if src_key in src_state:
                w = src_state[src_key]
                w = expand_weight(w, 4 * new_dim, dim=1)  # input
                w = expand_weight(w, new_dim, dim=0)  # output
                dst_state[f'{prefix}.mlp.c_proj.weight'] = w
            
            src_block_idx += 1
        else:
            print(f"  Block {block_idx}: Mamba (fresh init)")
            # Mamba blocks: initialize fresh (small std)
            d_inner = new_dim * 2  # expand=2
            d_state = 16
            d_conv = 4
            
            std = (3 ** 0.5) * (new_dim ** -0.5)
            
            # Mamba projections
            dst_state[f'{prefix}.mamba.layer.in_proj.weight'] = torch.randn(d_inner * 2, new_dim) * std
            dst_state[f'{prefix}.mamba.layer.out_proj.weight'] = torch.zeros(new_dim, d_inner)
            dst_state[f'{prefix}.mamba.layer.conv1d.weight'] = torch.randn(d_inner, 1, d_conv) * std
            dst_state[f'{prefix}.mamba.layer.conv1d.bias'] = torch.zeros(d_inner)
            
            # SSM params
            dst_state[f'{prefix}.mamba.layer.dt_proj.weight'] = torch.randn(d_inner, d_inner) * std
            dst_state[f'{prefix}.mamba.layer.dt_proj.bias'] = torch.zeros(d_inner)
            dst_state[f'{prefix}.mamba.layer.A_log'] = torch.zeros(d_inner, d_state)
            dst_state[f'{prefix}.mamba.layer.D'] = torch.ones(d_inner)
            
            # MLP for Mamba block
            dst_state[f'{prefix}.mlp.c_fc.weight'] = torch.randn(4 * new_dim, new_dim) * std
            dst_state[f'{prefix}.mlp.c_proj.weight'] = torch.zeros(new_dim, 4 * new_dim)
        
        # Probe (fresh init for all blocks)
        dst_state[f'{prefix}.probe.probe.0.weight'] = torch.randn(32, 3) * 0.1
        dst_state[f'{prefix}.probe.probe.0.bias'] = torch.zeros(32)
        dst_state[f'{prefix}.probe.probe.2.weight'] = torch.randn(2, 32) * 0.1
        dst_state[f'{prefix}.probe.probe.2.bias'] = torch.zeros(2)
        
        gc.collect()
    
    # Value embeddings (skip for now, can be initialized later)
    # They're small anyway
    kv_dim = n_kv_head * new_head_dim
    for i in range(total_layers):
        if i % 2 == 0:  # attention blocks
            dst_state[f'value_embeds.{i}.weight'] = torch.randn(padded_vocab, kv_dim) * 0.01
    
    # Think detector (fresh init)
    dst_state['think_detector.net.0.weight'] = torch.randn(32, 1) * 0.1
    dst_state['think_detector.net.0.bias'] = torch.zeros(32)
    dst_state['think_detector.net.2.weight'] = torch.randn(1, 32) * 0.1
    dst_state['think_detector.net.2.bias'] = torch.zeros(1)
    
    # Rotary embeddings are computed at runtime, not stored
    
    return dst_state


def surgery(
    src_checkpoint: Path,
    dst_checkpoint: Path,
    old_dim: int = 2048,
    new_dim: int = 4096,
):
    """Perform architecture surgery (memory efficient)."""
    print(f"Loading source checkpoint: {src_checkpoint}")
    
    # Load just the state dict, not the model
    src_state = torch.load(src_checkpoint, map_location='cpu', weights_only=True)
    print(f"Source state dict loaded: {len(src_state)} keys")
    
    # Infer config from state dict
    wte_shape = src_state['transformer.wte.weight'].shape
    vocab_size = wte_shape[0]
    n_embd = wte_shape[1]
    
    # Count layers
    n_layer = 0
    while f'transformer.h.{n_layer}.attn.c_q.weight' in src_state:
        n_layer += 1
    
    # Infer heads from Q shape
    q_shape = src_state['transformer.h.0.attn.c_q.weight'].shape
    head_dim = old_dim // 16  # Assume 16 heads for d32
    n_head = q_shape[0] // head_dim
    
    src_config = GPTConfig(
        sequence_len=2048,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
    )
    
    print(f"Inferred config: n_layer={n_layer}, n_embd={n_embd}, n_head={n_head}")
    
    # Build hybrid state dict
    dst_state = build_hybrid_state_dict(src_state, src_config, old_dim, new_dim)
    
    # Free source
    del src_state
    gc.collect()
    
    # Save
    print(f"Saving to {dst_checkpoint}")
    dst_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dst_state, dst_checkpoint)
    
    # Save config
    config_path = dst_checkpoint.parent / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'sequence_len': 32768,
            'vocab_size': vocab_size,
            'n_layer': n_layer,
            'n_mamba_layer': n_layer,
            'n_head': n_head,
            'n_kv_head': n_head,
            'n_embd': old_dim,
            'n_embd_expanded': new_dim,
        }, f, indent=2)
    
    print("Surgery complete!")
    print(f"  Saved {len(dst_state)} keys")
    
    # Count params
    total_params = sum(t.numel() for t in dst_state.values())
    print(f"  Total params: {total_params:,}")


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
