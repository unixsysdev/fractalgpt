"""
Surgery script with progressive expansion support.

Supports:
1. Initial surgery: nanochat-d32 → 6B hybrid
2. Progressive expansion: 6B → 10B → 20B

Usage:
    # Stage 1: Initial 6B
    python -m scripts.surgery --new-dim=2560
    
    # Stage 2: Expand to 10B
    python -m scripts.surgery --expand-from=2560 --new-dim=3072
    
    # Stage 3: Expand to 20B  
    python -m scripts.surgery --expand-from=3072 --new-dim=4096
"""

import argparse
import os
import json
import gc
from pathlib import Path
from typing import Dict, Optional

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


def expand_weight_lora(
    weight: torch.Tensor, 
    target_size: int, 
    dim: int = 0, 
    rank: int = 16, 
    std: float = 0.01
) -> torch.Tensor:
    """
    Expand weight tensor using LoRA-style low-rank initialization.
    
    Instead of zero-padding, the new dimensions are initialized as A @ B
    where A and B are small random matrices. This:
    - Preserves gradient flow (not dead zeros)
    - Has structure (low-rank, not random noise)
    - Starts near-zero (small std)
    """
    current_size = weight.size(dim)
    if current_size >= target_size:
        return weight
    
    expansion_size = target_size - current_size
    other_dim = 1 - dim if weight.dim() == 2 else 0
    other_size = weight.size(other_dim) if weight.dim() > 1 else 1
    
    # Create low-rank expansion: A @ B
    if weight.dim() == 2:
        if dim == 0:  # Expanding output dimension
            # A: (expansion_size, rank), B: (rank, input_size)
            A = torch.randn(expansion_size, rank, dtype=weight.dtype, device=weight.device) * std
            B = torch.randn(rank, weight.size(1), dtype=weight.dtype, device=weight.device)
            expansion = A @ B  # (expansion_size, input_size)
        else:  # dim == 1, expanding input dimension
            # A: (output_size, rank), B: (rank, expansion_size)
            A = torch.randn(weight.size(0), rank, dtype=weight.dtype, device=weight.device)
            B = torch.randn(rank, expansion_size, dtype=weight.dtype, device=weight.device) * std
            expansion = A @ B  # (output_size, expansion_size)
    else:
        # 1D tensor - just use small random
        expansion = torch.randn(expansion_size, dtype=weight.dtype, device=weight.device) * std
    
    return torch.cat([weight, expansion], dim=dim)


def expand_attention_interleaved(
    weight: torch.Tensor,
    old_dim: int,
    new_dim: int,
    n_head: int,
    expand_input: bool = True,
    expand_output: bool = True,
    rank: int = 16,
    std: float = 0.01,
) -> torch.Tensor:
    """
    Expand attention weight with INTERLEAVED head dimensions.
    
    MHA weights are stored as [Head1 | Head2 | ... | HeadN].
    Naive concatenation at the end scrambles heads.
    This function expands each head's dims separately.
    
    Args:
        weight: (out_dim, in_dim) attention weight
        old_dim: original model dimension
        new_dim: target model dimension
        n_head: number of attention heads
        expand_input: whether to expand input dimension
        expand_output: whether to expand output dimension (for Q/K/V)
    """
    old_head_dim = old_dim // n_head
    new_head_dim = new_dim // n_head
    head_expansion = new_head_dim - old_head_dim
    
    if head_expansion <= 0:
        return weight
    
    out_size, in_size = weight.shape
    
    # First expand input dimension (simple - not interleaved)
    if expand_input and in_size == old_dim:
        weight = expand_weight_lora(weight, new_dim, dim=1, rank=rank, std=std)
    
    # Expand output dimension with interleaving
    if expand_output and out_size == old_dim:
        # Reshape to (n_head, head_dim, input_dim)
        w_view = weight.view(n_head, old_head_dim, -1)
        
        # Create expansion for each head using LoRA
        expansions = []
        for h in range(n_head):
            # LoRA: A @ B for this head's expansion
            A = torch.randn(head_expansion, rank, dtype=weight.dtype, device=weight.device) * std
            B = torch.randn(rank, weight.size(1), dtype=weight.dtype, device=weight.device)
            head_ext = A @ B  # (head_expansion, input_dim)
            expansions.append(head_ext)
        
        expansions = torch.stack(expansions)  # (n_head, head_expansion, input_dim)
        
        # Concatenate along head_dim axis
        w_new = torch.cat([w_view, expansions], dim=1)  # (n_head, new_head_dim, input_dim)
        
        # Flatten back to 2D
        weight = w_new.reshape(-1, weight.size(1))
    
    return weight


def expand_state_dict(
    src_state: Dict[str, torch.Tensor],
    old_dim: int,
    new_dim: int,
    n_head: int = 16,  # Add n_head parameter for interleaved expansion
) -> Dict[str, torch.Tensor]:
    """
    Expand an existing hybrid state dict to larger dimensions.
    
    Uses LoRA-style initialization and interleaved head expansion for attention.
    """
    dst_state = {}
    
    print(f"Expanding from {old_dim} → {new_dim}")
    print(f"  Using LoRA-style initialization (rank=16, std=0.01)")
    print(f"  Using interleaved head expansion (n_head={n_head})")
    
    for key, tensor in src_state.items():
        new_tensor = tensor
        
        # Embedding: (vocab, dim) - use LoRA
        if 'wte.weight' in key or 'value_embeds' in key:
            if tensor.size(-1) == old_dim:
                new_tensor = expand_weight_lora(tensor, new_dim, dim=-1)
        
        # LM head: (vocab, dim) - use LoRA
        elif 'lm_head.weight' in key:
            if tensor.size(-1) == old_dim:
                new_tensor = expand_weight_lora(tensor, new_dim, dim=-1)
        
        # Attention Q/K/V: Use INTERLEAVED expansion
        elif any(x in key for x in ['c_q.weight', 'c_k.weight', 'c_v.weight']):
            new_tensor = expand_attention_interleaved(
                tensor, old_dim, new_dim, n_head,
                expand_input=True, expand_output=True
            )
        
        # Attention proj: (dim, dim) - use interleaved for input (heads), LoRA for output
        elif 'c_proj.weight' in key and 'attn' in key:
            # Input has n_head * head_dim structure
            new_tensor = expand_attention_interleaved(
                tensor, old_dim, new_dim, n_head,
                expand_input=False,  # Input is already head-structured
                expand_output=True
            )
            # Also expand the input dimension 
            new_tensor = expand_weight_lora(new_tensor, new_dim, dim=1)
        
        # MLP: c_fc (4*dim, dim), c_proj (dim, 4*dim) - use LoRA
        elif 'mlp.c_fc.weight' in key:
            new_tensor = expand_weight_lora(tensor, new_dim, dim=1)  # input
            new_tensor = expand_weight_lora(new_tensor, 4 * new_dim, dim=0)  # output
        elif 'mlp.c_proj.weight' in key:
            new_tensor = expand_weight_lora(tensor, 4 * new_dim, dim=1)  # input
            new_tensor = expand_weight_lora(new_tensor, new_dim, dim=0)  # output
        
        # Mamba: in_proj (2*d_inner, dim), out_proj (dim, d_inner)
        elif 'mamba.layer.in_proj.weight' in key:
            d_inner_old = tensor.size(0) // 2
            d_inner_new = new_dim * 2  # expand=2
            new_tensor = expand_weight(tensor, new_dim, dim=1)
            new_tensor = expand_weight(new_tensor, d_inner_new * 2, dim=0)
        elif 'mamba.layer.out_proj.weight' in key:
            d_inner_new = new_dim * 2
            new_tensor = expand_weight(tensor, d_inner_new, dim=1)
            new_tensor = expand_weight(new_tensor, new_dim, dim=0)
        
        # Mamba conv1d, dt_proj, etc - expand d_inner
        elif 'mamba.layer.conv1d' in key:
            d_inner_new = new_dim * 2
            new_tensor = expand_weight(tensor, d_inner_new, dim=0)
        elif 'mamba.layer.dt_proj' in key:
            d_inner_new = new_dim * 2
            if 'weight' in key:
                new_tensor = expand_weight(tensor, d_inner_new, dim=0)
                new_tensor = expand_weight(new_tensor, d_inner_new, dim=1)
            else:
                new_tensor = expand_weight(tensor, d_inner_new, dim=0)
        elif 'mamba.layer.A_log' in key or 'mamba.layer.D' in key:
            d_inner_new = new_dim * 2
            new_tensor = expand_weight(tensor, d_inner_new, dim=0)
        
        # Probes and think_detector stay same size (small)
        # resid_lambdas, x0_lambdas stay same (per-layer)
        
        dst_state[key] = new_tensor
        
        if new_tensor.shape != tensor.shape:
            print(f"  {key}: {tensor.shape} → {new_tensor.shape}")
    
    gc.collect()
    return dst_state


def build_hybrid_state_dict(
    src_state: Dict[str, torch.Tensor],
    src_config: GPTConfig,
    old_dim: int = 2048,
    new_dim: int = 2560,
) -> Dict[str, torch.Tensor]:
    """Build hybrid state dict from nanochat source (initial surgery)."""
    dst_state = {}
    
    n_layer = src_config.n_layer
    n_mamba = n_layer
    n_head = src_config.n_head
    n_kv_head = src_config.n_kv_head
    old_head_dim = old_dim // n_head
    new_head_dim = new_dim // n_head
    vocab_size = 32768
    padded_vocab = 32768
    
    print(f"Building hybrid state dict: {n_layer} attn + {n_mamba} mamba layers")
    print(f"Dimension expansion: {old_dim} → {new_dim}")
    
    # Embedding
    print("  Expanding embedding...")
    wte = src_state['transformer.wte.weight']
    dst_state['transformer.wte.weight'] = expand_weight(wte, new_dim, dim=1)
    del wte; gc.collect()
    
    # LM head
    print("  Expanding lm_head...")
    lm_head = src_state['lm_head.weight']
    dst_state['lm_head.weight'] = expand_weight(lm_head, new_dim, dim=1)
    del lm_head; gc.collect()
    
    # Per-layer scalars
    total_layers = n_layer + n_mamba
    dst_state['resid_lambdas'] = torch.ones(total_layers)
    dst_state['x0_lambdas'] = torch.zeros(total_layers)
    
    src_resid = src_state.get('resid_lambdas', torch.ones(n_layer))
    src_x0 = src_state.get('x0_lambdas', torch.zeros(n_layer))
    
    attn_idx = 0
    for i in range(total_layers):
        if i % 2 == 0 and attn_idx < n_layer:
            dst_state['resid_lambdas'][i] = src_resid[attn_idx]
            dst_state['x0_lambdas'][i] = src_x0[attn_idx]
            attn_idx += 1
    
    # Process blocks
    src_block_idx = 0
    d_inner = new_dim * 2  # Mamba expand=2
    d_state = 16
    d_conv = 4
    std = (3 ** 0.5) * (new_dim ** -0.5)
    
    for block_idx in range(total_layers):
        is_attention = (block_idx % 2 == 0) and (src_block_idx < n_layer)
        prefix = f'blocks.{block_idx}'
        
        if is_attention:
            print(f"  Block {block_idx}: Attention (from src {src_block_idx})")
            src_prefix = f'transformer.h.{src_block_idx}'
            
            # Q, K, V
            for name, out_mult in [('c_q', n_head), ('c_k', n_kv_head), ('c_v', n_kv_head)]:
                src_key = f'{src_prefix}.attn.{name}.weight'
                if src_key in src_state:
                    w = src_state[src_key]
                    w = expand_weight(w, new_dim, dim=1)
                    w = expand_weight(w, out_mult * new_head_dim, dim=0)
                    dst_state[f'{prefix}.attn.{name}.weight'] = w
            
            # c_proj
            src_key = f'{src_prefix}.attn.c_proj.weight'
            if src_key in src_state:
                w = src_state[src_key]
                w = expand_weight(w, n_head * new_head_dim, dim=1)
                w = expand_weight(w, new_dim, dim=0)
                dst_state[f'{prefix}.attn.c_proj.weight'] = w
            
            # VE gate
            src_key = f'{src_prefix}.attn.ve_gate.weight'
            if src_key in src_state:
                dst_state[f'{prefix}.attn.ve_gate.weight'] = src_state[src_key].clone()
            
            # MLP
            src_key = f'{src_prefix}.mlp.c_fc.weight'
            if src_key in src_state:
                w = src_state[src_key]
                w = expand_weight(w, new_dim, dim=1)
                w = expand_weight(w, 4 * new_dim, dim=0)
                dst_state[f'{prefix}.mlp.c_fc.weight'] = w
            
            src_key = f'{src_prefix}.mlp.c_proj.weight'
            if src_key in src_state:
                w = src_state[src_key]
                w = expand_weight(w, 4 * new_dim, dim=1)
                w = expand_weight(w, new_dim, dim=0)
                dst_state[f'{prefix}.mlp.c_proj.weight'] = w
            
            src_block_idx += 1
        else:
            print(f"  Block {block_idx}: Mamba (fresh init)")
            
            # Mamba projections
            dst_state[f'{prefix}.mamba.layer.in_proj.weight'] = torch.randn(d_inner * 2, new_dim) * std
            dst_state[f'{prefix}.mamba.layer.out_proj.weight'] = torch.zeros(new_dim, d_inner)
            dst_state[f'{prefix}.mamba.layer.conv1d.weight'] = torch.randn(d_inner, 1, d_conv) * std
            dst_state[f'{prefix}.mamba.layer.conv1d.bias'] = torch.zeros(d_inner)
            dst_state[f'{prefix}.mamba.layer.dt_proj.weight'] = torch.randn(d_inner, d_inner) * std
            dst_state[f'{prefix}.mamba.layer.dt_proj.bias'] = torch.zeros(d_inner)
            dst_state[f'{prefix}.mamba.layer.A_log'] = torch.zeros(d_inner, d_state)
            dst_state[f'{prefix}.mamba.layer.D'] = torch.ones(d_inner)
            
            # MLP for Mamba block
            dst_state[f'{prefix}.mlp.c_fc.weight'] = torch.randn(4 * new_dim, new_dim) * std
            dst_state[f'{prefix}.mlp.c_proj.weight'] = torch.zeros(new_dim, 4 * new_dim)
        
        # Probe
        dst_state[f'{prefix}.probe.probe.0.weight'] = torch.randn(32, 3) * 0.1
        dst_state[f'{prefix}.probe.probe.0.bias'] = torch.zeros(32)
        dst_state[f'{prefix}.probe.probe.2.weight'] = torch.randn(2, 32) * 0.1
        dst_state[f'{prefix}.probe.probe.2.bias'] = torch.zeros(2)
        
        gc.collect()
    
    # Value embeddings
    kv_dim = n_kv_head * new_head_dim
    for i in range(total_layers):
        if i % 2 == 0:
            dst_state[f'value_embeds.{i}.weight'] = torch.randn(padded_vocab, kv_dim) * 0.01
    
    # Think detector
    dst_state['think_detector.net.0.weight'] = torch.randn(32, 1) * 0.1
    dst_state['think_detector.net.0.bias'] = torch.zeros(32)
    dst_state['think_detector.net.2.weight'] = torch.randn(1, 32) * 0.1
    dst_state['think_detector.net.2.bias'] = torch.zeros(1)
    
    return dst_state


def surgery(
    src_checkpoint: Path,
    dst_checkpoint: Path,
    old_dim: int = 2048,
    new_dim: int = 2560,
    expand_from: Optional[int] = None,
):
    """
    Perform architecture surgery.
    
    If expand_from is set, expands an existing hybrid checkpoint.
    Otherwise, creates hybrid from nanochat source.
    """
    print(f"Loading source checkpoint: {src_checkpoint}")
    src_state = torch.load(src_checkpoint, map_location='cpu', weights_only=True)
    print(f"Source state dict loaded: {len(src_state)} keys")
    
    if expand_from:
        # Progressive expansion: hybrid → larger hybrid
        print(f"\n=== Progressive Expansion: {expand_from} → {new_dim} ===")
        dst_state = expand_state_dict(src_state, expand_from, new_dim)
    else:
        # Initial surgery: nanochat → hybrid
        print(f"\n=== Initial Surgery: nanochat → hybrid@{new_dim} ===")
        
        # Infer config
        wte_shape = src_state['transformer.wte.weight'].shape
        vocab_size = wte_shape[0]
        n_embd = wte_shape[1]
        
        n_layer = 0
        while f'transformer.h.{n_layer}.attn.c_q.weight' in src_state:
            n_layer += 1
        
        q_shape = src_state['transformer.h.0.attn.c_q.weight'].shape
        head_dim = old_dim // 16
        n_head = q_shape[0] // head_dim
        
        src_config = GPTConfig(
            sequence_len=2048,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_kv_head=n_head,
            n_embd=n_embd,
        )
        
        print(f"Source config: n_layer={n_layer}, n_embd={n_embd}, n_head={n_head}")
        dst_state = build_hybrid_state_dict(src_state, src_config, old_dim, new_dim)
    
    del src_state
    gc.collect()
    
    # Save
    print(f"\nSaving to {dst_checkpoint}")
    dst_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dst_state, dst_checkpoint)
    
    # Save/update config
    config_path = dst_checkpoint.parent / "config.json"
    config = {
        'sequence_len': 32768,
        'vocab_size': 32768,
        'n_layer': 32,
        'n_mamba_layer': 32,
        'n_head': 16,
        'n_kv_head': 16,
        'n_embd': 2048,
        'n_embd_expanded': new_dim,
        'stage': f'{new_dim}',
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    total_params = sum(t.numel() for t in dst_state.values())
    print(f"\nSurgery complete!")
    print(f"  Keys: {len(dst_state)}")
    print(f"  Total params: {total_params:,}")
    print(f"  Approx size: {total_params / 1e9:.1f}B")


def main():
    parser = argparse.ArgumentParser(description="Convert/expand to HybridGPT")
    parser.add_argument(
        '--src', type=Path,
        default=None,
        help='Path to source checkpoint (auto-detects nanochat vs hybrid)',
    )
    parser.add_argument(
        '--dst', type=Path,
        default=None,
        help='Path to save checkpoint',
    )
    parser.add_argument('--old-dim', type=int, default=2048,
                        help='Original nanochat dimension')
    parser.add_argument('--new-dim', type=int, default=2560,
                        help='Target dimension')
    parser.add_argument('--expand-from', type=int, default=None,
                        help='If set, expand existing hybrid from this dim')
    
    args = parser.parse_args()
    
    # Auto paths
    cache = Path.home() / '.cache/nanochat'
    if args.expand_from:
        src = args.src or cache / 'hybrid_checkpoints' / f'd32_{args.expand_from}' / 'model.pt'
    else:
        src = args.src or cache / 'chatsft_checkpoints/d32/model_000650.pt'
    
    dst = args.dst or cache / 'hybrid_checkpoints' / f'd32_{args.new_dim}' / 'model.pt'
    
    surgery(src, dst, args.old_dim, args.new_dim, args.expand_from)


if __name__ == '__main__':
    main()
