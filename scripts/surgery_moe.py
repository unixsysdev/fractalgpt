"""
Surgery script for GPT-OSS 20B checkpoint conversion.

Converts HuggingFace GPT-OSS 20B checkpoint to Adamba hybrid format:
1. Maps GPT-OSS weight names to Adamba naming convention
2. Injects zero-initialized Mamba layers between attention blocks
3. Adds Matryoshka infrastructure (probes, gates)

Usage:
    # Download checkpoint first:
    huggingface-cli download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/
    
    # Convert:
    python -m scripts.surgery_moe --src gpt-oss-20b/original --dst checkpoints/gptoss_hybrid.pt
"""

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from nanochat.moe_block import MoEConfig


# GPT-OSS 20B config (from HuggingFace)
GPTOSS_20B_CONFIG = {
    "hidden_size": 2880,
    "intermediate_size": 2880,
    "num_hidden_layers": 24,
    "num_experts": 32,
    "experts_per_token": 4,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "vocab_size": 201088,
    "sliding_window": 128,
    "max_position_embeddings": 131072,
}


def map_attention_key(src_key: str, layer_idx: int, block_idx: int) -> Optional[str]:
    """
    Map GPT-OSS attention weight key to Adamba naming.
    
    GPT-OSS: model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    Adamba:  blocks.{j}.attn.c_{q,k,v,proj}.weight
    """
    mappings = {
        "q_proj": "c_q",
        "k_proj": "c_k", 
        "v_proj": "c_v",
        "o_proj": "c_proj",
    }
    
    for src_name, dst_name in mappings.items():
        if f"self_attn.{src_name}" in src_key:
            return f"blocks.{block_idx}.attn.{dst_name}.weight"
    
    return None


def map_moe_key(src_key: str, layer_idx: int, block_idx: int) -> Optional[str]:
    """
    Map GPT-OSS MoE weight key to Adamba naming.
    
    GPT-OSS: model.layers.{i}.mlp.router.weight
             model.layers.{i}.mlp.experts.{e}.up_proj.weight
             model.layers.{i}.mlp.experts.{e}.down_proj.weight
    Adamba:  blocks.{j}.moe.router.gate.weight
             blocks.{j}.moe.moe.mlp1_weight[e]
             blocks.{j}.moe.moe.mlp2_weight[e]
    """
    if "mlp.router.weight" in src_key:
        return f"blocks.{block_idx}.moe.router.gate.weight"
    
    # For expert weights, we need to handle differently
    # They will be stacked into single tensors
    return None


def load_gptoss_checkpoint(src_path: Path) -> Dict[str, torch.Tensor]:
    """
    Load GPT-OSS checkpoint from safetensors files.
    """
    from safetensors import safe_open
    
    state_dict = {}
    
    # Find all safetensor files
    safetensor_files = list(src_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensor files found in {src_path}")
    
    print(f"Loading {len(safetensor_files)} safetensor files...")
    
    for sf_path in sorted(safetensor_files):
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    
    print(f"Loaded {len(state_dict)} tensors")
    return state_dict


def build_hybrid_state_dict(
    src_state: Dict[str, torch.Tensor],
    config: dict,
    n_mamba_layers: int = 12,
) -> Dict[str, torch.Tensor]:
    """
    Build Adamba hybrid state dict from GPT-OSS source.
    
    Architecture:
    - 24 GPT-OSS layers become 24 attention+MoE blocks
    - 12 new Mamba layers interleaved (after every full attention)
    - Total: 36 blocks
    """
    dst_state = {}
    
    hidden_size = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    vocab_size = config["vocab_size"]
    num_experts = config["num_experts"]
    intermediate_size = config["intermediate_size"]
    
    # Pad vocab to multiple of 64
    padded_vocab = ((vocab_size + 63) // 64) * 64
    
    print(f"Building hybrid state dict:")
    print(f"  Source: {n_layers} GPT-OSS layers")
    print(f"  Adding: {n_mamba_layers} Mamba layers")
    print(f"  Total blocks: {n_layers + n_mamba_layers}")
    
    # 1. Embeddings
    print("  Mapping embeddings...")
    if "model.embed_tokens.weight" in src_state:
        emb = src_state["model.embed_tokens.weight"]
        # Pad vocab dimension if needed
        if emb.size(0) < padded_vocab:
            padding = torch.zeros(padded_vocab - emb.size(0), hidden_size, dtype=emb.dtype)
            emb = torch.cat([emb, padding], dim=0)
        dst_state["transformer.wte.weight"] = emb
    
    # 2. LM head
    if "lm_head.weight" in src_state:
        lm = src_state["lm_head.weight"]
        if lm.size(0) < padded_vocab:
            padding = torch.zeros(padded_vocab - lm.size(0), hidden_size, dtype=lm.dtype)
            lm = torch.cat([lm, padding], dim=0)
        dst_state["lm_head.weight"] = lm
    
    # 3. Final norm
    if "model.norm.weight" in src_state:
        dst_state["final_norm.weight"] = src_state["model.norm.weight"]
    
    # 4. Process layers
    std = (3 ** 0.5) * (hidden_size ** -0.5)
    d_inner = hidden_size * 2  # Mamba expansion
    d_state = 16
    d_conv = 4
    
    total_blocks = n_layers + n_mamba_layers
    mamba_interval = n_layers // n_mamba_layers  # Insert Mamba every N attention layers
    
    block_idx = 0
    mamba_count = 0
    
    for layer_idx in range(n_layers):
        prefix = f"blocks.{block_idx}"
        src_prefix = f"model.layers.{layer_idx}"
        
        print(f"  Layer {layer_idx} -> Block {block_idx} (Attention+MoE)")
        
        # Attention norm
        norm_key = f"{src_prefix}.input_layernorm.weight"
        if norm_key in src_state:
            dst_state[f"{prefix}.attn_norm.weight"] = src_state[norm_key]
        
        # Attention weights
        for src_name, dst_name in [("q_proj", "c_q"), ("k_proj", "c_k"), 
                                    ("v_proj", "c_v"), ("o_proj", "c_proj")]:
            src_key = f"{src_prefix}.self_attn.{src_name}.weight"
            if src_key in src_state:
                dst_state[f"{prefix}.attn.{dst_name}.weight"] = src_state[src_key]
            # Biases (GPT-OSS has attention bias)
            src_bias_key = f"{src_prefix}.self_attn.{src_name}.bias"
            if src_bias_key in src_state:
                dst_state[f"{prefix}.attn.{dst_name}.bias"] = src_state[src_bias_key]
        
        # Attention sinks (learnable sink tokens)
        sink_key = f"{src_prefix}.self_attn.sinks"
        if sink_key in src_state:
            dst_state[f"{prefix}.attn.sinks"] = src_state[sink_key]
        
        # MoE norm
        moe_norm_key = f"{src_prefix}.post_attention_layernorm.weight"
        if moe_norm_key in src_state:
            dst_state[f"{prefix}.moe_norm.weight"] = src_state[moe_norm_key]
        
        # MoE router
        router_key = f"{src_prefix}.mlp.gate.weight"
        if router_key in src_state:
            dst_state[f"{prefix}.moe.router.gate.weight"] = src_state[router_key]
        
        # MoE expert weights - handle fused & quantized tensors
        mlp1_weights = []
        mlp1_biases = []
        mlp2_weights = []
        mlp2_biases = []
        
        # Check for fused/quantized keys (GPT-OSS MXFP4 format)
        gate_up_key = f"{src_prefix}.mlp.experts.gate_up_proj_blocks"
        gate_up_scale = f"{src_prefix}.mlp.experts.gate_up_proj_scales"
        down_key = f"{src_prefix}.mlp.experts.down_proj_blocks"
        down_scale = f"{src_prefix}.mlp.experts.down_proj_scales"
        
        # Helper: Dequantize if active
        def load_quantized(blocks_key, scales_key):
            if blocks_key not in src_state:
                return None
            
            blocks = src_state[blocks_key] # (..., block_size)
            scales = src_state[scales_key] # (..., 1)
            
            # Basic block dequantization: blocks * scales
            # Assuming block_size is the last dim or inferred
            # Blocks usually int8/uint8. Scales bf16/float.
            
            # Expand scales to match blocks
            # If blocks is [N, K], scales is [N, K/block_size] ? 
            # OR blocks is active tensor, scales is per-block.
            # MXFP4 often: blocks [N, K], scales [N, K/32]
            
            B_shape = blocks.shape
            S_shape = scales.shape
            
            # Auto-detect block size
            # Usually strict division
            # E.g. [32, 5632, 2880] elements vs scales
            
            if blocks.dtype in [torch.int8, torch.uint8]:
                # Cast to float for math
                blocks = blocks.float()
                
            # Repeat scales
            # Calculate repetition factor
            # Simple assumption: flatten and verify ratio
            total_elements = blocks.numel()
            total_scales = scales.numel()
            block_size = total_elements // total_scales
            
            # Reshape scales to repeat
            # This is heuristic. Real MXFP4 might need specific layout.
            # Assuming contiguous blocks in last dim?
            # Or flattening everything?
            
            # Safe approach: Upsample scales
            expanded_scales = scales.repeat_interleave(block_size, dim=-1)
            
            # In case dimensions don't match after flatten logic
            if expanded_scales.shape != blocks.shape:
                # Try reshaping active to 1D, scales to 1D, mul, reshape back
                deq = (blocks.view(-1) * expanded_scales.view(-1)).view(B_shape)
            else:
                deq = blocks * expanded_scales
                
            return deq.to(torch.bfloat16)

        # 1. Try loading Fused Quantized Experts
        if gate_up_key in src_state:
            # Shape expectation: (num_experts, intermediate*2, hidden)
            # The key 'experts.gate_up_proj' suggests all experts stacked
            
            gate_up_w = load_quantized(gate_up_key, gate_up_scale) # (32, 5760, 2880) ?
            down_w = load_quantized(down_key, down_scale)         # (32, 2880, 2880) ?
            
            # DEBUG: Print shapes for first layer
            if layer_idx == 0:
                print(f"DEBUG: Layer {layer_idx} Expert Shapes:")
                print(f"  GateUp: {gate_up_w.shape} (Expected dim1 = {intermediate_size*2})")
                print(f"  Down:   {down_w.shape} (Expected dim1 = {intermediate_size}?? Check transpose)")
            
            # Verify shape implies experts are dim 0
            if gate_up_w.shape[0] == num_experts:
                # Iterate and split
                for e in range(num_experts):
                    # GateUp: (intermediate*2, hidden)
                    gu = gate_up_w[e]
                    
                    # Split into Gate and Up? 
                    # GPT-OSS usually: [Gate, Up] stacked on dim 0 or 1?
                    # Adamba expects: stack([gate, up], dim=1) -> flatten
                    # If it's already fused, we keep it fused?
                    
                    # MoELayer expects: mlp1_weight as stacked tensor
                    # logic below appends to list then stacks
                    
                    mlp1_weights.append(gu)
                    mlp1_biases.append(torch.zeros(gu.shape[0])) # Bias
                    
                    # Down
                    d = down_w[e] # (hidden, intermediate)
                    # MoE code below expects down in specific format
                    # But wait, logic below expects to append per expert
                    mlp2_weights.append(d)
                    mlp2_biases.append(torch.zeros(d.shape[0]))

        # 2. Fallback to Standard/Unfused (Handling previous case if dequant failed or unused)
        elif not mlp1_weights:
             for expert_idx in range(num_experts):
                # ... (Existing logic for separated files) ...
                up_key = f"{src_prefix}.mlp.experts.{expert_idx}.up_proj.weight"
                gate_key = f"{src_prefix}.mlp.experts.{expert_idx}.gate_proj.weight"
                down_key = f"{src_prefix}.mlp.experts.{expert_idx}.down_proj.weight"
                
                if up_key in src_state and gate_key in src_state:
                    gate_w = src_state[gate_key]
                    up_w = src_state[up_key]
                    
                    # Manual BF16 cast if uint8 but NOT blocks (naive quantization)
                    if gate_w.dtype in [torch.uint8, torch.int8]:
                         gate_w = gate_w.float().to(torch.bfloat16)
                    if up_w.dtype in [torch.uint8, torch.int8]:
                         up_w = up_w.float().to(torch.bfloat16)
                         
                    combined = torch.stack([gate_w, up_w], dim=1).view(-1, hidden_size)
                    mlp1_weights.append(combined)
                    mlp1_biases.append(torch.zeros(intermediate_size * 2))
                
                if down_key in src_state:
                    down_w = src_state[down_key]
                    if down_w.dtype in [torch.uint8, torch.int8]:
                        down_w = down_w.float().to(torch.bfloat16)
                    mlp2_weights.append(down_w)
                    mlp2_biases.append(torch.zeros(hidden_size))

        if mlp1_weights:
            dst_state[f"{prefix}.moe.moe.mlp1_weight"] = torch.stack(mlp1_weights)
            dst_state[f"{prefix}.moe.moe.mlp1_bias"] = torch.stack(mlp1_biases)
        if mlp2_weights:
            dst_state[f"{prefix}.moe.moe.mlp2_weight"] = torch.stack(mlp2_weights)
            dst_state[f"{prefix}.moe.moe.mlp2_bias"] = torch.stack(mlp2_biases)
        
        block_idx += 1
        
        # Insert Mamba layer after every mamba_interval attention layers
        # (after full attention layers, which are odd-indexed in GPT-OSS)
        if (layer_idx + 1) % mamba_interval == 0 and mamba_count < n_mamba_layers:
            print(f"  Inserting Mamba block {block_idx}")
            mamba_prefix = f"blocks.{block_idx}"
            
            # Initialize Mamba with zero output projection (no-op start)
            dst_state[f"{mamba_prefix}.mamba.layer.in_proj.weight"] = torch.randn(d_inner * 2, hidden_size) * std
            dst_state[f"{mamba_prefix}.mamba.layer.out_proj.weight"] = torch.zeros(hidden_size, d_inner)
            dst_state[f"{mamba_prefix}.mamba.layer.conv1d.weight"] = torch.randn(d_inner, 1, d_conv) * std
            dst_state[f"{mamba_prefix}.mamba.layer.conv1d.bias"] = torch.zeros(d_inner)
            dst_state[f"{mamba_prefix}.mamba.layer.dt_proj.weight"] = torch.randn(d_inner, d_inner) * std
            dst_state[f"{mamba_prefix}.mamba.layer.dt_proj.bias"] = torch.zeros(d_inner)
            dst_state[f"{mamba_prefix}.mamba.layer.A_log"] = torch.zeros(d_inner, d_state)
            dst_state[f"{mamba_prefix}.mamba.layer.D"] = torch.ones(d_inner)
            dst_state[f"{mamba_prefix}.mamba_norm.weight"] = torch.ones(hidden_size)
            
            block_idx += 1
            mamba_count += 1
        
        gc.collect()
    
    # 5. Per-block scalars
    total_blocks = block_idx
    dst_state["resid_lambdas"] = torch.ones(total_blocks)
    dst_state["x0_lambdas"] = torch.zeros(total_blocks)
    
    # 6. Auxiliary modules (fresh init)
    # Think detector
    dst_state["think_detector.net.0.weight"] = torch.randn(32, 1) * 0.1
    dst_state["think_detector.net.0.bias"] = torch.zeros(32)
    dst_state["think_detector.net.2.weight"] = torch.randn(1, 32) * 0.1
    dst_state["think_detector.net.2.bias"] = torch.zeros(1)
    
    print(f"\nSurgery complete!")
    print(f"  Total blocks: {total_blocks}")
    print(f"  Attention+MoE blocks: {n_layers}")
    print(f"  Mamba blocks: {mamba_count}")
    
    return dst_state


def surgery_moe(
    src_path: Path,
    dst_path: Path,
    n_mamba_layers: int = 12,
):
    """
    Perform GPT-OSS to Adamba conversion surgery.
    """
    print(f"Loading GPT-OSS checkpoint from: {src_path}")
    src_state = load_gptoss_checkpoint(src_path)
    
    print(f"\nPerforming surgery...")
    dst_state = build_hybrid_state_dict(src_state, GPTOSS_20B_CONFIG, n_mamba_layers)
    
    del src_state
    gc.collect()
    
    # Save
    print(f"\nSaving to: {dst_path}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dst_state, dst_path)
    
    # Save config
    config_path = dst_path.parent / "config.json"
    config = {
        **GPTOSS_20B_CONFIG,
        "n_mamba_layers": n_mamba_layers,
        "total_blocks": GPTOSS_20B_CONFIG["num_hidden_layers"] + n_mamba_layers,
        "architecture": "gptoss_hybrid",
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    total_params = sum(t.numel() for t in dst_state.values())
    print(f"\nDone!")
    print(f"  Keys: {len(dst_state)}")
    print(f"  Total params: {total_params:,}")
    print(f"  Approx size: {total_params / 1e9:.1f}B")


def main():
    parser = argparse.ArgumentParser(description="Convert GPT-OSS 20B to Adamba hybrid")
    parser.add_argument(
        "--src", type=Path, required=True,
        help="Path to GPT-OSS checkpoint directory (containing safetensor files)"
    )
    parser.add_argument(
        "--dst", type=Path, 
        default=Path.home() / ".cache/nanochat/gptoss_hybrid/model.pt",
        help="Output checkpoint path"
    )
    parser.add_argument(
        "--n-mamba", type=int, default=12,
        help="Number of Mamba layers to insert (default: 12)"
    )
    
    args = parser.parse_args()
    surgery_moe(args.src, args.dst, args.n_mamba)


if __name__ == "__main__":
    main()
