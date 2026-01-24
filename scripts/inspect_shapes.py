import sys
import os
from safetensors import safe_open

if len(sys.argv) < 2:
    print("Usage: python scripts/inspect_shapes.py <path_to_safetensor>")
    sys.exit(1)

path = sys.argv[1]
print(f"Inspecting: {path}")

with safe_open(path, framework="pt", device="cpu") as f:
    keys = f.keys()
    # Find a set of related keys for one expert
    # Looking for: model.layers.X.mlp.experts.down_proj_{blocks,scales}
    
    found = False
    for k in keys:
        if "down_proj_blocks" in k:
            base = k.replace("down_proj_blocks", "")
            scales_key = base + "down_proj_scales"
            
            if scales_key in keys:
                blocks = f.get_tensor(k)
                scales = f.get_tensor(scales_key)
                
                print(f"\nFound Quantized Tensor Pair:")
                print(f"  Blocks Key: {k}")
                print(f"  Blocks Shape: {blocks.shape}")
                print(f"  Blocks Dtype: {blocks.dtype}")
                
                print(f"  Scales Key: {scales_key}")
                print(f"  Scales Shape: {scales.shape}")
                print(f"  Scales Dtype: {scales.dtype}")
                
                # Calculate ratio
                blocks_numel = blocks.numel()
                scales_numel = scales.numel()
                ratio = blocks_numel / scales_numel
                print(f"  Ratio (Block Size): {ratio}")
                
                found = True
                break
    
    if not found:
        print("No quantized block/scale pairs found in this file.")
