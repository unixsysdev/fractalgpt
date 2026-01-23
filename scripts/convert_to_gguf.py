#!/usr/bin/env python3
"""
Convert Adamba PyTorch checkpoint to GGUF format for llama.cpp inference.

This allows running the hybrid Attention+Mamba model on AMD GPUs, CPU, or 
any platform supported by llama.cpp.

Usage:
    python scripts/convert_to_gguf.py \
        --checkpoint path/to/model.pt \
        --output adamba-5b-f16.gguf \
        --dtype f16
        
Requirements:
    pip install gguf numpy torch
"""

import argparse
import json
import struct
import numpy as np
from pathlib import Path
from typing import Dict, Any

# GGUF format constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF data types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8

# GGUF value types for metadata
GGUF_METADATA_VALUE_TYPE_UINT32 = 4
GGUF_METADATA_VALUE_TYPE_INT32 = 5
GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6
GGUF_METADATA_VALUE_TYPE_STRING = 8
GGUF_METADATA_VALUE_TYPE_ARRAY = 9


class GGUFWriter:
    """Simple GGUF file writer."""
    
    def __init__(self, path: Path, arch: str = "adamba"):
        self.path = path
        self.arch = arch
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, np.ndarray] = {}
        self.tensor_types: Dict[str, int] = {}
        
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the GGUF file."""
        self.metadata[key] = value
        
    def add_tensor(self, name: str, data: np.ndarray, dtype: int = GGML_TYPE_F16):
        """Add a tensor to the GGUF file."""
        self.tensors[name] = data
        self.tensor_types[name] = dtype
        
    def _write_string(self, f, s: str):
        """Write a length-prefixed string."""
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)
        
    def _write_metadata_value(self, f, value):
        """Write a metadata value with its type."""
        if isinstance(value, str):
            f.write(struct.pack('<I', GGUF_METADATA_VALUE_TYPE_STRING))
            self._write_string(f, value)
        elif isinstance(value, int):
            f.write(struct.pack('<I', GGUF_METADATA_VALUE_TYPE_UINT32))
            f.write(struct.pack('<I', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', GGUF_METADATA_VALUE_TYPE_FLOAT32))
            f.write(struct.pack('<f', value))
        elif isinstance(value, list):
            f.write(struct.pack('<I', GGUF_METADATA_VALUE_TYPE_ARRAY))
            if len(value) > 0 and isinstance(value[0], int):
                f.write(struct.pack('<I', GGUF_METADATA_VALUE_TYPE_UINT32))
            else:
                f.write(struct.pack('<I', GGUF_METADATA_VALUE_TYPE_FLOAT32))
            f.write(struct.pack('<Q', len(value)))
            for v in value:
                if isinstance(v, int):
                    f.write(struct.pack('<I', v))
                else:
                    f.write(struct.pack('<f', v))
                    
    def write(self):
        """Write the GGUF file."""
        with open(self.path, 'wb') as f:
            # Header
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))  # n_tensors
            f.write(struct.pack('<Q', len(self.metadata)))  # n_metadata
            
            # Metadata
            for key, value in self.metadata.items():
                self._write_string(f, key)
                self._write_metadata_value(f, value)
                
            # Tensor info (names, shapes, types)
            tensor_data_offset = 0
            tensor_offsets = {}
            
            for name, data in self.tensors.items():
                self._write_string(f, name)
                # n_dims
                f.write(struct.pack('<I', len(data.shape)))
                # dims (reversed for row-major to column-major)
                for dim in reversed(data.shape):
                    f.write(struct.pack('<Q', dim))
                # type
                f.write(struct.pack('<I', self.tensor_types[name]))
                # offset
                f.write(struct.pack('<Q', tensor_data_offset))
                tensor_offsets[name] = tensor_data_offset
                
                # Calculate size based on dtype
                dtype = self.tensor_types[name]
                if dtype == GGML_TYPE_F32:
                    size = data.size * 4
                elif dtype == GGML_TYPE_F16:
                    size = data.size * 2
                else:
                    size = data.size  # Quantized types vary
                    
                # Align to 32 bytes
                tensor_data_offset += size
                tensor_data_offset = (tensor_data_offset + 31) & ~31
                
            # Padding to align tensor data
            current_pos = f.tell()
            padding = (32 - (current_pos % 32)) % 32
            f.write(b'\x00' * padding)
            
            # Tensor data
            for name, data in self.tensors.items():
                dtype = self.tensor_types[name]
                if dtype == GGML_TYPE_F32:
                    f.write(data.astype(np.float32).tobytes())
                elif dtype == GGML_TYPE_F16:
                    f.write(data.astype(np.float16).tobytes())
                    
                # Align to 32 bytes
                current_pos = f.tell()
                padding = (32 - (current_pos % 32)) % 32
                f.write(b'\x00' * padding)


def convert_adamba_to_gguf(
    checkpoint_path: Path,
    output_path: Path,
    dtype: str = "f16"
):
    """Convert Adamba checkpoint to GGUF format."""
    import torch
    
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # Determine output dtype
    if dtype == "f16":
        ggml_dtype = GGML_TYPE_F16
        np_dtype = np.float16
    elif dtype == "f32":
        ggml_dtype = GGML_TYPE_F32
        np_dtype = np.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # Create GGUF writer
    writer = GGUFWriter(output_path, arch="adamba")
    
    # Add architecture metadata
    writer.add_metadata("general.architecture", "adamba")
    writer.add_metadata("general.name", "Adamba Hybrid 5B")
    writer.add_metadata("general.description", "Attention + Mamba hybrid with Matryoshka support")
    
    # Model config (infer from state dict)
    # Find embedding size from token embedding
    if "transformer.wte.weight" in state_dict:
        vocab_size, n_embd = state_dict["transformer.wte.weight"].shape
        writer.add_metadata("adamba.vocab_size", int(vocab_size))
        writer.add_metadata("adamba.embedding_size", int(n_embd))
        print(f"  Vocab size: {vocab_size}, Embedding dim: {n_embd}")
    
    # Count layers
    n_layers = 0
    for key in state_dict.keys():
        if key.startswith("blocks.") and ".attn." in key:
            layer_num = int(key.split(".")[1])
            n_layers = max(n_layers, layer_num + 1)
    writer.add_metadata("adamba.n_layer", n_layers)
    print(f"  Layers: {n_layers}")
    
    # Convert tensors
    print(f"🔄 Converting {len(state_dict)} tensors to {dtype}...")
    
    converted = 0
    skipped = 0
    
    for name, tensor in state_dict.items():
        # Convert to numpy
        np_tensor = tensor.detach().float().numpy()
        
        # Skip 0-D tensors (scalars like lambda values)
        if np_tensor.ndim == 0:
            skipped += 1
            continue
            
        # Map PyTorch names to GGUF names
        gguf_name = name.replace(".", "_")
        
        writer.add_tensor(gguf_name, np_tensor.astype(np_dtype), ggml_dtype)
        converted += 1
        
    print(f"  Converted: {converted}, Skipped: {skipped}")
    
    # Write GGUF file
    print(f"💾 Writing GGUF: {output_path}")
    writer.write()
    
    # Report file size
    file_size = output_path.stat().st_size / (1024**3)
    print(f"✅ Done! File size: {file_size:.2f} GB")
    

def main():
    parser = argparse.ArgumentParser(description="Convert Adamba to GGUF")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Input checkpoint path")
    parser.add_argument("--output", type=Path, default=Path("adamba.gguf"), help="Output GGUF path")
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f16", help="Output dtype")
    args = parser.parse_args()
    
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
    convert_adamba_to_gguf(args.checkpoint, args.output, args.dtype)
    
    print(f"""
🎉 Conversion complete!

To run with llama.cpp:
    ./llama-cli -m {args.output} -p "Hello, I am Adamba"

Note: llama.cpp needs custom support for Mamba layers.
For now, this GGUF works with ggml-based inference engines that support Mamba.
""")


if __name__ == "__main__":
    main()
