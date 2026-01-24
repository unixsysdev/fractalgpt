import sys
from safetensors import safe_open
from pathlib import Path

def inspect(path):
    print(f"Inspecting: {path}")
    p = Path(path)
    files = list(p.glob("*.safetensors"))
    if not files:
        print("No safetensors found.")
        return

    # Check first file
    f = files[0]
    print(f"Reading keys from {f.name}...")
    with safe_open(f, framework="pt", device="cpu") as st:
        keys = st.keys()
        print(f"Total keys in first file: {len(keys)}")
        print("\nSample keys (first 20):")
        for k in sorted(list(keys))[:20]:
            print(f"  {k}")
            
        # Check for expert-like keys
        print("\nChecking for 'expert' or 'mlp' keys:")
        expert_keys = [k for k in keys if "expert" in k or "mlp" in k]
        for k in expert_keys[:20]:
            print(f"  {k}")
        
    print(f"\nTotal safetensor files: {len(files)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_keys.py <path_to_checkpoint_folder>")
    else:
        inspect(sys.argv[1])
