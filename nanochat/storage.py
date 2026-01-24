"""
Auto-detect best storage location for training caches.

Import this at the start of any script to automatically configure paths:
    from nanochat.storage import setup_storage
    paths = setup_storage()
    
Then use:
    paths.checkpoints / "model.pt"
    paths.surgery / "hybrid.pt"
    paths.huggingface  # For HF_HOME
"""

import os
import subprocess
from pathlib import Path
from dataclasses import dataclass


@dataclass
class StoragePaths:
    """Container for all storage paths."""
    root: Path
    checkpoints: Path
    surgery: Path
    huggingface: Path
    pip: Path
    torch: Path
    available_gb: int


def find_best_mount(min_space_gb: int = 50) -> tuple[Path, int]:
    """
    Find mount point with most available space.
    
    Priority:
    1. /mnt/pt (if exists and has space) - preferred for cloud VMs
    2. Mount with most available space
    3. Fallback to ~/.cache
    """
    # First check preferred mount
    PREFERRED_MOUNT = Path("/mnt/pt")
    
    if PREFERRED_MOUNT.exists():
        try:
            import shutil
            total, used, free = shutil.disk_usage(PREFERRED_MOUNT)
            free_gb = free // (1024**3)
            if free_gb >= min_space_gb:
                return PREFERRED_MOUNT, free_gb
        except Exception:
            pass
    
    # Fall back to auto-detection
    try:
        result = subprocess.run(
            ["df", "-BG", "--output=avail,target"],
            capture_output=True, text=True, check=True
        )
        
        best_mount = Path.home() / ".cache"
        best_space = 0
        
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 2:
                avail = parts[0].rstrip('G')
                try:
                    avail_gb = int(avail)
                except ValueError:
                    continue
                    
                target = parts[1]
                
                # Skip virtual filesystems
                if any(skip in target for skip in ['/boot', '/snap', '/run', '/dev', '/sys', '/proc']):
                    continue
                # Skip tmpfs (usually small)
                if target == '/tmp' and avail_gb < 100:
                    continue
                    
                if avail_gb > best_space:
                    best_space = avail_gb
                    best_mount = Path(target)
        
        return best_mount, best_space
        
    except Exception:
        # Fallback to home directory
        return Path.home() / ".cache", 0


def setup_storage(min_space_gb: int = 50, quiet: bool = False) -> StoragePaths:
    """
    Auto-detect best storage and configure all cache paths.
    
    Sets environment variables and returns a StoragePaths object.
    """
    mount, available_gb = find_best_mount(min_space_gb)
    
    # Create cache structure
    root = mount / "adamba_cache"
    paths = StoragePaths(
        root=root,
        checkpoints=root / "checkpoints",
        surgery=root / "surgery", 
        huggingface=root / "huggingface",
        pip=root / "pip",
        torch=root / "torch",
        available_gb=available_gb,
    )
    
    # Create directories
    for p in [paths.checkpoints, paths.surgery, paths.huggingface, paths.pip, paths.torch]:
        p.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables
    os.environ["HF_HOME"] = str(paths.huggingface)
    os.environ["HF_DATASETS_CACHE"] = str(paths.huggingface / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(paths.huggingface / "transformers")
    os.environ["PIP_CACHE_DIR"] = str(paths.pip)
    os.environ["TORCH_HOME"] = str(paths.torch)
    os.environ["ADAMBA_CHECKPOINT_DIR"] = str(paths.checkpoints)
    os.environ["ADAMBA_SURGERY_DIR"] = str(paths.surgery)
    
    if not quiet:
        print(f"📁 Storage: {mount} ({available_gb}GB available)")
        print(f"   Checkpoints: {paths.checkpoints}")
        if available_gb < min_space_gb:
            print(f"   ⚠️  Warning: Less than {min_space_gb}GB available!")
    
    return paths


# Auto-setup on import (optional, can be disabled)
_paths = None

def get_paths() -> StoragePaths:
    """Get or create storage paths (lazy initialization)."""
    global _paths
    if _paths is None:
        _paths = setup_storage(quiet=True)
    return _paths
