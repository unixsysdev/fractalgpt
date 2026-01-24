#!/bin/bash
# setup_cache_paths.sh - Auto-detect best storage location and configure all caches
#
# Usage: 
#   source scripts/setup_cache_paths.sh
#   # or add to ~/.bashrc: source /path/to/setup_cache_paths.sh
#
# This will set:
#   - HF_HOME (HuggingFace models/datasets)
#   - PIP_CACHE_DIR (pip packages)
#   - TORCH_HOME (PyTorch hub)
#   - ADAMBA_CHECKPOINT_DIR (our training checkpoints)

set -e

# Minimum required space in GB
MIN_SPACE_GB=${MIN_SPACE_GB:-100}

# Find mount point with most available space
find_best_mount() {
    # Get mount points with free space, sorted by available space (descending)
    # Filter out tmpfs, devtmpfs, and other virtual filesystems
    df -BG --output=avail,target 2>/dev/null | \
        tail -n +2 | \
        grep -vE '(/boot|/snap|/run|/dev|tmpfs)' | \
        sort -rn | \
        head -1 | \
        awk '{print $2, $1}'
}

# Parse result
read BEST_MOUNT AVAIL_SPACE <<< $(find_best_mount)
AVAIL_GB=${AVAIL_SPACE%G}

echo "🔍 Storage Analysis:"
echo "   Best mount: $BEST_MOUNT"
echo "   Available:  ${AVAIL_GB}GB"

if [ "$AVAIL_GB" -lt "$MIN_SPACE_GB" ]; then
    echo "⚠️  WARNING: Less than ${MIN_SPACE_GB}GB available on best mount!"
fi

# Create cache directory structure
CACHE_ROOT="${BEST_MOUNT}/adamba_cache"
mkdir -p "$CACHE_ROOT"/{huggingface,pip,torch,checkpoints,surgery}

# Export environment variables
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export TORCH_HOME="$CACHE_ROOT/torch"
export ADAMBA_CHECKPOINT_DIR="$CACHE_ROOT/checkpoints"
export ADAMBA_SURGERY_DIR="$CACHE_ROOT/surgery"

# Also set XDG cache if on Linux
export XDG_CACHE_HOME="$CACHE_ROOT"

echo ""
echo "✅ Cache paths configured:"
echo "   HF_HOME:              $HF_HOME"
echo "   PIP_CACHE_DIR:        $PIP_CACHE_DIR"
echo "   TORCH_HOME:           $TORCH_HOME"
echo "   ADAMBA_CHECKPOINT_DIR: $ADAMBA_CHECKPOINT_DIR"
echo "   ADAMBA_SURGERY_DIR:   $ADAMBA_SURGERY_DIR"
echo ""
echo "📝 To persist, add to ~/.bashrc:"
echo "   source $(realpath "$0")"
