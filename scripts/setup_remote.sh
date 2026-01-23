#!/bin/bash
# setup_remote.sh - Setup script for HPC-AI training environment

set -e

echo "=== Adamba Training Environment Setup ==="

# 1. System packages
echo "Installing system dependencies..."
apt-get update && apt-get install -y git python3-pip rsync

# 2. Clone repo
echo "Cloning repository..."
if [ ! -d "nanochat" ]; then
    git clone https://github.com/unixsysdev/adamba.git nanochat
fi
cd nanochat

# 3. Python dependencies
echo "Installing Python packages..."
pip install -e .  # Install from repo root (has pyproject.toml)
pip install mamba-ssm causal-conv1d
pip install wandb pyarrow requests

# 4. Create cache directory
echo "Creating cache directories..."
mkdir -p ~/.cache/nanochat/hybrid_checkpoints/d32_2048
mkdir -p ~/.cache/nanochat/base_data

# 5. Download small dataset sample
echo "Downloading dataset sample (5 shards)..."
python -m nanochat.dataset --num-files=5

# 6. Test imports
echo "Testing imports..."
python -c "from nanochat.hybrid_gpt import HybridGPT; print('✓ HybridGPT')"
python -c "from mamba_ssm import Mamba; print('✓ Mamba')"
python -c "import torch; print(f'✓ CUDA: {torch.cuda.device_count()} GPUs')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Transfer checkpoint from local machine:"
echo "   rsync -avz ~/.cache/nanochat/hybrid_checkpoints/ user@this-machine:~/.cache/nanochat/hybrid_checkpoints/"
echo ""
echo "2. Quick test (5 iterations):"
echo "   torchrun --nproc_per_node=8 -m scripts.fractal_train --phase=1 --num-iterations=5"
echo ""
echo "3. Full training:"
echo "   torchrun --nproc_per_node=8 -m scripts.fractal_train --phase=1 --checkpoint=~/.cache/nanochat/hybrid_checkpoints/d32_2048/model.pt"
