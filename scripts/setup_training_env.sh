#!/bin/bash
# setup_training_env.sh - One-click setup for Adamba training on cloud VMs
#
# Run this after SSH-ing into your rented GPU machine:
#   bash scripts/setup_training_env.sh
#
# Tested with: PyTorch 2.8.0 + CUDA 12.9 image

set -e

echo "🚀 Adamba Training Environment Setup"
echo "======================================"

# 1. Create mount point for data volume
echo ""
echo "📁 Setting up /mnt/pt mount point..."
if [ ! -d "/mnt/pt" ]; then
    sudo mkdir -p /mnt/pt
    sudo chown $USER:$USER /mnt/pt
    echo "   Created /mnt/pt (mount your volume here)"
else
    echo "   /mnt/pt already exists"
fi

# 2. Check CUDA version
echo ""
echo "🔧 Checking CUDA version..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo "   CUDA: $CUDA_VERSION"
else
    echo "   ⚠️  nvcc not found (mamba-ssm may use fallback)"
fi

# 3. Check PyTorch version
echo ""
echo "🔧 Checking PyTorch..."
PYTORCH_CUDA=$(python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" 2>/dev/null || echo "Not installed")
echo "   $PYTORCH_CUDA"

# 4. Install mamba-ssm with pre-built wheels if available
echo ""
echo "📦 Installing mamba-ssm and causal-conv1d..."

# Try pre-built wheels first (much faster)
pip install causal-conv1d>=1.4.0 --no-build-isolation 2>/dev/null || {
    echo "   Building causal-conv1d from source (this takes ~5 min)..."
    pip install causal-conv1d>=1.4.0
}

pip install mamba-ssm>=2.2.0 --no-build-isolation 2>/dev/null || {
    echo "   Building mamba-ssm from source (this takes ~10 min)..."
    pip install mamba-ssm>=2.2.0
}

# 5. Install other dependencies
echo ""
echo "📦 Installing other dependencies..."
pip install -q safetensors huggingface_hub wandb einops

# 6. Verify mamba works
echo ""
echo "✅ Verifying mamba-ssm..."
python -c "from mamba_ssm import Mamba; print('   mamba-ssm OK')" || echo "   ⚠️  mamba-ssm not working (will use fallback)"

# 7. Clone/update repo
echo ""
echo "📥 Setting up nanochat repository..."
if [ -d "nanochat" ]; then
    cd nanochat && git pull && cd ..
else
    git clone https://github.com/unixsysdev/adamba.git nanochat || echo "   Using existing directory"
fi

# 8. Test storage detection
echo ""
echo "📁 Testing storage detection..."
cd nanochat
python -c "from nanochat.storage import setup_storage; p = setup_storage(); print(f'   Checkpoints: {p.checkpoints}')"

echo ""
echo "======================================"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Mount your data volume:  sudo mount /dev/nvme1n1 /mnt/pt"
echo "  2. Run surgery:             python -m scripts.surgery_moe"
echo "  3. Start training:          torchrun --nproc_per_node=8 -m scripts.fractal_train --phase=1 --model-type=gptoss --depth=24 --compile"
