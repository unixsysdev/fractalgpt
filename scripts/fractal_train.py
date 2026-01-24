"""
Fractal training script for Nano-Fractal hybrid model.

Supports:
1. Progressive training: 6.4B (2048) → 9.3B (2560) → 20B (4096)
2. Matryoshka training with dimension sampling  
3. Energy penalty for compute minimization
4. Per-layer probes for adaptive capacity

Usage:
    # Stage 1: Train 6.4B model
    torchrun --nproc_per_node=8 -m scripts.fractal_train \
        --checkpoint ~/.cache/nanochat/hybrid_checkpoints/d32_2048/model.pt
    
    # Local smoke test (synthetic data)
    python -m scripts.fractal_train --depth=4 --num-iterations=5
"""

import argparse
import time
import os
import json
import random
from pathlib import Path
from contextlib import nullcontext

# Auto-configure storage paths (finds mount with most space)
from nanochat.storage import setup_storage
storage_paths = setup_storage(quiet=False)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Fix TF32 setting conflict (CUDA 12.9 issue)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from nanochat.common import (
    compute_init, get_dist_info, print0, print_banner,
    get_peak_flops, ColoredFormatter,
)
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.hybrid_gpt import HybridGPT, HybridConfig, HybridMoEGPT, GptOssMoEConfig
from nanochat.matryoshka import (
    MLP_DIM_LEVELS, sample_dim_level, compute_energy_penalty,
)

print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Train Nano-Fractal hybrid model")

# Logging
parser.add_argument("--run", type=str, default="fractal", help="wandb run name")

# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps")

# Model architecture
parser.add_argument("--depth", type=int, default=32, help="number of attention layers")
parser.add_argument("--n-mamba", type=int, default=32, help="number of mamba layers")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max sequence length")
parser.add_argument("--head-dim", type=int, default=128, help="head dimension")
parser.add_argument("--base-dim", type=int, default=2048, help="original nanochat dimension")
parser.add_argument("--expanded-dim", type=int, default=0, help="expanded model dimension (0=auto from checkpoint)")
parser.add_argument("--vocab-size", type=int, default=0, help="vocab size (0=auto from checkpoint or model type)")

# Matryoshka
parser.add_argument("--matryoshka", action="store_true", help="enable Matryoshka training")
parser.add_argument("--dim-levels", type=str, default="", help="comma-separated dim levels (auto if empty)")
parser.add_argument("--energy-lambda", type=float, default=0.01, help="energy penalty weight")
parser.add_argument("--sample-dim", action="store_true", help="sample one dim level per batch")

# Training
parser.add_argument("--device-batch-size", type=int, default=8, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=262144, help="total batch size in tokens")
parser.add_argument("--num-iterations", type=int, default=1000, help="number of iterations")
parser.add_argument("--warmup-iters", type=int, default=100, help="warmup iterations")

# Learning rates
parser.add_argument("--matrix-lr", type=float, default=0.02, help="LR for Muon")
parser.add_argument("--embedding-lr", type=float, default=0.2, help="LR for embeddings")
parser.add_argument("--probe-lr", type=float, default=0.01, help="LR for probes")

# Checkpointing
parser.add_argument("--checkpoint", type=Path, default=None, help="path to hybrid checkpoint")
parser.add_argument("--save-every", type=int, default=50, help="save checkpoint every N steps")
parser.add_argument("--eval-every", type=int, default=50, help="evaluate every N steps")
parser.add_argument("--keep-checkpoints", type=int, default=3, help="keep only last N checkpoints (0=keep all)")
parser.add_argument("--resume", action="store_true", help="resume from checkpoint (load optimizer state)")

# Probes
parser.add_argument("--train-probes", action="store_true", help="train per-layer probes")
parser.add_argument("--use-probes", action="store_true", help="use probes for dim decisions")

# Phase-aware training
parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=2,
                    help="Training phase: 1=Mamba only (freeze attn/mlp, no Matryoshka), "
                         "2=All + Matryoshka + Gates, 3=Expansion (train new dims)")

# Performance optimizations
parser.add_argument("--compile", action="store_true", help="use torch.compile for faster training")
parser.add_argument("--compile-mode", type=str, default="reduce-overhead", 
                    choices=["default", "reduce-overhead", "max-autotune"],
                    help="torch.compile mode")
parser.add_argument("--no-muon", action="store_true", help="disable Muon optimizer (use AdamW only, saves memory)")
parser.add_argument("--gradient-checkpointing", action="store_true", help="use gradient checkpointing (saves ~50%% memory)")
parser.add_argument("--model-type", type=str, default="nanochat", help="model variant (nanochat|gptoss)")

args = parser.parse_args()

# -----------------------------------------------------------------------------
# Initialize compute environment
device_type = args.device_type or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

# Mixed precision context
if device_type == "cuda":
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
else:
    autocast_ctx = nullcontext()

synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

# -----------------------------------------------------------------------------
# Load config from checkpoint if available
checkpoint_config = None
if args.checkpoint and args.checkpoint.exists():
    config_path = args.checkpoint.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            checkpoint_config = json.load(f)
        print0(f"Loaded config from checkpoint: {config_path}")

# Determine expanded dimension
if args.expanded_dim > 0:
    expanded_dim = args.expanded_dim
elif checkpoint_config and 'n_embd_expanded' in checkpoint_config:
    expanded_dim = checkpoint_config['n_embd_expanded']
else:
    expanded_dim = 2048  # Stage 1 default

print0(f"Using expanded dimension: {expanded_dim}")

# Determine dim levels (filter to not exceed expanded_dim)
if args.dim_levels:
    dim_levels = [int(d) for d in args.dim_levels.split(",")]
else:
    # Auto levels based on expanded dim
    all_levels = [128, 256, 512, 1024, 2048, 2560, 3072, 4096]
    dim_levels = [d for d in all_levels if d <= expanded_dim]

dim_levels = [d for d in dim_levels if d <= expanded_dim]  # Safety filter
print0(f"Matryoshka dim levels: {dim_levels}")

# Model setup
print0("Setting up model...")

# Determine vocab size
vocab_size = args.vocab_size
if vocab_size == 0: # If not specified by argument
    if checkpoint_config and "vocab_size" in checkpoint_config:
        vocab_size = checkpoint_config["vocab_size"]
        print0(f"Using vocab size from checkpoint config: {vocab_size}")
    else:
        vocab_size = 65536 # Default for nanochat
        print0(f"Using default nanochat vocab size: {vocab_size}")

if args.model_type == "gptoss" and vocab_size == 65536:
    # Override for GPT-OSS if default nanochat vocab size was used
    vocab_size = 201088
    print0(f"Overriding to GPT-OSS default vocab size: {vocab_size}")

# Create config and model based on type
if args.model_type == "gptoss":
    print0("Initializing GPT-OSS 20B MoE model architecture...")
    
    # Use correct defaults for GPT-OSS
    gptoss_depth = args.depth if args.depth != 32 else 24
    gptoss_n_mamba = args.n_mamba if args.n_mamba != 32 else 12
    
    # Also check checkpoint config
    if checkpoint_config:
        if "n_layer" in checkpoint_config:
            gptoss_depth = checkpoint_config["n_layer"]
        if "n_mamba_layer" in checkpoint_config:
            gptoss_n_mamba = checkpoint_config["n_mamba_layer"]
    
    print0(f"  GPT-OSS layers: {gptoss_depth} attention + {gptoss_n_mamba} mamba")
    
    # Map CLI args to MoE config
    config = GptOssMoEConfig(
        vocab_size=vocab_size,
        num_hidden_layers=gptoss_depth,
        n_mamba_layers=gptoss_n_mamba,
        num_attention_heads=64,       # Fixed for GPT-OSS 20B
        num_key_value_heads=8,        # Fixed
        head_dim=64,                  # Fixed
        num_experts=32,
        experts_per_token=4,
        mlp_dim_levels=dim_levels,    # Use our Matryoshka levels
    )
    
    with torch.device("meta"):
        model = HybridMoEGPT(config)
        
else:
    # Standard Nano# Determine layer counts
    n_layer = args.depth
    n_mamba = args.n_mamba

    # 1. Prioritize checkpoint config
    if checkpoint_config:
        if "n_layer" in checkpoint_config:
            n_layer = checkpoint_config["n_layer"]
            print0(f"Using n_layer from checkpoint: {n_layer}")
        if "n_mamba_layer" in checkpoint_config:
            n_mamba = checkpoint_config["n_mamba_layer"]
            print0(f"Using n_mamba_layer from checkpoint: {n_mamba}")

    # 2. If using GPT-OSS and still on defaults, force correct values
    if args.model_type == "gptoss":
        # If user didn't specify depth (still default 32)
        if n_layer == 32 and args.depth == 32:
            n_layer = 24
            print0(f"Overriding to GPT-OSS default n_layer: {n_layer}")
        
        # If user didn't specify n_mamba (still default 32)
        if n_mamba == 32 and args.n_mamba == 32:
            n_mamba = 12
            print0(f"Overriding to GPT-OSS default n_mamba: {n_mamba}")

    # Create config
    config = HybridConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_mamba_layer=n_mamba,
        n_head=expanded_dim // args.head_dim,
        n_kv_head=expanded_dim // args.head_dim,
        n_embd=args.base_dim,
        n_embd_expanded=expanded_dim,
        mlp_dim_levels=dim_levels,
    )

    # Create model
    with torch.device("meta"):
        model = HybridGPT(config)

# For large models with FSDP: only rank 0 loads the full model
# Other ranks stay empty, FSDP will broadcast via sync_module_states
if ddp and args.model_type == "gptoss":
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        init_device = "cpu"
        print0("Rank 0: Loading model to CPU (FSDP will broadcast to all ranks)")
    else:
        # Other ranks: keep on meta device, FSDP will broadcast from rank 0
        init_device = "meta"
else:
    init_device = device
    rank = 0

if init_device != "meta":
    model = model.to_empty(device=init_device)

# Enable gradient checkpointing (saves ~50% memory)
if args.gradient_checkpointing:
    model.gradient_checkpointing = True
    print0("🧠 Gradient checkpointing enabled (saves memory, slower training)")

# Load checkpoint or init fresh - ONLY on rank 0 for large models
if args.checkpoint and args.checkpoint.exists():
    if init_device != "meta":  # Only load if we're not on meta device
        print0(f"Loading checkpoint: {args.checkpoint}")
        # Always load to CPU first for large models
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        
        # Filter out keys with shape mismatches (strict=False doesn't handle this)
        model_state = model.state_dict()
        filtered_state = {}
        skipped = []
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_state[k] = v
                else:
                    skipped.append(f"{k}: checkpoint {v.shape} vs model {model_state[k].shape}")
            else:
                filtered_state[k] = v  # New keys, let strict=False handle
        
        if skipped:
            print0(f"  Skipped {len(skipped)} mismatched keys:")
            for s in skipped[:5]:  # Show first 5
                print0(f"    {s}")
        
        model.load_state_dict(filtered_state, strict=False)
        del state_dict, filtered_state  # Free CPU memory
        import gc; gc.collect()
else:
    if init_device != "meta":
        print0("Initializing fresh weights...")
        model.init_weights()

# Wrap in DDP if needed
# FSDP Setup
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

# Define wrapping policy
def get_fsdp_policy(model_type):
    from nanochat.hybrid_gpt import HybridBlock
    from nanochat.moe_block import MoEBlock
    
    # Wrap each transformer block
    transformer_layer_cls = {
        HybridBlock,
        MoEBlock,
    }
    
    # Use **kwargs for compatibility with different PyTorch versions
    def policy_fn(module, recurse, **kwargs):
        # Extract nonwrapped_numel from kwargs (different PyTorch versions use different names)
        nonwrapped_numel = kwargs.get('nonwrapped_numel', kwargs.get('nonwrapped_child_numel', 0))
        return transformer_auto_wrap_policy(
            module, recurse, nonwrapped_numel, transformer_layer_cls=transformer_layer_cls
        )
    
    return policy_fn

if ddp:
    print0("Wrapping model with FSDP...")
    
    # Mixed precision policy (keep params in LP, grads in LP, buffers in FP32)
    # Using BF16 for everything if available
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.float32,
    )
    
    # For large models: use CPU offload during FSDP init to avoid OOM
    from torch.distributed.fsdp import CPUOffload
    cpu_offload = CPUOffload(offload_params=False) if args.model_type == "gptoss" else None

    model = FSDP(
        model,
        auto_wrap_policy=get_fsdp_policy(args.model_type),
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD, # Shard model + optimizer + grads
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True, # Needed for torch.compile
        sync_module_states=True, # Broadcast weights from rank 0 (needed for CPU loading)
        cpu_offload=cpu_offload,
    )
    
    print0(f"FSDP wrapping complete. Sharding strategy: FULL_SHARD")
orig_model = model.module if ddp else model

num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")

# -----------------------------------------------------------------------------
# Phase-specific freeze/unfreeze and Matryoshka settings
print0(f"🔧 Phase {args.phase} configuration:")

if args.phase == 1:
    # Phase 1: Train Mamba only (freeze attention/MLP), no Matryoshka, no gates
    print0("  ❄️  Freezing attention and MLP layers (even blocks)")
    print0("  🔥 Training Mamba layers only (odd blocks)")
    print0("  📐 No Matryoshka (full dims only)")
    
    frozen_count = 0
    trainable_count = 0
    
    for name, param in orig_model.named_parameters():
        should_train = False
        
    # Phase 1: Freeze everything except Mamba
    print0("Phase 1: Freezing Attention/MoE, Training Mamba only")
    for name, p in orig_model.named_parameters():
        p.requires_grad = False
        
        # Check if this is a Mamba layer
        # Naming convention: blocks.X.mamba... OR blocks.X.mlp... (if block X is Mamba)
        if "mamba" in name:
            p.requires_grad = True # The actual Mamba SSM
        
        # Critical Fix: Unfreeze everything inside Mamba blocks (including norms)
        # Mamba blocks are at indices 2, 5, 8... in the 2:1 interleave pattern (A, A, M)
        # Formula: (block_id + 1) % 3 == 0  => block_id % 3 == 2
        if "blocks." in name:
            try:
                parts = name.split('.') 
                # name is like blocks.2.norm.weight
                block_id = int(parts[1])
                if block_id % 3 == 2: # Indices 2, 5, 8... are Mamba
                     p.requires_grad = True
            except (ValueError, IndexError):
                pass
        
        # Also unfreeze explicit learnable scalars if they exist
        if "lambdas" in name:
            p.requires_grad = True
            
    # Verify freezing
    trainable = [n for n, p in orig_model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in orig_model.named_parameters() if not p.requires_grad]
    print0(f"  ❄️  Frozen: {len(frozen)} tensors")
    print0(f"  🔥 Trainable: {len(trainable)} tensors")
    
    # Force disable Matryoshka for Phase 1
    args.matryoshka = False
    args.sample_dim = False

elif args.phase == 2:
    # Phase 2: Train everything with Matryoshka and gates
    print0("  🔓 All layers trainable")
    print0("  📐 Matryoshka enabled")
    print0("  🚪 Gates training enabled")
    
    for param in orig_model.parameters():
        param.requires_grad = True
    
    # Force enable Matryoshka for Phase 2
    args.matryoshka = True
    args.sample_dim = True

elif args.phase == 3:
    # Phase 3: Expansion - train new expanded dims, lower LR for original
    print0("  📈 Training expanded weights")
    print0("  📐 Matryoshka enabled")
    
    # All trainable (can add dim-specific LR later)
    for param in orig_model.parameters():
        param.requires_grad = True
    
    args.matryoshka = True
    args.sample_dim = True

trainable_params = sum(p.numel() for p in orig_model.parameters() if p.requires_grad)
print0(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/num_params:.1f}%)")

# -----------------------------------------------------------------------------
# Optimizer setup
print0("Setting up optimizers...")

optimizers = orig_model.setup_optimizers(
    matrix_lr=args.matrix_lr,
    embedding_lr=args.embedding_lr,
    probe_lr=args.probe_lr,
    no_muon=args.no_muon,
)

# Get Muon optimizer for momentum scheduling
muon_optimizer = optimizers[1] if len(optimizers) > 1 and not args.no_muon else None

# -----------------------------------------------------------------------------
# Data loader
print0("Setting up data loader...")

# Synthetic data generator for smoke testing (no dataset needed)
def synthetic_data_loader():
    """Generates random token data for testing."""
    while True:
        x = torch.randint(0, 32768, (args.device_batch_size, args.max_seq_len), device=device)
        y = torch.randint(0, 32768, (args.device_batch_size, args.max_seq_len), device=device)
        yield x, y, {}

# Use synthetic data for testing, real data for actual training
try:
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    train_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device
    )
    print0("Using real dataset")
except (FileNotFoundError, AssertionError) as e:
    print0(f"Dataset not found, using synthetic data for smoke test: {e}")
    train_loader = synthetic_data_loader()

# Calculate gradient accumulation
tokens_per_batch = args.device_batch_size * args.max_seq_len
world_tokens_per_batch = tokens_per_batch * ddp_world_size
grad_accum_steps = max(1, args.total_batch_size // world_tokens_per_batch)

print0(f"Tokens per micro-batch: {tokens_per_batch:,}")
print0(f"Gradient accumulation steps: {grad_accum_steps}")
print0(f"Effective batch size: {world_tokens_per_batch * grad_accum_steps:,} tokens")

# -----------------------------------------------------------------------------
# LR scheduler
def get_lr_multiplier(step):
    """Cosine annealing with warmup."""
    if step < args.warmup_iters:
        return (step + 1) / args.warmup_iters
    decay_ratio = (step - args.warmup_iters) / (args.num_iterations - args.warmup_iters)
    return 0.1 + 0.9 * (1 + torch.cos(torch.tensor(torch.pi * decay_ratio)).item()) / 2

def get_muon_momentum(step):
    """Momentum warmup for Muon."""
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# -----------------------------------------------------------------------------
# Wandb setup
if master_process and args.run != "dummy":
    import wandb
    wandb_run = wandb.init(project="nano-fractal", name=args.run)
else:
    wandb_run = type('DummyRun', (), {'log': lambda *a, **kw: None})()

# -----------------------------------------------------------------------------
# torch.compile for faster training
if args.compile and hasattr(torch, 'compile'):
    print0(f"🚀 Compiling model with torch.compile (mode={args.compile_mode})...")
    model = torch.compile(model, mode=args.compile_mode)
    print0("   Compilation will happen on first forward pass")

# -----------------------------------------------------------------------------
# Training loop
print0("Starting training...")

# Use ADAMBA_CHECKPOINT_DIR env var if set (from setup_cache_paths.sh), else default
checkpoint_base = Path(os.environ.get("ADAMBA_CHECKPOINT_DIR", Path.home() / ".cache/nanochat/fractal_checkpoints"))
checkpoint_dir = checkpoint_base / args.run
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Track checkpoints for rotation
saved_checkpoints = []

model.train()
step = 0
smooth_loss = 0
use_synthetic = False

# Try to get first batch, fall back to synthetic if dataset not found
try:
    batch = next(train_loader)
    x, y = batch[0], batch[1]  # Handle both 2 and 3 return values
except (AssertionError, FileNotFoundError, StopIteration) as e:
    print0(f"Dataset error, switching to synthetic data: {e}")
    train_loader = synthetic_data_loader()
    batch = next(train_loader)
    x, y = batch[0], batch[1]
    use_synthetic = True

while step < args.num_iterations:
    t0 = time.time()
    total_loss = 0
    metrics = {}
    
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            if args.matryoshka:
                if args.sample_dim:
                    # Sample one dim level per batch
                    active_dim = sample_dim_level(dim_levels)
                    loss = orig_model(x, y, active_dim=active_dim)
                    energy = compute_energy_penalty(
                        torch.tensor(active_dim, device=device),
                        max_dim=config.n_embd_expanded,
                    )
                    loss = loss + args.energy_lambda * energy
                    metrics[f'dim_{active_dim}'] = loss.item()
                else:
                    # Train on all dim levels
                    loss, dim_metrics = orig_model.forward_matryoshka(
                        x, y,
                        dim_levels=dim_levels,
                        energy_lambda=args.energy_lambda,
                    )
                    metrics.update(dim_metrics)
            else:
                # Standard training at full dim
                loss = orig_model(x, y)
        
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        
        batch = next(train_loader)
        x, y = batch[0], batch[1]
    
    # Optimizer step
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    
    muon_momentum = get_muon_momentum(step)
    if muon_optimizer is not None:
        for group in muon_optimizer.param_groups:
            group["momentum"] = muon_momentum
    
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    
    # Logging
    train_loss_f = train_loss.item()
    smooth_loss = 0.9 * smooth_loss + 0.1 * train_loss_f
    
    if step % 10 == 0:
        tok_per_sec = int(args.total_batch_size / dt)
        print0(f"Step {step:05d} | Loss: {train_loss_f:.4f} (smooth: {smooth_loss:.4f}) | "
               f"LR: {lrm * args.matrix_lr:.6f} | {tok_per_sec:,} tok/s")
    
    # Wandb logging
    wandb_run.log({
        "step": step,
        "train/loss": train_loss_f,
        "train/smooth_loss": smooth_loss,
        "train/lr": lrm * args.matrix_lr,
        "train/tok_per_sec": args.total_batch_size / dt,
        **{f"train/{k}": v for k, v in metrics.items()},
    })
    
    # Evaluation (skip Matryoshka dims in Phase 1 - not trained yet)
    if args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
        model.eval()
        with torch.no_grad(), autocast_ctx:
            eval_batch = next(train_loader)
            eval_x, eval_y = eval_batch[0], eval_batch[1]
            # Phase 1: only eval at full dim (Matryoshka not trained)
            if args.phase == 1:
                eval_loss = orig_model(eval_x, eval_y)
                print0(f"  Eval loss: {eval_loss.item():.4f}")
                wandb_run.log({"eval/loss": eval_loss.item(), "step": step})
            else:
                # Phase 2+: eval at different dim levels
                for dim in dim_levels:
                    eval_loss = orig_model(eval_x, eval_y, active_dim=dim)
                    print0(f"  Eval @ dim={dim}: {eval_loss.item():.4f}")
                    wandb_run.log({f"eval/loss_dim_{dim}": eval_loss.item(), "step": step})
        model.train()
    
    # Checkpointing
    if args.save_every > 0 and step > 0 and step % args.save_every == 0:
        if master_process:
            ckpt_path = checkpoint_dir / f"model_{step:06d}.pt"
            torch.save(orig_model.state_dict(), ckpt_path)
            saved_checkpoints.append(ckpt_path)
            
            # Backup latest to separate volume (survives machine failure)
            backup_dir = Path("/root/highspeedstorage/AdambaCheckpoints")
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / "latest_checkpoint.pt"
            torch.save(orig_model.state_dict(), backup_path)
            
            # Checkpoint rotation: keep only last N
            if args.keep_checkpoints > 0 and len(saved_checkpoints) > args.keep_checkpoints:
                old_ckpt = saved_checkpoints.pop(0)
                if old_ckpt.exists():
                    old_ckpt.unlink()
                    print0(f"  Deleted old checkpoint: {old_ckpt.name}")
            
            # Save config
            config_path = checkpoint_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump({
                    "n_layer": config.n_layer,
                    "n_mamba_layer": config.n_mamba_layer,
                    "n_embd_expanded": config.n_embd_expanded,
                    "dim_levels": dim_levels,
                    "step": step,
                }, f)
            
            # Also copy config to backup
            import shutil
            shutil.copy(config_path, backup_dir / "config.json")
            
            print0(f"Saved checkpoint: {ckpt_path} + backup")
    
    step += 1

# Final checkpoint
if master_process:
    final_path = checkpoint_dir / f"model_final.pt"
    torch.save(orig_model.state_dict(), final_path)
    # Also backup to persistent storage
    backup_dir = Path("/root/highspeedstorage/AdambaCheckpoints")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_final = backup_dir / f"phase{args.phase}_final.pt"
    torch.save(orig_model.state_dict(), backup_final)
    print0(f"Training complete! Final checkpoint: {final_path}")
    print0(f"Backup saved to: {backup_final}")

if ddp:
    dist.destroy_process_group()
