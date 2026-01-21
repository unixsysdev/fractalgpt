"""
Fractal training script for Nano-Fractal hybrid model.

Supports:
1. Matryoshka training with dimension sampling
2. Energy penalty for compute minimization
3. Think token training
4. Per-layer probe training

Usage:
    # Local smoke test
    python -m scripts.fractal_train --depth=8 --num-iterations=20
    
    # Distributed training
    torchrun --nproc_per_node=8 -m scripts.fractal_train
"""

import argparse
import time
import os
import json
import random
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nanochat.common import (
    compute_init, get_dist_info, print0, print_banner,
    get_peak_flops, ColoredFormatter,
)
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.hybrid_gpt import HybridGPT, HybridConfig
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
parser.add_argument("--base-dim", type=int, default=2048, help="base model dimension")
parser.add_argument("--expanded-dim", type=int, default=4096, help="expanded model dimension")

# Matryoshka
parser.add_argument("--matryoshka", action="store_true", help="enable Matryoshka training")
parser.add_argument("--dim-levels", type=str, default="128,512,1024,2048,4096", help="comma-separated dim levels")
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
parser.add_argument("--save-every", type=int, default=100, help="save checkpoint every N steps")
parser.add_argument("--eval-every", type=int, default=50, help="evaluate every N steps")

# Probes
parser.add_argument("--train-probes", action="store_true", help="train per-layer probes")
parser.add_argument("--use-probes", action="store_true", help="use probes for dim decisions")

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
# Model setup
print0("Setting up model...")

dim_levels = [int(d) for d in args.dim_levels.split(",")]
print0(f"Matryoshka dim levels: {dim_levels}")

# Create config
config = HybridConfig(
    sequence_len=args.max_seq_len,
    vocab_size=32768,
    n_layer=args.depth,
    n_mamba_layer=args.n_mamba,
    n_head=args.expanded_dim // args.head_dim,
    n_kv_head=args.expanded_dim // args.head_dim,
    n_embd=args.base_dim,
    n_embd_expanded=args.expanded_dim,
    mlp_dim_levels=dim_levels,
)

# Create model
with torch.device("meta"):
    model = HybridGPT(config)

# Materialize on device
model = model.to_empty(device=device)

# Load checkpoint or init fresh
if args.checkpoint and args.checkpoint.exists():
    print0(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
else:
    print0("Initializing fresh weights...")
    model.init_weights()

# Wrap in DDP if needed
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
orig_model = model.module if ddp else model

num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")

# -----------------------------------------------------------------------------
# Optimizer setup
print0("Setting up optimizers...")

optimizers = orig_model.setup_optimizers(
    matrix_lr=args.matrix_lr,
    embedding_lr=args.embedding_lr,
    probe_lr=args.probe_lr,
)

# Get Muon optimizer for momentum scheduling
muon_optimizer = optimizers[1] if len(optimizers) > 1 else None

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
# Training loop
print0("Starting training...")

checkpoint_dir = Path.home() / ".cache/nanochat/fractal_checkpoints" / args.run
checkpoint_dir.mkdir(parents=True, exist_ok=True)

model.train()
step = 0
smooth_loss = 0
use_synthetic = False

# Try to get first batch, fall back to synthetic if dataset not found
try:
    x, y, _ = next(train_loader)
except (AssertionError, FileNotFoundError, StopIteration) as e:
    print0(f"Dataset error, switching to synthetic data: {e}")
    train_loader = synthetic_data_loader()
    x, y, _ = next(train_loader)
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
        
        x, y, _ = next(train_loader)
    
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
    
    # Evaluation
    if args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
        model.eval()
        with torch.no_grad(), autocast_ctx:
            # Evaluate at different dim levels
            eval_x, eval_y, _ = next(train_loader)
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
            
            # Save config
            config_path = checkpoint_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump({
                    "n_layer": config.n_layer,
                    "n_mamba_layer": config.n_mamba_layer,
                    "n_embd_expanded": config.n_embd_expanded,
                    "dim_levels": dim_levels,
                }, f)
            
            print0(f"Saved checkpoint: {ckpt_path}")
    
    step += 1

# Final checkpoint
if master_process:
    final_path = checkpoint_dir / f"model_final.pt"
    torch.save(orig_model.state_dict(), final_path)
    print0(f"Training complete! Final checkpoint: {final_path}")

if ddp:
    dist.destroy_process_group()
