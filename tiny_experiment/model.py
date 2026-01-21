"""
Tiny Fractal Experiment - Self-contained architecture validation.

This is a minimal implementation to validate:
1. Matryoshka MLP dimension scaling
2. Matryoshka KV cache scaling  
3. Mamba layer integration
4. Energy penalty learning
5. Model learns to use minimum compute

No dependencies on nanochat - completely standalone.

Usage:
    python -m tiny_experiment.train
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple
import random
import math


@dataclass
class TinyConfig:
    """Minimal config for validation."""
    vocab_size: int = 256  # Small vocab (bytes)
    n_layer: int = 2  # 1 attention + 1 mamba
    n_head: int = 4
    n_embd: int = 128  # Tiny dimension
    max_seq_len: int = 64
    
    # Matryoshka levels
    dim_levels: List[int] = None
    kv_levels: List[int] = None
    
    def __post_init__(self):
        if self.dim_levels is None:
            self.dim_levels = [16, 32, 64, 128]
        if self.kv_levels is None:
            self.kv_levels = [8, 16, 32]


# =============================================================================
# Matryoshka Utilities
# =============================================================================

def slice_hidden(x: torch.Tensor, active_dim: int) -> torch.Tensor:
    """Slice hidden state to active dimensions."""
    if active_dim >= x.size(-1):
        return x
    return x[..., :active_dim]


def pad_hidden(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Pad hidden state back to full dimension."""
    if x.size(-1) >= target_dim:
        return x
    pad_size = target_dim - x.size(-1)
    return F.pad(x, (0, pad_size))


class MatryoshkaMLP(nn.Module):
    """MLP that can operate at any dimension level."""
    
    def __init__(self, d_model: int, dim_levels: List[int]):
        super().__init__()
        self.d_model = d_model
        self.dim_levels = dim_levels
        self.d_hidden = d_model * 4
        
        self.fc1 = nn.Linear(d_model, self.d_hidden, bias=False)
        self.fc2 = nn.Linear(self.d_hidden, d_model, bias=False)
    
    def forward(self, x: torch.Tensor, active_dim: Optional[int] = None) -> torch.Tensor:
        d = active_dim or self.d_model
        
        # Scale hidden proportionally
        active_hidden = int(self.d_hidden * (d / self.d_model))
        
        # Slice input
        x_slice = slice_hidden(x, d)
        
        # Forward with sliced weights
        h = F.linear(x_slice, self.fc1.weight[:active_hidden, :d])
        h = F.relu(h) ** 2  # ReLU²
        out = F.linear(h, self.fc2.weight[:d, :active_hidden])
        
        # Pad back
        return pad_hidden(out, self.d_model)


# =============================================================================
# Matryoshka Attention (with KV scaling)
# =============================================================================

class MatryoshkaAttention(nn.Module):
    """Attention with Matryoshka dimension AND KV scaling."""
    
    def __init__(self, config: TinyConfig):
        super().__init__()
        self.n_head = config.n_head
        self.d_model = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.kv_levels = config.kv_levels
        
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        active_dim: Optional[int] = None,
        active_kv: Optional[int] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        d = active_dim or self.d_model
        kv_dim = active_kv or self.head_dim
        
        # Slice input
        x_slice = slice_hidden(x, d)
        n_head = d // (self.d_model // self.n_head)
        head_dim = d // n_head
        
        # Project with sliced weights
        q = F.linear(x_slice, self.q_proj.weight[:d, :d])
        k = F.linear(x_slice, self.k_proj.weight[:d, :d])
        v = F.linear(x_slice, self.v_proj.weight[:d, :d])
        
        # Reshape for attention
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)
        
        # KV Matryoshka: slice KV head dimension
        if kv_dim < head_dim:
            k = k[..., :kv_dim]
            v = v[..., :kv_dim]
            q = q[..., :kv_dim]  # Must match for attention
        
        # Causal attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(kv_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        
        # Pad back if we sliced KV
        if kv_dim < head_dim:
            out = F.pad(out, (0, head_dim - kv_dim))
        
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        out = F.linear(out, self.o_proj.weight[:d, :d])
        
        return pad_hidden(out, self.d_model)


# =============================================================================
# Simple Mamba Block (Pure PyTorch, no CUDA kernels)
# =============================================================================

class SimpleMamba(nn.Module):
    """
    Minimal Mamba-style block for validation.
    Uses simple SSM approximation (no CUDA kernels needed).
    """
    
    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * 2
        self.d_state = d_state
        self.d_conv = d_conv
        
        # Projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Conv1d for local context
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv, 
                              padding=d_conv-1, groups=self.d_inner)
        
        # SSM parameters (simplified)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
    
    def forward(self, x: torch.Tensor, active_dim: Optional[int] = None) -> torch.Tensor:
        B, T, D = x.shape
        d = active_dim or self.d_model
        
        x_slice = slice_hidden(x, d)
        d_inner = int(self.d_inner * (d / self.d_model))
        
        # Project
        xz = F.linear(x_slice, self.in_proj.weight[:d_inner*2, :d])
        x_proj, z = xz.chunk(2, dim=-1)
        
        # Conv (local context)
        x_conv = self.conv(x_proj.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # Simplified SSM: just learned gating (approximation)
        dt = F.softplus(self.dt_proj(x_conv)[:, :, :d_inner])
        y = x_conv * dt + x_conv * self.D[:d_inner]
        
        # Gate and project out
        y = y * F.silu(z)
        out = F.linear(y, self.out_proj.weight[:d, :d_inner])
        
        return pad_hidden(out, self.d_model)


# =============================================================================
# Confidence Probe (decides dimension level)
# =============================================================================

class ConfidenceProbe(nn.Module):
    """Small network that decides what dimension to use."""
    
    def __init__(self, d_model: int, n_levels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, n_levels),
        )
        self.n_levels = n_levels
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (level_logits, confidence)."""
        # Pool over sequence
        pooled = x.mean(dim=1)  # (B, D)
        logits = self.net(pooled)  # (B, n_levels)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        return logits, confidence


# =============================================================================
# Tiny Fractal Model
# =============================================================================

class TinyFractal(nn.Module):
    """
    Minimal hybrid model for architecture validation.
    
    Structure:
    - 1 Attention layer (with Matryoshka dim + KV)
    - 1 Mamba layer (with Matryoshka dim)
    - 1 MLP per layer (Matryoshka)
    - Confidence probes per layer
    """
    
    def __init__(self, config: TinyConfig):
        super().__init__()
        self.config = config
        
        # Embedding
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Layers
        self.attn = MatryoshkaAttention(config)
        self.mamba = SimpleMamba(config.n_embd)
        self.mlp1 = MatryoshkaMLP(config.n_embd, config.dim_levels)
        self.mlp2 = MatryoshkaMLP(config.n_embd, config.dim_levels)
        
        # Probes
        self.probe1 = ConfidenceProbe(config.n_embd, len(config.dim_levels))
        self.probe2 = ConfidenceProbe(config.n_embd, len(config.dim_levels))
        
        # Output
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed.weight
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        active_dim: Optional[int] = None,
        use_probes: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass.
        
        Returns: (loss or logits, metrics dict)
        """
        B, T = idx.shape
        x = self.embed(idx)
        
        metrics = {'dims_used': [], 'confidences': []}
        
        # Layer 1: Attention + MLP
        if use_probes:
            logits1, conf1 = self.probe1(x)
            level_idx = logits1.argmax(dim=-1).float().mean().item()
            dim1 = self.config.dim_levels[int(level_idx)]
            metrics['confidences'].append(conf1.mean().item())
        else:
            dim1 = active_dim or self.config.n_embd
        
        metrics['dims_used'].append(dim1)
        x = x + self.attn(x, active_dim=dim1)
        x = x + self.mlp1(x, active_dim=dim1)
        
        # Layer 2: Mamba + MLP
        if use_probes:
            logits2, conf2 = self.probe2(x)
            level_idx = logits2.argmax(dim=-1).float().mean().item()
            dim2 = self.config.dim_levels[int(level_idx)]
            metrics['confidences'].append(conf2.mean().item())
        else:
            dim2 = active_dim or self.config.n_embd
        
        metrics['dims_used'].append(dim2)
        x = x + self.mamba(x, active_dim=dim2)
        x = x + self.mlp2(x, active_dim=dim2)
        
        # Output
        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss, metrics
        
        return logits, metrics
    
    def compute_energy(self, dims_used: List[int]) -> float:
        """Compute energy penalty from dimensions used."""
        max_dim = self.config.n_embd
        energy = sum((d / max_dim) ** 2 for d in dims_used) / len(dims_used)
        return energy


# =============================================================================
# Test it works
# =============================================================================

if __name__ == "__main__":
    print("Testing TinyFractal components...")
    
    config = TinyConfig()
    model = TinyFractal(config)
    
    # Test forward
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    
    # Test at different dims
    for dim in config.dim_levels:
        loss, metrics = model(x, y, active_dim=dim)
        print(f"  dim={dim:3d}: loss={loss.item():.4f}, energy={model.compute_energy(metrics['dims_used']):.4f}")
    
    # Test with probes
    loss, metrics = model(x, y, use_probes=True)
    print(f"  probes: loss={loss.item():.4f}, dims={metrics['dims_used']}, conf={metrics['confidences']}")
    
    print("\n✓ All components work!")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
