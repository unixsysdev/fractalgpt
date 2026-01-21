"""
Matryoshka utilities for Nano-Fractal.

Provides dimension sampling, slicing, and loss computation
for nested representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import random


# Default dimension levels (Ghost → God)
MLP_DIM_LEVELS = [128, 512, 1024, 2048, 4096]
KV_DIM_LEVELS = [32, 64, 128, 256]


def sample_dim_level(
    levels: List[int] = MLP_DIM_LEVELS,
    min_level: int | None = None,
) -> int:
    """
    Randomly sample a dimension level for training.
    
    Args:
        levels: List of possible dimension values
        min_level: Minimum level to sample (index)
        
    Returns:
        Sampled dimension value
    """
    if min_level is not None:
        levels = levels[min_level:]
    return random.choice(levels)


def slice_hidden(
    x: torch.Tensor,
    active_dim: int,
    rescale: bool = True,
) -> torch.Tensor:
    """
    Slice hidden state to active dimensions.
    
    Args:
        x: Input tensor (..., D)
        active_dim: Number of active dimensions
        rescale: If True, rescale to preserve expected norm
        
    Returns:
        Sliced tensor (..., active_dim)
    """
    full_dim = x.size(-1)
    
    if active_dim >= full_dim:
        return x
    
    sliced = x[..., :active_dim]
    
    if rescale:
        # Rescale to preserve expected magnitude
        scale = (full_dim / active_dim) ** 0.5
        sliced = sliced * scale
    
    return sliced


def pad_hidden(
    x: torch.Tensor,
    target_dim: int,
    value: float = 0.0,
) -> torch.Tensor:
    """
    Pad hidden state back to full dimension.
    
    Args:
        x: Input tensor (..., D)
        target_dim: Target dimension
        value: Value to pad with
        
    Returns:
        Padded tensor (..., target_dim)
    """
    current_dim = x.size(-1)
    
    if current_dim >= target_dim:
        return x[..., :target_dim]
    
    pad_size = target_dim - current_dim
    return F.pad(x, (0, pad_size), value=value)


def slice_linear_weight(
    weight: torch.Tensor,
    in_dim: int | None = None,
    out_dim: int | None = None,
) -> torch.Tensor:
    """
    Slice a linear layer weight matrix for Matryoshka.
    
    Args:
        weight: Weight matrix (out_features, in_features)
        in_dim: Active input dimension (slice columns)
        out_dim: Active output dimension (slice rows)
        
    Returns:
        Sliced weight matrix
    """
    if out_dim is not None:
        weight = weight[:out_dim, :]
    if in_dim is not None:
        weight = weight[:, :in_dim]
    return weight


def matryoshka_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    active_in: int | None = None,
    active_out: int | None = None,
) -> torch.Tensor:
    """
    Apply linear transformation with Matryoshka dimension slicing.
    
    Args:
        x: Input (..., in_features)
        weight: Full weight (out_features, in_features)
        bias: Optional bias (out_features,)
        active_in: Active input dimension
        active_out: Active output dimension
        
    Returns:
        Output (..., active_out or out_features)
    """
    full_out, full_in = weight.shape
    
    # Determine active dimensions
    act_in = active_in if active_in is not None else full_in
    act_out = active_out if active_out is not None else full_out
    
    # Slice input if needed
    if act_in < full_in:
        x = x[..., :act_in]
    
    # Slice weight
    w = weight[:act_out, :act_in]
    
    # Slice bias
    b = bias[:act_out] if bias is not None else None
    
    # Apply linear
    return F.linear(x, w, b)


class MatryoshkaMLP(nn.Module):
    """
    MLP with Matryoshka dimension support.
    
    Can operate at any dimension level from the predefined set.
    """
    
    def __init__(
        self,
        d_model: int,
        d_intermediate: int | None = None,
        dim_levels: List[int] = MLP_DIM_LEVELS,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_intermediate = d_intermediate or d_model * 4
        self.dim_levels = dim_levels
        
        # Full-size weights
        self.c_fc = nn.Linear(d_model, self.d_intermediate, bias=False)
        self.c_proj = nn.Linear(self.d_intermediate, d_model, bias=False)
        
    def forward(
        self,
        x: torch.Tensor,
        active_dim: int | None = None,
    ) -> torch.Tensor:
        """
        Forward with optional Matryoshka slicing.
        
        Args:
            x: Input (B, T, D)
            active_dim: Active model dimension
            
        Returns:
            Output (B, T, D)
        """
        full_dim = self.d_model
        act_dim = active_dim if active_dim is not None else full_dim
        
        # Scale intermediate dimension proportionally
        act_intermediate = int(self.d_intermediate * (act_dim / full_dim))
        
        # Slice and compute
        x_slice = slice_hidden(x, act_dim, rescale=False)
        
        h = matryoshka_linear(
            x_slice,
            self.c_fc.weight,
            None,
            active_in=act_dim,
            active_out=act_intermediate,
        )
        
        h = F.relu(h).square()  # ReLU² like nanochat
        
        h = matryoshka_linear(
            h,
            self.c_proj.weight,
            None,
            active_in=act_intermediate,
            active_out=act_dim,
        )
        
        # Pad back if needed
        if act_dim < full_dim:
            h = pad_hidden(h, full_dim)
        
        return h


def compute_energy_penalty(
    active_dims: torch.Tensor,
    active_kv: torch.Tensor | None = None,
    max_dim: int = 4096,
    max_kv: int = 256,
) -> torch.Tensor:
    """
    Compute energy penalty for capacity usage.
    
    The "Prison" that forces the model to be lazy but correct.
    
    Args:
        active_dims: Active MLP dimensions used (B, T) or scalar
        active_kv: Active KV dimensions used
        max_dim: Maximum possible dimension
        max_kv: Maximum possible KV dimension
        
    Returns:
        Energy penalty (scalar)
    """
    # Normalize to [0, 1]
    dim_ratio = active_dims.float() / max_dim
    energy = (dim_ratio ** 2).mean()
    
    if active_kv is not None:
        kv_ratio = active_kv.float() / max_kv
        energy = energy + (kv_ratio ** 2).mean()
    
    return energy


def matryoshka_loss(
    logits_by_dim: dict[int, torch.Tensor],
    targets: torch.Tensor,
    energy_lambda: float = 0.01,
    max_dim: int = 4096,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute Matryoshka loss across multiple dimension levels.
    
    Args:
        logits_by_dim: Dict mapping dimension -> logits
        targets: Target token ids (B, T)
        energy_lambda: Weight for energy penalty
        max_dim: Maximum dimension (for normalization)
        
    Returns:
        total_loss: Combined loss
        metrics: Dict of per-level losses and energy
    """
    metrics = {}
    total_task_loss = 0
    total_energy = 0
    
    for dim, logits in logits_by_dim.items():
        # Task loss at this dimension
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        metrics[f'loss_dim_{dim}'] = loss.item()
        total_task_loss = total_task_loss + loss
        
        # Energy for this dimension
        energy = (dim / max_dim) ** 2
        total_energy = total_energy + energy
    
    num_levels = len(logits_by_dim)
    avg_task_loss = total_task_loss / num_levels
    avg_energy = total_energy / num_levels
    
    total_loss = avg_task_loss + energy_lambda * avg_energy
    
    metrics['task_loss'] = avg_task_loss.item()
    metrics['energy'] = avg_energy
    metrics['total_loss'] = total_loss.item()
    
    return total_loss, metrics
