"""
Mamba Block wrapper for Nano-Fractal hybrid architecture.

This module provides a Mamba layer that integrates with the nanochat architecture,
supporting Matryoshka dimension scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Warning: mamba-ssm not installed. Using SSM fallback.")


def norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm without learnable parameters (matches nanochat style)."""
    return F.normalize(x, dim=-1) * (x.size(-1) ** 0.5)


class SSMFallback(nn.Module):
    """
    Simple SSM fallback when mamba-ssm is not available.
    Uses a basic linear recurrence for testing purposes.
    
    Matches the weight shapes of mamba_ssm.Mamba for checkpoint compatibility.
    """
    def __init__(self, d_model: int, d_state: int = 128, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv
        
        # Compute dt_rank (matches mamba-ssm default: ceil(d_model / 16))
        import math
        self.dt_rank = math.ceil(d_model / 16)
        
        # Match mamba-ssm layer names and shapes exactly
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )
        
        # x_proj: projects to dt, B, C (dt_rank + d_state * 2)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # dt_proj: from dt_rank to d_inner  
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            y: (B, T, D)
        """
        B, T, D = x.shape
        
        # Project and split
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # Each (B, T, d_inner)
        
        # Conv1d (transpose for conv, then back)
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # Simplified SSM (just a gated linear unit for fallback)
        y = x_conv * F.silu(z)
        
        # Output projection
        y = self.out_proj(y)
        return y


class MambaBlock(nn.Module):
    """
    Mamba block with Matryoshka dimension support.
    
    This block can operate at reduced dimensions for efficiency:
    - Ghost mode: 128 dims
    - Full mode: 4096 dims
    
    Args:
        d_model: Full model dimension (e.g., 4096)
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Use real Mamba if available, else fallback
        if HAS_MAMBA:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mamba = SSMFallback(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        
        # Optional: MLP after Mamba (like in Jamba)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model, bias=False),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        active_dim: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional Matryoshka dimension slicing.
        
        Args:
            x: Input tensor (B, T, D)
            active_dim: If set, only use first `active_dim` dimensions
            
        Returns:
            Output tensor (B, T, D) - same shape as input
        """
        B, T, D = x.shape
        
        # Matryoshka: slice to active dimensions
        if active_dim is not None and active_dim < D:
            # Slice input
            x_active = x[:, :, :active_dim]
            
            # Process at reduced dimension
            # Note: Mamba weights need to match, so we create a view
            # This requires the Mamba to be initialized at full size
            # and we slice the computation
            residual = x_active
            h = norm(x_active)
            
            # For proper Matryoshka, we'd need dimension-adjusted Mamba
            # For now, we project down, process, project up
            scale = (D / active_dim) ** 0.5
            h_padded = F.pad(h, (0, D - active_dim))
            h_out = self.mamba(h_padded)[:, :, :active_dim]
            x_active = residual + h_out
            
            # MLP at reduced dim
            residual = x_active
            h = norm(x_active)
            h_padded = F.pad(h, (0, D - active_dim))
            h_out = self.mlp(h_padded)[:, :, :active_dim] * scale
            x_active = residual + h_out
            
            # Pad back to full dimension
            output = F.pad(x_active, (0, D - active_dim))
            return output
        else:
            # Full dimension processing
            residual = x
            h = norm(x)
            h = self.mamba(h)
            x = residual + h
            
            residual = x
            h = norm(x)
            h = self.mlp(h)
            x = residual + h
            
            return x

    @torch.no_grad()
    def init_weights(self, std: float = 0.02):
        """Initialize weights with small values, but out_proj and MLP output to zeros for no-op start."""
        for name, p in self.named_parameters():
            if 'out_proj.weight' in name or 'mlp.2.weight' in name:
                # Zero-init output projections so Mamba+MLP starts as no-op
                nn.init.zeros_(p)
            elif 'weight' in name and p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=std)
            elif 'bias' in name:
                nn.init.zeros_(p)


class MambaLayer(nn.Module):
    """
    A single Mamba layer matching nanochat's Block interface.
    
    Designed to be interleaved with attention blocks.
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_embd = config.n_embd
        
        self.mamba_block = MambaBlock(
            d_model=config.n_embd,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        ve=None,  # Unused, for compatibility with attention Block
        cos_sin=None,  # Unused
        window_size=None,  # Unused
        kv_cache=None,  # Unused
        active_dim: int | None = None,
    ) -> torch.Tensor:
        """Forward pass compatible with nanochat Block interface."""
        return self.mamba_block(x, active_dim=active_dim)
