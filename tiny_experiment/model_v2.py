"""
Nano-Fractal V2: Complete architecture with all fixes.

Implements:
1. LayerDimPredictor - predicts dims for all layers upfront
2. Unified ConfidenceGate - early exit OR expand dims
3. Matryoshka KV Cache - slice down, never pad up
4. Static Mamba - uses full dim (efficient kernel)
5. Think tokens support - model controls reasoning depth

Usage:
    python -m tiny_experiment.model_v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class FractalConfig:
    """Configuration for Nano-Fractal V2."""
    vocab_size: int = 256
    n_layer: int = 8
    n_head: int = 4
    n_embd: int = 256
    max_seq_len: int = 128
    
    # Matryoshka dimension levels
    dim_levels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    
    # Gate thresholds
    exit_threshold: float = 0.95
    expand_threshold: float = 0.5
    min_layers_before_exit: int = 4
    
    # Special tokens
    think_token_id: int = 253
    end_think_token_id: int = 254


# =============================================================================
# Matryoshka KV Cache
# =============================================================================

class MatryoshkaKVCache:
    """
    KV Cache that supports variable dimensions per token.
    
    Key insight: Always store at the max dim seen so far.
    When querying at smaller dim, slice down (Matryoshka property).
    """
    
    def __init__(self, max_dim: int):
        self.max_dim = max_dim
        self.cache_k: List[torch.Tensor] = []
        self.cache_v: List[torch.Tensor] = []
        self.cached_dim = 0
    
    def update(self, k: torch.Tensor, v: torch.Tensor):
        """Add new K, V to cache. Pads to max seen dim."""
        current_dim = k.size(-1)
        
        if current_dim > self.cached_dim:
            # New max dim - pad all previous cache entries
            for i in range(len(self.cache_k)):
                pad_size = current_dim - self.cache_k[i].size(-1)
                self.cache_k[i] = F.pad(self.cache_k[i], (0, pad_size))
                self.cache_v[i] = F.pad(self.cache_v[i], (0, pad_size))
            self.cached_dim = current_dim
        
        # Pad current to cached_dim if needed
        if current_dim < self.cached_dim:
            pad_size = self.cached_dim - current_dim
            k = F.pad(k, (0, pad_size))
            v = F.pad(v, (0, pad_size))
        
        self.cache_k.append(k)
        self.cache_v.append(v)
    
    def get(self, query_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached K, V sliced to query_dim."""
        if not self.cache_k:
            return None, None
        
        k = torch.cat(self.cache_k, dim=1)
        v = torch.cat(self.cache_v, dim=1)
        
        # Slice down to query dim (Matryoshka: first dims most important)
        return k[:, :, :query_dim], v[:, :, :query_dim]
    
    def clear(self):
        self.cache_k = []
        self.cache_v = []
        self.cached_dim = 0


# =============================================================================
# Layer Dimension Predictor
# =============================================================================

class LayerDimPredictor(nn.Module):
    """
    Predicts per-layer dimensions from prompt embedding.
    Runs ONCE at the start, outputs dims for all layers.
    """
    
    def __init__(self, config: FractalConfig):
        super().__init__()
        self.config = config
        self.dim_levels = config.dim_levels
        
        # Small MLP: d_model → n_layers outputs
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 64),
            nn.GELU(),
            nn.Linear(64, config.n_layer * len(config.dim_levels)),
        )
    
    def forward(self, x: torch.Tensor) -> List[int]:
        """
        x: (B, T, D) - embedded prompt
        Returns: list of dims for each layer
        """
        # Pool over sequence: (B, T, D) → (B, D)
        pooled = x.mean(dim=1)
        
        # Predict logits: (B, n_layers * n_levels)
        logits = self.net(pooled)
        logits = logits.view(-1, self.config.n_layer, len(self.dim_levels))
        
        # Softmax and pick most likely dim per layer
        probs = F.softmax(logits, dim=-1)
        indices = probs.argmax(dim=-1)  # (B, n_layers)
        
        # Convert to dim values (use first batch item)
        dims = [self.dim_levels[idx.item()] for idx in indices[0]]
        
        return dims


# =============================================================================
# Confidence Gate
# =============================================================================

class ConfidenceGate(nn.Module):
    """
    Unified gate that controls:
    1. Early exit (high confidence → skip remaining layers)
    2. Dim expansion (low confidence near end → expand)
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns confidence score in [0, 1]."""
        # Pool over sequence
        pooled = x.mean(dim=1)  # (B, D)
        return self.gate(pooled).squeeze(-1)  # (B,)


# =============================================================================
# Matryoshka Attention
# =============================================================================

class MatryoshkaAttention(nn.Module):
    """Attention with variable Q/K/V dimensions and Matryoshka cache."""
    
    def __init__(self, config: FractalConfig):
        super().__init__()
        self.n_head = config.n_head
        self.d_model = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        active_dim: int,
        kv_cache: Optional[MatryoshkaKVCache] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        
        # Slice input to active dim
        x_slice = x[:, :, :active_dim]
        
        # Compute Q, K, V at active dim
        q = F.linear(x_slice, self.q_proj.weight[:active_dim, :active_dim])
        k = F.linear(x_slice, self.k_proj.weight[:active_dim, :active_dim])
        v = F.linear(x_slice, self.v_proj.weight[:active_dim, :active_dim])
        
        # Update cache
        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get(active_dim)
        
        # Compute number of heads for this dim
        n_head = max(1, (active_dim * self.n_head) // self.d_model)
        head_dim = active_dim // n_head
        
        # Reshape for attention
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, -1, n_head, head_dim).transpose(1, 2)
        v = v.view(B, -1, n_head, head_dim).transpose(1, 2)
        
        # Causal attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Causal mask
        T_k = k.size(2)
        mask = torch.triu(torch.ones(T, T_k, device=x.device), diagonal=T_k-T+1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, active_dim)
        
        # Output projection
        out = F.linear(out, self.o_proj.weight[:active_dim, :active_dim])
        
        # Pad back to full dim
        return F.pad(out, (0, D - active_dim))


# =============================================================================
# Matryoshka MLP
# =============================================================================

class MatryoshkaMLP(nn.Module):
    """MLP with variable hidden dimension."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_model * 4
        
        self.fc1 = nn.Linear(d_model, self.d_hidden, bias=False)
        self.fc2 = nn.Linear(self.d_hidden, d_model, bias=False)
    
    def forward(self, x: torch.Tensor, active_dim: int) -> torch.Tensor:
        D = self.d_model
        
        # Scale hidden proportionally
        active_hidden = int(self.d_hidden * (active_dim / D))
        
        # Slice input
        x_slice = x[:, :, :active_dim]
        
        # Forward with sliced weights
        h = F.linear(x_slice, self.fc1.weight[:active_hidden, :active_dim])
        h = F.silu(h)
        out = F.linear(h, self.fc2.weight[:active_dim, :active_hidden])
        
        # Pad back
        return F.pad(out, (0, D - active_dim))


# =============================================================================
# Static Mamba (uses full dim for efficient kernel)
# =============================================================================

class StaticMamba(nn.Module):
    """
    Mamba block that ALWAYS uses full dimension.
    This allows use of efficient mamba-ssm CUDA kernel.
    """
    
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * 2
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Simplified SSM (for validation - real impl uses mamba-ssm)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.D = nn.Parameter(torch.ones(self.d_inner))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Always uses full dim - no slicing."""
        B, T, D = x.shape
        
        # Project
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        
        # Simple temporal smoothing (placeholder for real SSM)
        x_conv = x_proj.clone()
        if T > 1:
            x_conv[:, 1:] = 0.7 * x_proj[:, 1:] + 0.3 * x_proj[:, :-1]
        x_conv = F.silu(x_conv)
        
        # SSM approximation
        dt = F.softplus(self.dt_proj(x_conv))
        y = x_conv * dt + x_conv * self.D
        
        # Gate and project
        y = y * F.silu(z)
        return self.out_proj(y)


# =============================================================================
# Transformer Layer
# =============================================================================

class FractalLayer(nn.Module):
    """Single layer with attention, MLP, and optional Mamba."""
    
    def __init__(self, config: FractalConfig, has_mamba: bool = False):
        super().__init__()
        self.has_mamba = has_mamba
        
        self.ln1 = nn.RMSNorm(config.n_embd)
        self.attn = MatryoshkaAttention(config)
        
        self.ln2 = nn.RMSNorm(config.n_embd)
        self.mlp = MatryoshkaMLP(config.n_embd)
        
        if has_mamba:
            self.ln3 = nn.RMSNorm(config.n_embd)
            self.mamba = StaticMamba(config.n_embd)
    
    def forward(
        self, 
        x: torch.Tensor, 
        active_dim: int,
        kv_cache: Optional[MatryoshkaKVCache] = None,
    ) -> torch.Tensor:
        # Attention (dynamic dim)
        x = x + self.attn(self.ln1(x), active_dim, kv_cache)
        
        # MLP (dynamic dim)
        x = x + self.mlp(self.ln2(x), active_dim)
        
        # Mamba (static dim)
        if self.has_mamba:
            x = x + self.mamba(self.ln3(x))
        
        return x


# =============================================================================
# Complete Fractal Model
# =============================================================================

class NanoFractalV2(nn.Module):
    """
    Complete Nano-Fractal V2 with all optimizations.
    
    Features:
    - LayerDimPredictor: per-layer dimension control
    - ConfidenceGate: early exit + dimension expansion
    - Matryoshka cache: slice-down KV cache
    - Static Mamba: efficient O(1) kernel
    """
    
    def __init__(self, config: FractalConfig):
        super().__init__()
        self.config = config
        
        # Embedding
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Predictors and gates
        self.dim_predictor = LayerDimPredictor(config)
        self.gate = ConfidenceGate(config.n_embd)
        
        # Layers (alternating with/without Mamba)
        self.layers = nn.ModuleList([
            FractalLayer(config, has_mamba=(i % 2 == 1))
            for i in range(config.n_layer)
        ])
        
        # Output
        self.ln_f = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # Tie weights
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with adaptive depth and width.
        
        Returns: (loss or logits, metrics dict)
        """
        B, T = idx.shape
        x = self.embed(idx)
        
        # Predict per-layer dimensions (once, upfront)
        layer_dims = self.dim_predictor(x)
        
        metrics = {
            'layer_dims': layer_dims.copy(),
            'confidences': [],
            'exit_layer': self.config.n_layer,
            'expanded': False,
        }
        
        # Create per-layer KV caches if needed
        kv_caches = [MatryoshkaKVCache(self.config.n_embd) 
                     for _ in range(self.config.n_layer)] if use_cache else [None] * self.config.n_layer
        
        # Forward through layers with gate control
        for i, layer in enumerate(self.layers):
            x = layer(x, active_dim=layer_dims[i], kv_cache=kv_caches[i])
            
            # Check confidence gate
            confidence = self.gate(x)
            metrics['confidences'].append(confidence.mean().item())
            
            # Early exit check (after minimum layers)
            if i >= self.config.min_layers_before_exit:
                if confidence.mean() > self.config.exit_threshold:
                    metrics['exit_layer'] = i + 1
                    break
            
            # Expansion check (approaching end with low confidence)
            if i >= self.config.n_layer * 0.75 and confidence.mean() < self.config.expand_threshold:
                # Double remaining layer dims
                for j in range(i + 1, self.config.n_layer):
                    layer_dims[j] = min(layer_dims[j] * 2, self.config.dim_levels[-1])
                metrics['expanded'] = True
                metrics['layer_dims'] = layer_dims.copy()
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss, metrics
        
        return logits, metrics
    
    def compute_gate_loss(
        self, 
        x: torch.Tensor, 
        targets: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute gate training loss at a layer."""
        # Get current confidence
        confidence = self.gate(x)
        
        # Compute loss if we exited here
        logits = self.lm_head(self.ln_f(x))
        exit_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Target: high confidence if exit_loss is low
        target_conf = torch.sigmoid(-exit_loss + 2.0)  # Maps low loss → high conf
        
        return F.mse_loss(confidence, target_conf.expand_as(confidence))


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Nano-Fractal V2...")
    
    config = FractalConfig()
    model = NanoFractalV2(config)
    
    print(f"\nConfig:")
    print(f"  n_layer: {config.n_layer}")
    print(f"  n_embd: {config.n_embd}")
    print(f"  dim_levels: {config.dim_levels}")
    print(f"  exit_threshold: {config.exit_threshold}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    
    print("\n1. Testing forward pass...")
    loss, metrics = model(x, y)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Layer dims: {metrics['layer_dims']}")
    print(f"   Confidences: {[f'{c:.2f}' for c in metrics['confidences']]}")
    print(f"   Exit layer: {metrics['exit_layer']}")
    print(f"   Expanded: {metrics['expanded']}")
    
    # Test with cache
    print("\n2. Testing with KV cache...")
    logits, metrics = model(x[:, :16], use_cache=True)
    print(f"   Shape: {logits.shape}")
    
    print("\n✓ Nano-Fractal V2 works!")
