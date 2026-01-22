"""
Confidence Probe V2 for Nano-Fractal dynamic capacity allocation.

V2 Architecture:
1. LayerDimPredictor - predicts dims for ALL layers upfront (once)
2. ConfidenceGate - unified control for early exit + dim expansion
3. MatryoshkaKVCache - slice-down KV cache strategy

These replace the per-layer probes from V1 which caused GPU graph breaks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# Default dimension levels
MLP_DIM_LEVELS = [128, 256, 512, 1024, 2048, 4096]
KV_DIM_LEVELS = [32, 64, 128, 256]


class LayerDimPredictor(nn.Module):
    """
    Predicts per-layer dimensions from prompt embedding.
    
    Runs ONCE at the start, outputs dims for all layers.
    This is GPU-friendly: no graph breaks, shapes known upfront.
    """
    
    def __init__(
        self, 
        n_layers: int, 
        d_model: int, 
        dim_levels: List[int] = None,
        probe_dim: int = 128,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dim_levels = dim_levels or MLP_DIM_LEVELS
        self.n_levels = len(self.dim_levels)
        
        # Small MLP: d_model → n_layers * n_levels
        self.net = nn.Sequential(
            nn.Linear(d_model, probe_dim),
            nn.GELU(),
            nn.Linear(probe_dim, probe_dim),
            nn.GELU(),
            nn.Linear(probe_dim, n_layers * self.n_levels),
        )
        
        # Register dim_levels as buffer
        self.register_buffer(
            'dim_tensor', 
            torch.tensor(self.dim_levels, dtype=torch.long)
        )
    
    def forward(self, x: torch.Tensor) -> List[int]:
        """
        Predict dimensions for all layers.
        
        Args:
            x: Embedded prompt (B, T, D)
            
        Returns:
            List of dims for each layer
            
        Uses LAST TOKEN instead of mean pooling to avoid
        washing out critical instruction signals in long prompts.
        """
        # Use last token (not mean) to avoid losing signal in long prompts
        last_token = x[:, -1, :]  # (B, D)
        
        # Predict logits: (B, n_layers * n_levels)
        logits = self.net(last_token)
        logits = logits.view(-1, self.n_layers, self.n_levels)
        
        # Softmax and pick most likely dim per layer
        probs = F.softmax(logits, dim=-1)
        indices = probs.argmax(dim=-1)  # (B, n_layers)
        
        # Convert to dim values (use first batch item for consistency)
        dims = [self.dim_levels[idx.item()] for idx in indices[0]]
        
        return dims
    
    def forward_soft(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get soft (differentiable) dimension predictions for training.
        
        Returns:
            expected_dims: (B, n_layers) expected dimension per layer
        """
        last_token = x[:, -1, :]  # Use last token
        logits = self.net(last_token)
        logits = logits.view(-1, self.n_layers, self.n_levels)
        
        probs = F.softmax(logits, dim=-1)  # (B, n_layers, n_levels)
        dim_values = self.dim_tensor.float()  # (n_levels,)
        
        # Expected dim = weighted sum
        expected = (probs * dim_values).sum(dim=-1)  # (B, n_layers)
        return expected


class ConfidenceGate(nn.Module):
    """Gate for early exit decision. High confidence = exit early."""
    
    def __init__(self, d_model: int, probe_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, probe_dim),
            nn.GELU(),
            nn.Linear(probe_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence score from hidden state.
        
        Uses LAST TOKEN (not mean) for consistency with LayerDimPredictor.
        """
        last_token = x[:, -1, :]  # (B, D)
        return self.net(last_token).squeeze(-1)  # (B,)


class ExpansionGate(nn.Module):
    """
    Learnable gate for deciding when to expand dimensions.
    
    Replaces hardcoded threshold with learned decision.
    High output = model needs more capacity.
    """
    
    def __init__(self, d_model: int, probe_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, probe_dim),
            nn.GELU(),
            nn.Linear(probe_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns expansion probability in [0, 1]. High = need to expand."""
        last_token = x[:, -1, :]
        return self.net(last_token).squeeze(-1)


class MatryoshkaKVCache:
    """
    KV Cache that supports variable dimensions per token.
    
    Strategy: Store at max dim seen, slice down when querying.
    This works because Matryoshka ensures first dims are most important.
    """
    
    def __init__(self, n_layers: int, max_dim: int):
        self.n_layers = n_layers
        self.max_dim = max_dim
        
        # Per-layer caches
        self.cache_k: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
        self.cache_v: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
        self.cached_dims: List[int] = [0] * n_layers
    
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Add K, V to layer's cache. Pads to max seen dim."""
        current_dim = k.size(-1)
        
        if current_dim > self.cached_dims[layer_idx]:
            # New max dim for this layer - pad previous entries
            for i in range(len(self.cache_k[layer_idx])):
                pad_size = current_dim - self.cache_k[layer_idx][i].size(-1)
                self.cache_k[layer_idx][i] = F.pad(
                    self.cache_k[layer_idx][i], (0, pad_size)
                )
                self.cache_v[layer_idx][i] = F.pad(
                    self.cache_v[layer_idx][i], (0, pad_size)
                )
            self.cached_dims[layer_idx] = current_dim
        
        # Pad current to cached_dim if needed
        if current_dim < self.cached_dims[layer_idx]:
            pad_size = self.cached_dims[layer_idx] - current_dim
            k = F.pad(k, (0, pad_size))
            v = F.pad(v, (0, pad_size))
        
        self.cache_k[layer_idx].append(k)
        self.cache_v[layer_idx].append(v)
    
    def get(
        self, 
        layer_idx: int, 
        query_dim: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get cached K, V sliced to query_dim."""
        if not self.cache_k[layer_idx]:
            return None, None
        
        k = torch.cat(self.cache_k[layer_idx], dim=1)
        v = torch.cat(self.cache_v[layer_idx], dim=1)
        
        # Slice down to query dim (Matryoshka: first dims most important)
        return k[:, :, :query_dim], v[:, :, :query_dim]
    
    def clear(self, layer_idx: Optional[int] = None):
        """Clear cache for one or all layers."""
        if layer_idx is not None:
            self.cache_k[layer_idx] = []
            self.cache_v[layer_idx] = []
            self.cached_dims[layer_idx] = 0
        else:
            for i in range(self.n_layers):
                self.cache_k[i] = []
                self.cache_v[i] = []
                self.cached_dims[i] = 0


class ConfusionDetector(nn.Module):
    """
    Lightweight confusion detector using topological signals.
    
    No learned parameters - uses hidden state statistics.
    Based on Topology of Truth research: H₀ entropy as confusion signal.
    """
    
    def __init__(self, threshold: float = 0.7):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Detect confusion from hidden state entropy.
        
        Args:
            hidden: (B, T, D) hidden states
            
        Returns:
            is_confused: (B,) boolean tensor
        """
        # Compute softmax entropy over hidden dimension
        probs = F.softmax(hidden, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (B, T)
        
        # Average entropy over sequence
        avg_entropy = entropy.mean(dim=1)  # (B,)
        
        # Normalize by max possible entropy
        max_entropy = torch.log(torch.tensor(hidden.size(-1), dtype=hidden.dtype, device=hidden.device))
        normalized = avg_entropy / max_entropy
        
        return normalized > self.threshold


class AdaptiveController:
    """
    Stateful controller for adaptive inference.
    
    Manages:
    - Per-layer dimensions
    - Early exit decisions
    - Dimension expansion
    - KV cache
    """
    
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        dim_levels: List[int],
        exit_threshold: float = 0.95,
        expand_threshold: float = 0.5,
        min_layers_before_exit: int = 8,
    ):
        self.n_layers = n_layers
        self.d_model = d_model
        self.dim_levels = dim_levels
        self.exit_threshold = exit_threshold
        self.expand_threshold = expand_threshold
        self.min_layers_before_exit = min_layers_before_exit
        
        # State
        self.layer_dims: List[int] = [d_model] * n_layers
        self.current_layer = 0
        self.exited_early = False
        self.expanded = False
    
    def set_layer_dims(self, dims: List[int]):
        """Set dimensions from LayerDimPredictor."""
        self.layer_dims = dims.copy()
    
    def check_gate(self, confidence: float, layer_idx: int) -> str:
        """
        Check gate and return action.
        
        Returns:
            'continue': Keep going
            'exit': Exit early
            'expand': Expand remaining layers
        """
        # Early exit check
        if layer_idx >= self.min_layers_before_exit:
            if confidence > self.exit_threshold:
                self.exited_early = True
                return 'exit'
        
        # Expansion check (near end with low confidence)
        if layer_idx >= self.n_layers * 0.75:
            if confidence < self.expand_threshold and not self.expanded:
                # Double remaining layer dims
                max_dim = max(self.dim_levels)
                for j in range(layer_idx + 1, self.n_layers):
                    self.layer_dims[j] = min(self.layer_dims[j] * 2, max_dim)
                self.expanded = True
                return 'expand'
        
        return 'continue'
    
    def reset(self):
        """Reset for new sequence."""
        self.layer_dims = [self.d_model] * self.n_layers
        self.current_layer = 0
        self.exited_early = False
        self.expanded = False


# =============================================================================
# Legacy compatibility (V1 interfaces)
# =============================================================================

class ConfidenceProbe(nn.Module):
    """
    Legacy V1 per-layer probe. Kept for backward compatibility.
    
    For new code, use LayerDimPredictor + ConfidenceGate instead.
    """
    
    DIM_LEVELS = [128, 512, 1024, 2048, 4096]
    KV_LEVELS = [32, 64, 128, 256]
    
    def __init__(
        self,
        hidden_dim: int,
        num_dim_levels: int = 5,
        probe_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_dim_levels = num_dim_levels
        
        self.net = nn.Sequential(
            nn.Linear(3, probe_dim),
            nn.ReLU(),
            nn.Linear(probe_dim, probe_dim),
            nn.ReLU(),
            nn.Linear(probe_dim, num_dim_levels + 1),
        )
        
        self.register_buffer(
            'dim_levels',
            torch.tensor(self.DIM_LEVELS[:num_dim_levels], dtype=torch.long)
        )
    
    def compute_signals(
        self,
        hidden: torch.Tensor,
        layer_outputs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, T, D = hidden.shape
        
        variance = hidden.var(dim=-1, keepdim=True)
        variance = variance / (variance.mean() + 1e-8)
        
        if layer_outputs is not None and len(layer_outputs) >= 2:
            early_idx = len(layer_outputs) // 4
            early = layer_outputs[early_idx]
            agreement = F.cosine_similarity(
                early.view(B * T, D),
                hidden.view(B * T, D),
                dim=-1
            ).view(B, T, 1)
        else:
            agreement = torch.ones(B, T, 1, device=hidden.device)
        
        spread = torch.ones(B, T, 1, device=hidden.device)  # Simplified
        
        return torch.cat([variance, agreement, spread], dim=-1)
    
    def forward(
        self,
        hidden: torch.Tensor,
        layer_outputs: Optional[List[torch.Tensor]] = None,
        return_soft: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        signals = self.compute_signals(hidden, layer_outputs)
        out = self.net(signals)
        
        dim_logits = out[:, :, :-1]
        confidence = torch.sigmoid(out[:, :, -1])
        
        if return_soft:
            dim_probs = F.softmax(dim_logits, dim=-1)
            return dim_probs, confidence
        else:
            dim_idx = dim_logits.argmax(dim=-1)
            dim_level = self.dim_levels[dim_idx]
            return dim_level, confidence


class ThinkDetector(nn.Module):
    """Legacy V1 think detector. Kept for backward compatibility."""
    
    def __init__(self, hidden_dim: int, probe_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, probe_dim),
            nn.ReLU(),
            nn.Linear(probe_dim, 1),
        )
        self.probe = ConfidenceProbe(hidden_dim)
    
    def forward(
        self,
        hidden: torch.Tensor,
        layer_outputs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        signals = self.probe.compute_signals(hidden, layer_outputs)
        logits = self.net(signals).squeeze(-1)
        return logits > 0


class LayerProbe(nn.Module):
    """Legacy V1 layer probe. Kept for backward compatibility."""
    
    def __init__(self, hidden_dim: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.mlp_probe = ConfidenceProbe(hidden_dim, num_dim_levels=5)
        self.kv_probe = ConfidenceProbe(hidden_dim, num_dim_levels=4)
    
    def forward(
        self,
        hidden: torch.Tensor,
        layer_outputs: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mlp_dim, mlp_conf = self.mlp_probe(hidden, layer_outputs)
        kv_dim, kv_conf = self.kv_probe(hidden, layer_outputs)
        confidence = torch.min(mlp_conf, kv_conf)
        return mlp_dim, kv_dim, confidence
