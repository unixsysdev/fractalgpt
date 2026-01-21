"""
Confidence Probe for Nano-Fractal dynamic capacity allocation.

This module implements the per-layer neural network that decides
what dimension level each layer should operate at, based on
topological signals from the hidden state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConfidenceProbe(nn.Module):
    """
    Per-layer neural network that decides capacity based on hidden state topology.
    
    Uses three signals:
    1. Hidden variance (coherence proxy)
    2. Inter-layer agreement (processing status)
    3. Pairwise spread (H₀ entropy proxy)
    
    Outputs:
    - dim_level: Which Matryoshka dimension to use
    - confidence: How confident the model is (for logging/debugging)
    """
    
    # Matryoshka dimension levels
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
        
        # Input: 3 topological signals
        # Output: dim_level logits + confidence
        self.net = nn.Sequential(
            nn.Linear(3, probe_dim),
            nn.ReLU(),
            nn.Linear(probe_dim, probe_dim),
            nn.ReLU(),
            nn.Linear(probe_dim, num_dim_levels + 1),  # +1 for confidence
        )
        
        # Learnable thresholds for hard decisions during inference
        self.register_buffer(
            'dim_levels',
            torch.tensor(self.DIM_LEVELS[:num_dim_levels], dtype=torch.long)
        )
        
    def compute_signals(
        self,
        hidden: torch.Tensor,
        layer_outputs: List[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Compute topological signals from hidden state.
        
        Args:
            hidden: Current hidden state (B, T, D)
            layer_outputs: List of previous layer outputs for agreement
            
        Returns:
            signals: (B, T, 3) tensor of [variance, agreement, spread]
        """
        B, T, D = hidden.shape
        
        # Signal 1: Hidden variance (per-token)
        # Low variance = model is "coasting", high = actively processing
        variance = hidden.var(dim=-1, keepdim=True)  # (B, T, 1)
        variance = variance / (variance.mean() + 1e-8)  # Normalize
        
        # Signal 2: Inter-layer agreement
        # Compare current to early layer (if available)
        if layer_outputs is not None and len(layer_outputs) >= 2:
            early_idx = len(layer_outputs) // 4
            early = layer_outputs[early_idx]
            # Cosine similarity between early and current
            agreement = F.cosine_similarity(
                early.view(B * T, D),
                hidden.view(B * T, D),
                dim=-1
            ).view(B, T, 1)
        else:
            # No comparison available, assume neutral
            agreement = torch.ones(B, T, 1, device=hidden.device)
        
        # Signal 3: Pairwise spread (H₀ entropy proxy)
        # Sample tokens for efficiency
        if T > 16:
            indices = torch.randperm(T, device=hidden.device)[:16]
            hidden_sample = hidden[:, indices, :]
        else:
            hidden_sample = hidden
        
        # Compute mean pairwise distance
        # This approximates topological "spread" without full TDA
        hidden_flat = hidden_sample.view(B, -1, D)
        dists = torch.cdist(hidden_flat, hidden_flat, p=2)  # (B, T', T')
        spread = dists.mean(dim=(-1, -2), keepdim=True)  # (B, 1, 1)
        spread = spread.expand(B, T, 1)
        spread = spread / (spread.mean() + 1e-8)  # Normalize
        
        # Stack signals
        signals = torch.cat([variance, agreement, spread], dim=-1)  # (B, T, 3)
        return signals
    
    def forward(
        self,
        hidden: torch.Tensor,
        layer_outputs: List[torch.Tensor] | None = None,
        return_soft: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decide what dimension level to use.
        
        Args:
            hidden: Current hidden state (B, T, D)
            layer_outputs: Previous layer outputs
            return_soft: If True, return soft probabilities instead of hard choice
            
        Returns:
            dim_level: Chosen dimension (B, T) or probabilities (B, T, num_levels)
            confidence: Confidence score (B, T)
        """
        B, T, D = hidden.shape
        
        # Compute topological signals
        signals = self.compute_signals(hidden, layer_outputs)  # (B, T, 3)
        
        # Run through probe network
        out = self.net(signals)  # (B, T, num_levels + 1)
        
        # Split into logits and confidence
        dim_logits = out[:, :, :-1]  # (B, T, num_levels)
        confidence = torch.sigmoid(out[:, :, -1])  # (B, T)
        
        if return_soft:
            # Return probabilities for training (differentiable)
            dim_probs = F.softmax(dim_logits, dim=-1)
            return dim_probs, confidence
        else:
            # Return hard choice for inference
            dim_idx = dim_logits.argmax(dim=-1)  # (B, T)
            dim_level = self.dim_levels[dim_idx]  # (B, T)
            return dim_level, confidence
    
    def get_expected_dim(
        self,
        hidden: torch.Tensor,
        layer_outputs: List[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Get expected dimension using soft weighting (for training).
        
        Returns:
            expected_dim: (B, T) expected dimension value
        """
        dim_probs, _ = self.forward(hidden, layer_outputs, return_soft=True)
        
        # Weight dimensions by probabilities
        dim_values = self.dim_levels.float()  # (num_levels,)
        expected_dim = (dim_probs * dim_values).sum(dim=-1)  # (B, T)
        
        return expected_dim


class ThinkDetector(nn.Module):
    """
    Detects when the model should generate <THINK> tokens.
    
    Uses the same topological signals as ConfidenceProbe but
    outputs a binary decision.
    """
    
    def __init__(self, hidden_dim: int, probe_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(3, probe_dim),
            nn.ReLU(),
            nn.Linear(probe_dim, probe_dim),
            nn.ReLU(),
            nn.Linear(probe_dim, 1),
        )
        
        # Shared signal computation
        self.probe = ConfidenceProbe(hidden_dim)
        
    def forward(
        self,
        hidden: torch.Tensor,
        layer_outputs: List[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Decide if model should think.
        
        Returns:
            should_think: (B, T) boolean tensor
        """
        signals = self.probe.compute_signals(hidden, layer_outputs)
        logits = self.net(signals).squeeze(-1)  # (B, T)
        return logits > 0  # Binary decision
    
    def get_think_probability(
        self,
        hidden: torch.Tensor,
        layer_outputs: List[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Get probability of thinking (for training).
        
        Returns:
            think_prob: (B, T) probability values
        """
        signals = self.probe.compute_signals(hidden, layer_outputs)
        logits = self.net(signals).squeeze(-1)
        return torch.sigmoid(logits)


class LayerProbe(nn.Module):
    """
    Combined probe for a single layer.
    
    Decides:
    1. What MLP dim to use
    2. What KV dim to use
    """
    
    def __init__(self, hidden_dim: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.mlp_probe = ConfidenceProbe(
            hidden_dim,
            num_dim_levels=5,  # 128, 512, 1024, 2048, 4096
        )
        self.kv_probe = ConfidenceProbe(
            hidden_dim,
            num_dim_levels=4,  # 32, 64, 128, 256
        )
        
    def forward(
        self,
        hidden: torch.Tensor,
        layer_outputs: List[torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get dimension decisions for this layer.
        
        Returns:
            mlp_dim: (B, T) MLP dimension to use
            kv_dim: (B, T) KV dimension to use
            confidence: (B, T) overall confidence
        """
        mlp_dim, mlp_conf = self.mlp_probe(hidden, layer_outputs)
        kv_dim, kv_conf = self.kv_probe(hidden, layer_outputs)
        
        # Overall confidence is minimum of both
        confidence = torch.min(mlp_conf, kv_conf)
        
        return mlp_dim, kv_dim, confidence
