"""
MoE Block for GPT-OSS 20B integration.

Implements Mixture-of-Experts layer matching GPT-OSS architecture:
- 32 local experts per layer
- Top-4 routing per token
- SwiGLU activation with clipping
- Load balancing auxiliary loss
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    """Configuration for MoE layers."""
    hidden_size: int = 2880
    intermediate_size: int = 2880  # Per expert
    num_experts: int = 32
    experts_per_token: int = 4
    swiglu_limit: float = 7.0
    router_aux_loss_coef: float = 0.01


def swiglu(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
    """
    SwiGLU activation with clamping (matches GPT-OSS).
    
    Input shape: (..., 2 * intermediate_size)
    Output shape: (..., intermediate_size)
    """
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    # SiLU (Swish) on gate path
    out_glu = x_glu * torch.sigmoid(1.702 * x_glu)
    # Note: GPT-OSS adds bias of 1 to linear path
    return out_glu * (x_linear + 1)


class MoERouter(nn.Module):
    """
    Top-K expert router with load balancing loss.
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.aux_loss_coef = config.router_aux_loss_coef
        
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
    
    def forward(
        self, 
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            hidden_states: (batch, seq, hidden)
            
        Returns:
            expert_weights: (batch, seq, k) normalized weights
            expert_indices: (batch, seq, k) expert indices
            aux_loss: scalar load balancing loss
        """
        # Flatten batch and seq for routing
        batch, seq, hidden = hidden_states.shape
        h_flat = hidden_states.view(-1, hidden)
        
        # Compute router logits
        router_logits = self.gate(h_flat)  # (batch*seq, num_experts)
        
        # Top-K selection
        top_k = torch.topk(router_logits, k=self.experts_per_token, dim=-1, sorted=True)
        expert_indices = top_k.indices  # (batch*seq, k)
        expert_weights = F.softmax(top_k.values, dim=-1)  # (batch*seq, k)
        
        # Compute load balancing auxiliary loss
        # This encourages uniform expert utilization
        aux_loss = self._compute_aux_loss(router_logits, expert_indices)
        
        # Reshape back
        expert_weights = expert_weights.view(batch, seq, self.experts_per_token)
        expert_indices = expert_indices.view(batch, seq, self.experts_per_token)
        
        return expert_weights, expert_indices, aux_loss
    
    def _compute_aux_loss(
        self, 
        router_logits: torch.Tensor, 
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing loss.
        
        Penalizes uneven expert utilization to prevent collapse.
        """
        num_tokens = router_logits.shape[0]
        
        # Fraction of tokens routed to each expert
        router_probs = F.softmax(router_logits, dim=-1)
        expert_mask = F.one_hot(expert_indices, self.num_experts).sum(dim=1)  # (tokens, experts)
        tokens_per_expert = expert_mask.float().sum(dim=0)  # (experts,)
        
        # Average probability assigned to each expert
        avg_prob = router_probs.mean(dim=0)  # (experts,)
        
        # Load balancing loss: dot product of utilization and probability
        # Minimized when both are uniform
        aux_loss = self.aux_loss_coef * self.num_experts * (
            (tokens_per_expert / num_tokens) * avg_prob
        ).sum()
        
        return aux_loss


class MoELayer(nn.Module):
    """
    Mixture-of-Experts feed-forward layer.
    
    Each expert is a SwiGLU MLP. Top-K experts are selected per token.
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        
        self.router = MoERouter(config)
        
        # Expert weights: (num_experts, out_features, in_features)
        # MLP1: hidden -> 2 * intermediate (for SwiGLU gate + linear)
        self.mlp1_weight = nn.Parameter(
            torch.empty(config.num_experts, config.intermediate_size * 2, config.hidden_size)
        )
        self.mlp1_bias = nn.Parameter(
            torch.empty(config.num_experts, config.intermediate_size * 2)
        )
        
        # MLP2: intermediate -> hidden
        self.mlp2_weight = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size, config.intermediate_size)
        )
        self.mlp2_bias = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize expert weights."""
        std = (3 ** 0.5) * (self.hidden_size ** -0.5)
        nn.init.normal_(self.mlp1_weight, std=std)
        nn.init.zeros_(self.mlp1_bias)
        nn.init.zeros_(self.mlp2_weight)  # Zero init for residual-safe start
        nn.init.zeros_(self.mlp2_bias)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        active_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.
        
        Args:
            hidden_states: (batch, seq, hidden)
            active_dim: Optional Matryoshka dimension (not used in MoE, kept for interface)
            
        Returns:
            output: (batch, seq, hidden)
            aux_loss: scalar load balancing loss
        """
        batch, seq, hidden = hidden_states.shape
        
        # Get routing decisions
        expert_weights, expert_indices, aux_loss = self.router(hidden_states)
        # expert_weights: (batch, seq, k)
        # expert_indices: (batch, seq, k)
        
        # Gather expert weights for selected experts
        # This is the "naive" implementation - production would use scatter/gather
        mlp1_w = self.mlp1_weight[expert_indices]  # (batch, seq, k, 2*inter, hidden)
        mlp1_b = self.mlp1_bias[expert_indices]    # (batch, seq, k, 2*inter)
        mlp2_w = self.mlp2_weight[expert_indices]  # (batch, seq, k, hidden, inter)
        mlp2_b = self.mlp2_bias[expert_indices]    # (batch, seq, k, hidden)
        
        # Expand hidden states for k experts
        h = hidden_states.unsqueeze(2)  # (batch, seq, 1, hidden)
        
        # MLP1: project up
        # (batch, seq, k, 2*inter, hidden) @ (batch, seq, k, hidden, 1) -> (batch, seq, k, 2*inter)
        h1 = torch.einsum('bskoh,bskh->bsko', mlp1_w, h.expand(-1, -1, self.experts_per_token, -1))
        h1 = h1 + mlp1_b
        
        # SwiGLU activation
        h1 = swiglu(h1, limit=self.swiglu_limit)  # (batch, seq, k, inter)
        
        # MLP2: project down
        h2 = torch.einsum('bskho,bsko->bskh', mlp2_w, h1)
        h2 = h2 + mlp2_b  # (batch, seq, k, hidden)
        
        # Weighted sum of expert outputs
        output = torch.einsum('bskh,bsk->bsh', h2, expert_weights)
        
        return output, aux_loss


class MoEBlock(nn.Module):
    """
    Full MoE block with RMSNorm and residual connection.
    
    Matches GPT-OSS TransformerBlock structure for the MLP portion.
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.norm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.moe = MoELayer(config)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        active_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual.
        
        Returns:
            output: (batch, seq, hidden)
            aux_loss: scalar load balancing loss
        """
        residual = hidden_states
        h = self.norm(hidden_states)
        h, aux_loss = self.moe(h, active_dim)
        return residual + h, aux_loss
