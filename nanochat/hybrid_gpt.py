"""
Hybrid GPT for Nano-Fractal: Interleaved Mamba + Attention with Matryoshka.

This module extends nanochat's GPT with:
1. Interleaved Mamba layers (for O(n) efficiency)
2. Matryoshka dimension scaling (Ghost → God)
3. Per-layer ConfidenceProbes for dynamic capacity
4. Energy-aware loss computation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat.flash_attention import flash_attn

# Import our new modules
from nanochat.mamba_block import MambaBlock, MambaLayer
from nanochat.confidence_probe import (
    ConfidenceProbe, LayerProbe, ThinkDetector,  # Legacy V1
    LayerDimPredictor, ConfidenceGate, MatryoshkaKVCache,  # V2
    ExpansionGate,  # V2 learnable expansion
    AdaptiveController,
)
from nanochat.matryoshka import (
    MLP_DIM_LEVELS, KV_DIM_LEVELS,
    slice_hidden, pad_hidden, sample_dim_level,
    compute_energy_penalty, MatryoshkaMLP,
)


def norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm without learnable params."""
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Returns True if layer should have Value Embedding."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    """Apply rotary positional embeddings."""
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


@dataclass
class HybridConfig:
    """Configuration for Hybrid Mamba-Attention model."""
    # Base model config
    sequence_len: int = 32768  # Extended for Mamba efficiency
    vocab_size: int = 32768
    n_layer: int = 32  # Original attention layers
    n_mamba_layer: int = 32  # New Mamba layers (interleaved)
    n_head: int = 16
    n_kv_head: int = 16
    n_embd: int = 2048  # Base dimension
    n_embd_expanded: int = 4096  # Expanded dimension for Matryoshka
    head_dim: int = 0  # 0 = auto (n_embd_expanded // n_head), or explicit for GPT-OSS
    window_pattern: str = "SSSL"
    
    # Matryoshka config
    mlp_dim_levels: List[int] = field(default_factory=lambda: [128, 512, 1024, 2048, 4096])
    kv_dim_levels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    
    # Mamba config
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    
    # Which layers are attention vs Mamba
    # Default: interleave (Mamba after each attention)
    attn_layer_indices: List[int] = field(default_factory=lambda: list(range(32)))


class MatryoshkaAttention(nn.Module):
    """
    Self-attention with Matryoshka KV dimension support.
    
    Supports dynamic KV head dimensions for memory/compute savings.
    """
    
    def __init__(self, config: HybridConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd_expanded  # Use expanded dim for projections
        self.n_embd_input = config.n_embd  # Input dimension
        # Use explicit head_dim if provided, otherwise calculate
        self.head_dim = config.head_dim if config.head_dim > 0 else self.n_embd // self.n_head
        self.kv_dim_levels = config.kv_dim_levels
        
        # Full-size projections (Matryoshka slicing done at runtime)
        # Input is n_embd, output is n_head * head_dim
        self.c_q = nn.Linear(self.n_embd_input, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd_input, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd_input, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_head * self.head_dim, self.n_embd_input, bias=False)
        
        # Value embedding gate (like nanochat)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
    
    def forward(
        self,
        x: torch.Tensor,
        ve: Optional[torch.Tensor],
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        window_size: Tuple[int, int],
        kv_cache=None,
        active_dim: Optional[int] = None,
        active_kv_dim: Optional[int] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()
        
        # Matryoshka: use active dimensions
        act_dim = active_dim if active_dim is not None else self.n_embd
        act_kv_dim = active_kv_dim if active_kv_dim is not None else self.head_dim
        
        # Slice input if needed
        if act_dim < self.n_embd:
            x = slice_hidden(x, act_dim)
        
        # Project with sliced weights
        act_head_dim = min(act_kv_dim, self.head_dim)
        
        # For Matryoshka attention, we compute at reduced head_dim
        q = self.c_q(x)[:, :, :self.n_head * act_head_dim].view(B, T, self.n_head, act_head_dim)
        k = self.c_k(x)[:, :, :self.n_kv_head * act_head_dim].view(B, T, self.n_kv_head, act_head_dim)
        v = self.c_v(x)[:, :, :self.n_kv_head * act_head_dim].view(B, T, self.n_kv_head, act_head_dim)
        
        # Value residual
        if ve is not None:
            ve_sliced = ve[:, :, :self.n_kv_head * act_head_dim].view(B, T, self.n_kv_head, act_head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[:, :, :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve_sliced
        
        # Rotary embeddings (slice to match head_dim)
        cos, sin = cos_sin
        cos = cos[:, :, :, :act_head_dim // 2]
        sin = sin[:, :, :, :act_head_dim // 2]
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        
        # Flash Attention
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
        
        # Project back
        y = y.contiguous().view(B, T, -1)
        
        # Pad y if we used reduced head_dim
        if act_head_dim < self.head_dim:
            y = pad_hidden(y, self.n_head * self.head_dim)
        
        y = self.c_proj(y)
        
        # Pad output if we used reduced dim
        if act_dim < self.n_embd:
            y = pad_hidden(y, self.n_embd)
        
        return y


class HybridBlock(nn.Module):
    """
    A block that can be either Attention or Mamba.
    
    Supports Matryoshka dimension selection.
    """
    
    def __init__(
        self,
        config: HybridConfig,
        layer_idx: int,
        is_attention: bool = True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_attention = is_attention
        self.n_embd = config.n_embd_expanded
        
        if is_attention:
            self.attn = MatryoshkaAttention(config, layer_idx)
        else:
            self.mamba = MambaBlock(
                d_model=config.n_embd_expanded,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
            )
        
        # Matryoshka MLP for both types
        self.mlp = MatryoshkaMLP(
            d_model=config.n_embd_expanded,
            dim_levels=config.mlp_dim_levels,
        )
        
        # Per-layer confidence probe
        self.probe = LayerProbe(config.n_embd_expanded, layer_idx)
    
    def forward(
        self,
        x: torch.Tensor,
        ve: Optional[torch.Tensor] = None,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        window_size: Optional[Tuple[int, int]] = None,
        kv_cache=None,
        active_dim: Optional[int] = None,
        active_kv_dim: Optional[int] = None,
        layer_outputs: Optional[List[torch.Tensor]] = None,
        use_probe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            x: Output tensor
            confidence: Confidence score from probe
        """
        # Optionally use probe to decide dimensions
        if use_probe:
            mlp_dim, kv_dim, confidence = self.probe(x, layer_outputs)
            # Use mean across sequence for this forward pass
            active_dim = int(mlp_dim.float().mean().item())
            active_kv_dim = int(kv_dim.float().mean().item())
        else:
            confidence = torch.ones(x.size(0), x.size(1), device=x.device)
        
        # Main computation
        if self.is_attention:
            residual = x
            h = norm(x)
            h = self.attn(h, ve, cos_sin, window_size, kv_cache, active_dim, active_kv_dim)
            x = residual + h
        else:
            residual = x
            h = norm(x)
            h = self.mamba(h, active_dim)
            x = residual + h
        
        # MLP
        residual = x
        h = norm(x)
        h = self.mlp(h, active_dim)
        x = residual + h
        
        return x, confidence


class HybridGPT(nn.Module):
    """
    Hybrid Mamba-Attention GPT with Matryoshka scaling.
    
    Architecture:
    - 32 Attention layers (kept from nanochat-d32)
    - 32 Mamba layers (new, interleaved)
    - Dimensions expandable from 2048 → 4096
    - Per-layer probes for dynamic capacity
    """
    
    def __init__(self, config: HybridConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        
        # Compute window sizes for attention layers
        self.window_sizes = self._compute_window_sizes(config)
        
        # Pad vocab
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        
        # Embeddings at expanded dimension
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd_expanded),
        })
        
        # Build interleaved blocks: Attention, Mamba, Attention, Mamba, ...
        self.blocks = nn.ModuleList()
        total_layers = config.n_layer + config.n_mamba_layer
        
        attn_idx = 0
        mamba_idx = 0
        for i in range(total_layers):
            # Interleave: even = attention, odd = mamba
            if i % 2 == 0 and attn_idx < config.n_layer:
                self.blocks.append(HybridBlock(config, i, is_attention=True))
                attn_idx += 1
            else:
                self.blocks.append(HybridBlock(config, i, is_attention=False))
                mamba_idx += 1
        
        # LM head at expanded dimension
        self.lm_head = nn.Linear(config.n_embd_expanded, padded_vocab_size, bias=False)
        
        # Per-layer scalars
        n_total = len(self.blocks)
        self.resid_lambdas = nn.Parameter(torch.ones(n_total))
        self.x0_lambdas = nn.Parameter(torch.zeros(n_total))
        
        # Value embeddings for attention layers
        kv_dim = config.n_kv_head * (config.n_embd_expanded // config.n_head)
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, kv_dim)
            for i, block in enumerate(self.blocks)
            if block.is_attention and has_ve(i, n_total)
        })
        
        # Think detector (legacy V1)
        self.think_detector = ThinkDetector(config.n_embd_expanded)
        
        # V2: LayerDimPredictor - predicts dims for all layers upfront
        self.dim_predictor = LayerDimPredictor(
            n_layers=len(self.blocks),
            d_model=config.n_embd_expanded,
            dim_levels=config.mlp_dim_levels,
        )
        
        # V2: ConfidenceGate - unified control for early exit + dim expansion
        self.confidence_gate = ConfidenceGate(config.n_embd_expanded)
        
        # V2: Adaptive inference settings
        self.exit_threshold = 0.95
        self.expand_threshold = 0.5
        self.min_layers_before_exit = len(self.blocks) // 4  # Exit after 25% of layers
        
        # Gradient checkpointing (saves memory by recomputing activations)
        self.gradient_checkpointing = False
        
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 2
        head_dim = config.n_embd_expanded // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int = 10000):
        device = self.transformer.wte.weight.device if hasattr(self, 'transformer') else 'meta'
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin
    
    def _compute_window_sizes(self, config: HybridConfig):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for i in range(config.n_layer + config.n_mamba_layer):
            char = pattern[i % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes
    
    @torch.no_grad()
    def init_weights(self):
        """Initialize weights."""
        # Embeddings
        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        
        # Blocks
        n_embd = self.config.n_embd_expanded
        s = (3 ** 0.5) * (n_embd ** -0.5)
        
        for block in self.blocks:
            if block.is_attention:
                nn.init.uniform_(block.attn.c_q.weight, -s, s)
                nn.init.uniform_(block.attn.c_k.weight, -s, s)
                nn.init.uniform_(block.attn.c_v.weight, -s, s)
                nn.init.zeros_(block.attn.c_proj.weight)
            else:
                block.mamba.init_weights(std=s)
            
            # MLP
            nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            nn.init.zeros_(block.mlp.c_proj.weight)
        
        # Scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.0)
        
        # Value embeddings
        for ve in self.value_embeds.values():
            nn.init.uniform_(ve.weight, -s, s)
        
        # Rotary embeddings
        head_dim = self.config.n_embd_expanded // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache=None,
        loss_reduction: str = 'mean',
        active_dim: Optional[int] = None,
        use_probes: bool = False,
        return_intermediates: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        Forward pass with Matryoshka support.
        
        Args:
            idx: Input token ids (B, T)
            targets: Target token ids for loss (B, T)
            kv_cache: Optional KV cache for inference
            loss_reduction: 'mean' or 'none'
            active_dim: Force specific dimension (None = use probes or full)
            use_probes: Use per-layer probes to decide dimensions
            return_intermediates: Return layer outputs for analysis
        
        Returns:
            loss or logits, optionally with intermediates
        """
        B, T = idx.size()
        
        # Rotary embeddings
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        
        # Embedding
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        
        # Track for probes and analysis
        layer_outputs = [] if (use_probes or return_intermediates) else None
        confidences = []
        active_dims_used = []
        
        # Forward through blocks
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            
            # Get value embedding if attention block
            ve = None
            if block.is_attention and str(i) in self.value_embeds:
                ve = self.value_embeds[str(i)](idx)
            
            # Forward block (with optional gradient checkpointing)
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                def block_forward(x_, ve_, cos_sin_, window_size_, active_dim_):
                    return block(x_, ve_, cos_sin_, window_size_, None, active_dim=active_dim_)
                x, conf = checkpoint(block_forward, x, ve, cos_sin, self.window_sizes[i], active_dim, use_reentrant=False)
            else:
                x, conf = block(
                    x, ve, cos_sin, self.window_sizes[i], kv_cache,
                    active_dim=active_dim,
                    layer_outputs=layer_outputs,
                    use_probe=use_probes,
                )
            
            confidences.append(conf)
            
            if layer_outputs is not None:
                layer_outputs.append(x.detach())
        
        x = norm(x)
        
        # LM head
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            
            if return_intermediates:
                return loss, {
                    'layer_outputs': layer_outputs,
                    'confidences': confidences,
                }
            return loss
        else:
            if return_intermediates:
                return logits, {
                    'layer_outputs': layer_outputs,
                    'confidences': confidences,
                }
            return logits
    
    def forward_matryoshka(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor,
        dim_levels: List[int] = None,
        energy_lambda: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass for Matryoshka training.
        
        Computes loss at multiple dimension levels.
        """
        if dim_levels is None:
            dim_levels = self.config.mlp_dim_levels
        
        total_loss = 0
        metrics = {}
        
        for dim in dim_levels:
            loss = self.forward(idx, targets, active_dim=dim)
            metrics[f'loss_dim_{dim}'] = loss.item()
            
            # Energy penalty
            energy = (dim / self.config.n_embd_expanded) ** 2
            total_loss = total_loss + loss + energy_lambda * energy
        
        avg_loss = total_loss / len(dim_levels)
        metrics['total_loss'] = avg_loss.item()
        
        return avg_loss, metrics
    
    def _sample_random_dims(self) -> List[int]:
        """Sample random dims for Matryoshka training (avoids causality violation)."""
        import random
        return [random.choice(self.config.mlp_dim_levels) for _ in range(len(self.blocks))]
    
    def forward_v2_train(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor,
        kv_cache=None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        V2 Training forward with NO causality violation.
        
        Uses random dims (Matryoshka dropout) instead of predictor.
        No early exit during training - all layers run.
        """
        B, T = idx.size()
        
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        
        # Random dims (no predictor leakage)
        layer_dims = self._sample_random_dims()
        
        metrics = {'layer_dims': layer_dims.copy(), 'mode': 'train'}
        
        # Forward ALL layers (no early exit during training)
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            
            ve = None
            if block.is_attention and str(i) in self.value_embeds:
                ve = self.value_embeds[str(i)](idx)
            
            x, _ = block(
                x, ve, cos_sin, self.window_sizes[i], kv_cache,
                active_dim=layer_dims[i],
            )
        
        x = norm(x)
        
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        return loss, metrics
    
    def forward_v2_inference(
        self,
        idx: torch.Tensor,
        kv_cache=None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        V2 Inference forward with adaptive compute.
        
        Uses LayerDimPredictor + gates for dynamic behavior.
        No causality violation: at inference we generate autoregressively,
        so last token IS the current token.
        """
        B, T = idx.size()
        
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        
        # Use predictor (safe at inference - last token is current)
        layer_dims = self.dim_predictor(x)
        
        metrics = {
            'layer_dims': layer_dims.copy(),
            'confidences': [],
            'exit_layer': len(self.blocks),
            'expanded': False,
            'mode': 'inference',
        }
        
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            
            ve = None
            if block.is_attention and str(i) in self.value_embeds:
                ve = self.value_embeds[str(i)](idx)
            
            x, _ = block(
                x, ve, cos_sin, self.window_sizes[i], kv_cache,
                active_dim=layer_dims[i],
            )
            
            confidence = self.confidence_gate(x)
            metrics['confidences'].append(confidence.mean().item())
            
            # Early exit (inference only)
            if i >= self.min_layers_before_exit:
                if confidence.mean() > self.exit_threshold:
                    metrics['exit_layer'] = i + 1
                    break
            
            # Expansion (inference only)
            if i >= len(self.blocks) * 0.75 and not metrics['expanded']:
                expand_prob = self.confidence_gate(x)  # Reuse confidence for now
                if expand_prob.mean() < self.expand_threshold:
                    max_dim = max(self.config.mlp_dim_levels)
                    for j in range(i + 1, len(self.blocks)):
                        layer_dims[j] = min(layer_dims[j] * 2, max_dim)
                    metrics['expanded'] = True
        
        x = norm(x)
        
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        
        return logits, metrics
    
    def forward_v2(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache=None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Unified V2 forward that routes to train or inference mode.
        
        - If targets provided → training mode (random dims, no gate)
        - If no targets → inference mode (predictor + gates)
        """
        if targets is not None:
            return self.forward_v2_train(idx, targets, kv_cache)
        else:
            return self.forward_v2_inference(idx, kv_cache)
    
    def setup_optimizers(self, **kwargs):
        """Setup optimizers (matching nanochat interface)."""
        ddp, rank, local_rank, world_size = get_dist_info()
        
        # Separate parameters - Muon only works with 2D matrices
        # ONLY include trainable params (requires_grad=True)
        matrix_params = []  # 2D only, for Muon
        non_matrix_params = []  # Non-2D, goes to AdamW
        probe_params = []
        
        for block in self.blocks:
            if block.is_attention:
                for p in block.attn.parameters():
                    if not p.requires_grad:
                        continue  # Skip frozen
                    if p.ndim == 2:
                        matrix_params.append(p)
                    else:
                        non_matrix_params.append(p)
            else:
                # Mamba has some non-2D params (conv1d, etc)
                for p in block.mamba.parameters():
                    if not p.requires_grad:
                        continue  # Skip frozen
                    if p.ndim == 2:
                        matrix_params.append(p)
                    else:
                        non_matrix_params.append(p)
            
            # MLP params
            for p in block.mlp.parameters():
                if not p.requires_grad:
                    continue  # Skip frozen
                if p.ndim == 2:
                    matrix_params.append(p)
                else:
                    non_matrix_params.append(p)
            
            probe_params.extend([p for p in block.probe.parameters() if p.requires_grad])
        
        # Filter trainable only
        embedding_params = [p for p in self.transformer.wte.parameters() if p.requires_grad]
        lm_head_params = [p for p in self.lm_head.parameters() if p.requires_grad]
        value_embeds_params = [p for p in self.value_embeds.parameters() if p.requires_grad]
        think_params = [p for p in self.think_detector.parameters() if p.requires_grad]
        scalar_params = [p for p in [self.resid_lambdas, self.x0_lambdas] if p.requires_grad]
        
        # LR scaling
        model_dim = self.config.n_embd_expanded
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        
        # AdamW groups (includes non-2D matrix params)
        adam_groups = [
            dict(params=lm_head_params, lr=kwargs.get('unembedding_lr', 0.004) * dmodel_lr_scale),
            dict(params=embedding_params, lr=kwargs.get('embedding_lr', 0.2) * dmodel_lr_scale),
            dict(params=value_embeds_params, lr=kwargs.get('embedding_lr', 0.2) * dmodel_lr_scale),
            dict(params=non_matrix_params, lr=kwargs.get('matrix_lr', 0.02) * dmodel_lr_scale),  # Non-2D from blocks
            dict(params=probe_params, lr=kwargs.get('probe_lr', 0.01)),  # Probes
            dict(params=think_params, lr=kwargs.get('probe_lr', 0.01)),  # Think detector
            dict(params=scalar_params, lr=kwargs.get('scalar_lr', 0.5) * 0.01),
        ]
        
        # Filter out empty param groups
        adam_groups = [g for g in adam_groups if g['params']]
        
        adamw_kwargs = dict(betas=kwargs.get('adam_betas', (0.8, 0.95)), eps=1e-10, weight_decay=0.0)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        
        # Muon for 2D matrices only (can be disabled with no_muon=True)
        no_muon = kwargs.get('no_muon', False)
        if matrix_params and not no_muon:
            muon_kwargs = dict(
                lr=kwargs.get('matrix_lr', 0.02),
                momentum=0.95,
                weight_decay=kwargs.get('weight_decay', 0.0),
            )
            MuonFactory = DistMuon if ddp else Muon
            muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
            optimizers = [adamw_optimizer, muon_optimizer]
        else:
            # Add matrix params to AdamW instead
            if matrix_params and no_muon:
                model_dim = self.config.n_embd_expanded
                dmodel_lr_scale = (model_dim / 768) ** -0.5
                adam_groups.append(dict(params=matrix_params, lr=kwargs.get('matrix_lr', 0.02) * dmodel_lr_scale))
                adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
            optimizers = [adamw_optimizer]
        
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        
        return optimizers
    
    def get_device(self):
        return self.transformer.wte.weight.device
    
    @torch.inference_mode()
    def generate_adaptive(
        self,
        tokens: List[int],
        max_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Generate with adaptive capacity based on probes.
        
        Yields tokens while dynamically adjusting compute.
        """
        device = self.get_device()
        rng = torch.Generator(device=device) if temperature > 0 else None
        if rng:
            rng.manual_seed(seed)
        
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        
        for _ in range(max_tokens):
            logits, intermediates = self.forward(ids, use_probes=True, return_intermediates=True)
            logits = logits[:, -1, :]
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            
            # Compute average confidence for logging
            avg_conf = torch.stack([c.mean() for c in intermediates['confidences']]).mean().item()
            
            yield token, avg_conf


# =============================================================================
# GPT-OSS MoE Configuration and Model
# =============================================================================

@dataclass
class GptOssMoEConfig:
    """Configuration for GPT-OSS 20B MoE hybrid model."""
    # Model dimensions
    hidden_size: int = 2880
    intermediate_size: int = 2880
    vocab_size: int = 201088
    
    # Layer structure
    num_hidden_layers: int = 24
    n_mamba_layers: int = 12  # Interleaved Mamba layers
    
    # Attention
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    sliding_window: int = 128
    max_position_embeddings: int = 131072
    
    # MoE
    num_experts: int = 32
    experts_per_token: int = 4
    swiglu_limit: float = 7.0
    router_aux_loss_coef: float = 0.01
    
    # Mamba
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    
    # Matryoshka (for dimension scaling)
    mlp_dim_levels: List[int] = field(default_factory=lambda: [720, 1440, 2160, 2880, 3584, 4608])
    
    # Reasoning level thresholds (maps to early exit)
    reasoning_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.5,    # Exit at 50% confidence
        "medium": 0.8,  # Exit at 80% confidence
        "high": 0.95,   # Exit at 95% confidence
    })


class HybridMoEGPT(nn.Module):
    """
    GPT-OSS 20B MoE with Mamba layers and Matryoshka scaling.
    
    Architecture:
    - 24 GPT-OSS attention+MoE layers
    - 12 interleaved Mamba layers (after every 2 attention layers)
    - Total: 36 blocks
    """
    
    def __init__(self, config: GptOssMoEConfig):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.hidden_size = config.hidden_size
        self.padded_vocab_size = ((config.vocab_size + 63) // 64) * 64
        
        # Embeddings
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(self.padded_vocab_size, config.hidden_size),
        })
        
        # Build blocks
        self.blocks = nn.ModuleList()
        num_attn_layers = config.num_hidden_layers
        num_mamba_layers = config.n_mamba_layers
        
        # Mamba insertion interval (every N attention layers)
        mamba_interval = num_attn_layers // num_mamba_layers
        
        block_idx = 0
        mamba_count = 0
        
        for i in range(num_attn_layers):
            # 1. Add MoE+Attention Block
            # Layer Pattern: Even=Sliding, Odd=Full
            is_sliding = (i % 2 == 0)
            window_size = config.sliding_window if is_sliding else config.max_position_embeddings
            
            # We use MoEBlock which needs to be updated to include Attention
            # Wait, MoEBlock currently only has MoE MLP. 
            # We need a full block that has Attn + MoE. 
            # Let's define it inline or assume MoEBlock will be updated?
            # Creating a composite block here using existing classes is safer.
            
            # Note: We need to import MoEBlock inside or assume it's available
            # Ideally MoEBlock should be "MoEFeedForward".
            # Let's assume we use HybridBlock-like structure but with MoE.
            
            # Since we can't easily change MoEBlock definition from here without tool calls,
            # Let's construct the block using submodules.
            from nanochat.moe_block import MoEConfig, MoEBlock
            
            moe_config = MoEConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                experts_per_token=config.experts_per_token,
                swiglu_limit=config.swiglu_limit,
                router_aux_loss_coef=config.router_aux_loss_coef
            )
            
            # We need an Attention+MoE block. 
            # MoEBlock in moe_block.py is just Norm -> MoE -> Residual.
            # We need Norm -> Attn -> Residual -> Norm -> MoE -> Residual.
            
            # Reusing HybridBlock logic but swapping MLP for MoE?
            # HybridBlock is hardcoded for dense MLP.
            # Ideally we'd define GptOssBlock class.
            
            # For this implementation to work, we'll define a custom block container
            self.blocks.append(GptOssBlock(config, moe_config, layer_idx=i))
            block_idx += 1
            
            # 2. Insert Mamba Block?
            if (i + 1) % mamba_interval == 0 and mamba_count < num_mamba_layers:
                self.blocks.append(MambaBlock(
                    d_model=config.hidden_size,
                    d_state=config.mamba_d_state,
                    d_conv=config.mamba_d_conv,
                    expand=config.mamba_expand
                ))
                block_idx += 1
                mamba_count += 1
        
        # Final Norm & Head
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.lm_head = nn.Linear(config.hidden_size, self.padded_vocab_size, bias=False)
        
        # Dim Predictor for Matryoshka
        self.dim_predictor = LayerDimPredictor(
            n_layers=len(self.blocks),
            d_model=config.hidden_size,
            dim_levels=config.mlp_dim_levels
        )
        
        # Learnable scalars for residual scaling
        self.resid_lambdas = nn.Parameter(torch.ones(len(self.blocks)))
        self.x0_lambdas = nn.Parameter(torch.zeros(len(self.blocks)))

    def forward(
        self,
        input_ids: torch.Tensor,
        active_dim: Optional[int] = None,
        use_probes: bool = False,
        return_intermediates: bool = False,
    ):
        # ... implementation ...
        x = self.transformer['wte'](input_ids)
        
        # Predict dimensions if dynamic
        layer_dims = None
        if use_probes:
             layer_dims, _ = self.dim_predictor(x)
        
        total_aux_loss = 0.0
        
        for i, block in enumerate(self.blocks):
            # Determine dim for this layer
            target_dim = active_dim
            if layer_dims is not None:
                # Map discrete level index to actual dim
                level_idx = layer_dims[i]
                target_dim = self.config.mlp_dim_levels[level_idx]
            
            # Forward block
            # If Mamba: just x
            # If GPT-OSS: x + aux_loss
            if isinstance(block, GptOssBlock):
                x, aux_loss = block(x, active_dim=target_dim)
                total_aux_loss += aux_loss
            else:
                x = block(x) # Mamba
                
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        if self.training:
            return logits, total_aux_loss
            
        return logits


class GptOssBlock(nn.Module):
    """
    Composition of Attention and MoE.
    """
    def __init__(self, config: GptOssMoEConfig, moe_config, layer_idx: int):
        super().__init__()
        from nanochat.moe_block import MoELayer
        
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        # Attention
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        # GPT-OSS uses head_dim=64 with 64 heads = 4096 total for Q
        # But hidden_size is 2880, so we need expanded attention dim
        attn_dim = config.num_attention_heads * config.head_dim  # 64 * 64 = 4096
        self.attn = MatryoshkaAttention(
            config=HybridConfig(
                n_embd=config.hidden_size,  # Input dim
                n_head=config.num_attention_heads,
                n_kv_head=config.num_key_value_heads,
                n_embd_expanded=attn_dim,  # Q/K/V projection output dim
                head_dim=config.head_dim,  # Explicit head_dim
            ),
            layer_idx=layer_idx
        )
        
        # MoE
        self.moe_norm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.moe = MoELayer(moe_config)
        
    def forward(self, x, active_dim=None):
        # Attention
        resid = x
        h = self.attn_norm(x)
        h = self.attn(h, active_dim=active_dim)
        x = resid + h
        
        # MoE
        resid = x
        h = self.moe_norm(x)
        h, aux_loss = self.moe(h, active_dim=active_dim)
        x = resid + h
        
        return x, aux_loss
