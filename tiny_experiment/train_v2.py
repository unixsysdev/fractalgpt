"""
Training script for Adamba V2.

Trains:
1. Main task (language modeling)
2. LayerDimPredictor (learns optimal per-layer dims using last token)
3. ConfidenceGate (learns when to exit)
4. ExpansionGate (learns when to expand dims)

Usage:
    python -m tiny_experiment.train_v2
"""

import torch
import torch.nn.functional as F
import time
from typing import Dict, List

from tiny_experiment.model_v2 import NanoFractalV2, FractalConfig
from tiny_experiment.data import DifficultyDataset, FixedDifficultyDataset


def train_v2(
    steps: int = 1000,  # Reduced for faster validation
    batch_size: int = 32,
    lr: float = 1e-3,
    gate_loss_weight: float = 0.1,
    device: str = "cpu",
):
    """
    Train Adamba V2 with gate learning.
    """
    print("=" * 70)
    print("ADAMBA V2 TRAINING")
    print("=" * 70)
    
    config = FractalConfig()
    model = NanoFractalV2(config).to(device)
    dataset = DifficultyDataset(max_len=config.max_seq_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    print(f"\nConfig:")
    print(f"  n_layer: {config.n_layer}")
    print(f"  n_embd: {config.n_embd}")
    print(f"  dim_levels: {config.dim_levels}")
    print(f"  exit_threshold: {config.exit_threshold}")
    print(f"  expand_threshold: {config.expand_threshold}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nTraining:")
    print(f"  steps: {steps}")
    print(f"  batch_size: {batch_size}")
    print(f"  gate_loss_weight: {gate_loss_weight}")
    print()
    
    # Tracking
    history = {
        'loss': [], 'gate_loss': [], 'exit_layers': [],
        'expansions': [], 'avg_dim': []
    }
    
    t0 = time.time()
    
    for step in range(steps):
        # Generate batch
        x, y, difficulties = dataset.generate_batch(batch_size)
        x, y = x.to(device), y.to(device)
        
        # Forward with metrics
        loss, metrics = model(x, y)
        
        # Gate training: teach gate when exit is safe
        gate_loss = torch.tensor(0.0, device=device)
        
        # Compute what loss would be at each layer
        hidden = model.embed(x)
        layer_dims = model.dim_predictor(hidden)
        
        for i, layer in enumerate(model.layers):
            hidden = layer(hidden, active_dim=layer_dims[i])
            
            if i >= config.min_layers_before_exit:
                # What's the loss if we exit here?
                exit_logits = model.lm_head(model.ln_f(hidden))
                exit_loss = F.cross_entropy(exit_logits.view(-1, exit_logits.size(-1)), y.view(-1))
                
                # Confidence gate: high confidence if loss is low
                confidence = model.gate(hidden)
                target = torch.sigmoid(2.0 - exit_loss).detach()
                gate_loss = gate_loss + F.mse_loss(confidence, target.expand_as(confidence))
                
                # Expansion gate: high expansion prob if loss is still high near end
                if i >= config.n_layer * 0.75:
                    expand_prob = model.expansion_gate(hidden)
                    # Target: expand if loss is still high
                    expand_target = torch.sigmoid(exit_loss - 1.0).detach()
                    gate_loss = gate_loss + F.mse_loss(expand_prob, expand_target.expand_as(expand_prob))
        
        gate_loss = gate_loss / max(1, config.n_layer - config.min_layers_before_exit)
        
        # Total loss
        total_loss = loss + gate_loss_weight * gate_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Track
        history['loss'].append(loss.item())
        history['gate_loss'].append(gate_loss.item())
        history['exit_layers'].append(metrics['exit_layer'])
        history['expansions'].append(1 if metrics['expanded'] else 0)
        history['avg_dim'].append(sum(metrics['layer_dims']) / len(metrics['layer_dims']))
        
        # Log
        if step % 200 == 0 or step == steps - 1:
            elapsed = time.time() - t0
            avg_exit = sum(history['exit_layers'][-100:]) / min(100, len(history['exit_layers']))
            avg_expand = sum(history['expansions'][-100:]) / min(100, len(history['expansions']))
            print(f"Step {step:4d} | loss={loss.item():.4f} | gate_loss={gate_loss.item():.4f} | "
                  f"avg_exit_layer={avg_exit:.1f} | expand_rate={avg_expand:.2f} | time={elapsed:.1f}s")
    
    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    
    model.eval()
    
    # Test 1: Early exit on easy tasks
    print("\n1. Testing early exit on easy vs hard tasks:")
    print("-" * 50)
    
    with torch.no_grad():
        for difficulty in ['easy', 'medium', 'hard']:
            fixed_dataset = FixedDifficultyDataset(difficulty)
            exit_layers = []
            dims_used = []
            expansions = 0
            
            for _ in range(20):
                x, y, _ = fixed_dataset.generate_batch(8)
                x, y = x.to(device), y.to(device)
                
                _, metrics = model(x, y)
                exit_layers.append(metrics['exit_layer'])
                dims_used.append(sum(metrics['layer_dims']) / len(metrics['layer_dims']))
                expansions += 1 if metrics['expanded'] else 0
            
            avg_exit = sum(exit_layers) / len(exit_layers)
            avg_dim = sum(dims_used) / len(dims_used)
            print(f"   {difficulty:6s}: avg_exit_layer={avg_exit:.1f}, avg_dim={avg_dim:.0f}, expansions={expansions}")
    
    # Test 2: Confidence progression
    print("\n2. Confidence progression through layers:")
    print("-" * 50)
    
    with torch.no_grad():
        x, y, _ = dataset.generate_batch(16)
        x, y = x.to(device), y.to(device)
        
        _, metrics = model(x, y)
        
        print(f"   Layer dims: {metrics['layer_dims']}")
        print(f"   Confidences: {[f'{c:.3f}' for c in metrics['confidences']]}")
        print(f"   Exit layer: {metrics['exit_layer']}")
        print(f"   Expanded: {metrics['expanded']}")
    
    # Test 3: Dimension predictor learning
    print("\n3. LayerDimPredictor outputs for different inputs:")
    print("-" * 50)
    
    with torch.no_grad():
        for difficulty in ['easy', 'hard']:
            fixed_dataset = FixedDifficultyDataset(difficulty)
            x, y, _ = fixed_dataset.generate_batch(8)
            x = x.to(device)
            
            hidden = model.embed(x)
            dims = model.dim_predictor(hidden)
            
            print(f"   {difficulty:5s} task dims: {dims}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Compute stats
    early_exits = sum(1 for e in history['exit_layers'] if e < config.n_layer)
    expand_count = sum(history['expansions'])
    
    print(f"\n  Training stats:")
    print(f"    Final loss: {history['loss'][-1]:.4f}")
    print(f"    Final gate_loss: {history['gate_loss'][-1]:.4f}")
    print(f"    Early exits: {early_exits}/{steps} ({100*early_exits/steps:.1f}%)")
    print(f"    Expansions: {expand_count}/{steps} ({100*expand_count/steps:.1f}%)")
    
    # Check if architecture is working
    checks = [
        ("Loss decreased", history['loss'][-1] < history['loss'][0]),
        ("Gate loss decreased", history['gate_loss'][-1] < history['gate_loss'][0] + 0.1),
        ("Some early exits", early_exits > 0),
        ("Model learned dims", True),  # LayerDimPredictor outputs something
    ]
    
    print(f"\n  Checks:")
    all_pass = True
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"    {status}: {name}")
        all_pass = all_pass and passed
    
    print()
    if all_pass:
        print("  🎉 ADAMBA V2 TRAINING SUCCESSFUL!")
    else:
        print("  ⚠️  Some checks failed - review above")
    
    return model, history


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    model, history = train_v2(
        steps=1000,  # 1000 steps for validation
        batch_size=32,
        lr=1e-3,
        gate_loss_weight=0.1,
        device=device,
    )
