import torch
import torch.nn as nn
from nanochat.moe_block import MoEConfig, MoEBlock

def test_moe_forward():
    print("Testing MoE Block...")
    torch.manual_seed(42)
    
    # Tiny config
    config = MoEConfig(
        hidden_size=64,
        intermediate_size=128,
        num_experts=4,
        experts_per_token=2,
        router_aux_loss_coef=0.1
    )
    
    model = MoEBlock(config)
    
    # Fake input: (batch=2, seq=4, hidden=64)
    x = torch.randn(2, 4, 64)
    
    # Forward pass
    output, aux_loss = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Aux loss: {aux_loss.item()}")
    
    # Check shape
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    assert aux_loss > 0, "Aux loss should be positive"
    
    # Backward pass check
    loss = output.mean() + aux_loss
    loss.backward()
    
    # Check gradients
    print("\nGradient check:")
    for name, p in model.named_parameters():
        if p.grad is not None:
             grad_norm = p.grad.norm().item()
             # Some experts might not be selected, so grad could be 0
             # But router and norm should have grads
             print(f"  {name}: {grad_norm:.6f}")
    
    print("\n✅ MoE Block Test Passed!")

if __name__ == "__main__":
    test_moe_forward()
