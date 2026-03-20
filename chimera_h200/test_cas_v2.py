
import torch
import torch.nn as nn
import torch.nn.functional as F
from cas_swarm import ChimeraAutonomousSwarm, MicroProbe, DepthThreshold

def test_cas_v2_parameters():
    print("=" * 60)
    print("TEST: CAS V2 (Parameter-based Weights + Triton Hook)")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    D = 256
    E = 8
    cas = ChimeraAutonomousSwarm(d_model=D, n_experts=E, d_ff=D*2).to(device)
    
    # 1. Verify Parameters exist
    assert hasattr(cas, 'W_up'), "Missing W_up parameter"
    assert isinstance(cas.W_up, nn.Parameter), "W_up is not a Parameter"
    assert cas.W_up.shape == (E, D*2, D), f"W_up shape mismatch: {cas.W_up.shape}"
    print("PASS: Parameters initialized correctly")

    # 2. Forward Pass (Fallback Path)
    # Force high activation probability for test
    nn.init.constant_(cas.probes[0].proj.bias, 5.0) # Expert 0 will activate
    
    x = torch.randn(2, 128, D, device=device, requires_grad=True)
    out, aux = cas(x)
    print(f"Forward output shape: {out.shape}")
    print(f"Active ratio: {aux['cas_active_ratio']:.4f}")
    
    # 3. Backward Pass
    loss = out.mean() + aux['cas_budget_loss']
    if torch.isnan(loss):
        print("LOSS IS NAN")
        print(f"Budget: {aux['cas_budget_loss']}")
        print(f"Diversity: {aux['cas_diversity_loss']}")
        print(f"Balance: {aux['cas_balance_loss']}")
    
    loss.backward()
    
    # 4. Gradient Check
    assert x.grad is not None, "Input x has no gradient"
    print(f"Input gradient norm: {x.grad.norm():.4f}")
    
    assert cas.W_up.grad is not None, "W_up has no gradient"
    print(f"W_up gradient norm: {cas.W_up.grad.norm():.4f}")
    
    assert cas.W_down.grad is not None, "W_down has no gradient"
    print(f"W_down gradient norm: {cas.W_down.grad.norm():.4f}")
    
    if cas.W_gate is not None:
         assert cas.W_gate.grad is not None, "W_gate has no gradient"
         print(f"W_gate gradient norm: {cas.W_gate.grad.norm():.4f}")

    print("PASS: Backward flow through manual F.linear path")
    
    # 5. Triton Path Mock (if on CPU/weak GPU, force call to sparse_expert_dispatch with True)
    # We can't easily force it without modifying the class to accept an override, 
    # but we can call sparse_expert_dispatch directly.
    from cas_swarm import sparse_expert_dispatch
    
    # Mock confidence and mask
    T = 32
    x_small = torch.randn(T, D, device=device)
    # random active mask
    conf = torch.rand(T, E, device=device)
    mask = conf > 0.5
    conf = conf * mask.float() # Zero out inactive confidence
    
    # Note: sparse_expert_dispatch signature changed to accept Tensors not ModuleList
    # And it handles normalization internally? No, x is input.
    
    # Force use_triton_gemm=True (even if it might fail if Triton not compiled, we catch it)
    try:
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            print("Attempting Triton path directly...")
            out_triton = sparse_expert_dispatch(
                x_small, mask, 
                cas.W_up, cas.W_down, cas.W_gate,
                conf, use_triton_gemm=True
            )
            print("PASS: Triton path ran without error")
            print(f"Triton Output shape: {out_triton.shape}")
        else:
            print("SKIP: Triton path requires SM80+")
    except Exception as e:
        print(f"FAIL: Triton path raised exception: {e}")
        # traceback
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cas_v2_parameters()
