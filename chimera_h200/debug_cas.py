
import torch
import torch.nn as nn
from cas_swarm import ChimeraAutonomousSwarm

def debug_cas_nan():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    D = 256
    E = 8
    cas = ChimeraAutonomousSwarm(d_model=D, n_experts=E, d_ff=D*2).to(device)
    
    print(f"W_up requires_grad: {cas.W_up.requires_grad}")
    print(f"W_up is leaf: {cas.W_up.is_leaf}")
    print(f"W_up mean: {cas.W_up.mean().item()}")
    print(f"W_up has nan: {torch.isnan(cas.W_up).any().item()}")
    
    # Init bias moderately to ensure activation
    nn.init.constant_(cas.probes[0].proj.bias, 1.0) 

    x = torch.randn(2, 64, D, device=device, requires_grad=True)
    out, aux = cas(x)
    
    print(f"Out mean: {out.mean().item()}")
    print(f"Out has nan: {torch.isnan(out).any().item()}")
    
    if torch.isnan(out).any():
        print("Out is NAN. Checking swarm_out...")
        # Recalculate manually to investigate
        # ...
        pass

    loss = out.mean()
    if torch.isnan(loss):
        print("Loss is NAN")
    
    loss.backward()
    
    if x.grad is not None:
        print(f"x.grad norm: {x.grad.norm().item()}")
    else:
        print("x.grad is None")

    if cas.W_up.grad is not None:
        print(f"W_up.grad norm: {cas.W_up.grad.norm().item()}")
    else:
        print("W_up.grad is None")

if __name__ == "__main__":
    debug_cas_nan()
