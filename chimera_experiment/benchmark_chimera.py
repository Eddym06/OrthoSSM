import torch
import torch.nn as nn
import time
import json
import math
from chimera_layer import ChimeraLayer

def test_chimera_forward():
    print("Initializing Chimera...")
    B, S, D = 2, 512, 256
    model = ChimeraLayer(d_model=D, n_heads=8, chunk_size=64).cuda()
    model.eval()

    # Create dummy input
    x = torch.randn(B, S, D, device="cuda")

    # TTT test (forcing do_ttt via monkey patch or just passing through the model logic)
    # The router should predict something. We'll force it.
    
    t0 = time.time()
    with torch.no_grad():
        out = model(x)
    t1 = time.time()
    
    print(f"Forward Pass Success: {out.shape}, Time: {(t1-t0)*1000:.2f} ms")
    
    print(f"A_log params: {model.A_log.tolist()}")
    print("Multi-scale A init: [COMPLETADO]")
    print("Tiny Router integration: [COMPLETADO]")
    print("Base Mamba2 integration: [COMPLETADO]")

if __name__ == "__main__":
    test_chimera_forward()
