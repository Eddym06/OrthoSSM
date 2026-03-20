import torch
import time
from chimera_layer import ChimeraLayer

def measure_memory_and_latency(d_model=256, n_heads=8, seq_len=512, batch_size=2):
    print(f"\n--- Benchmarking CHIMERA ---")
    print(f"B={batch_size}, S={seq_len}, D={d_model}")
    
    model = ChimeraLayer(d_model=d_model, n_heads=n_heads, chunk_size=64).cuda()
    model.train() 
    
    x = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)
    
    print("Warmup...")
    for _ in range(2):
        model(x)

    torch.cuda.reset_peak_memory_stats()
    
    print("Test...")
    t0 = time.time()
    iters = 10
    for _ in range(iters):
        model(x)
    torch.cuda.synchronize()
    t1 = time.time()
    
    avg_ms = ((t1 - t0) / iters) * 1000
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    
    print(f"Latency (avg over {iters}): {avg_ms:.2f} ms")
    print(f"Peak VRAM: {peak_mem:.2f} MB")
    print("\n--- Analisis interno ---")
    tier_idx, probs = model.router(x)
    print(f"Router probabilities (batch 0): {probs[0].tolist()}")
    print("El TTT-lite se ha simulado sobre dt_bias (Lion optimizer implementado inline).")

if __name__ == "__main__":
    measure_memory_and_latency()
