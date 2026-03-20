"""
profile_chimera.py — Script ligero para Nsight Systems profiling
=================================================================

Ejecuta 10 forward + backward passes de Chimera-1.3 Heavy-Duty
para capturar el timeline de kernels CUDA con nsys.

Uso:
    cd /home/OrthoSSM/chimera_experiment
    nsys profile -o profile_chimera -f true --stats=true \
        /home/OrthoSSM/venv/bin/python tests/profile_chimera.py

Luego analizar:
    nsys stats profile_chimera.nsys-rep
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

from chimera_config import ChimeraConfig
from chimera_lm import ChimeraLM

DEVICE = torch.device('cuda')
torch.set_float32_matmul_precision('high')
torch.manual_seed(42)

VOCAB_SIZE = 4096
SEQ_LEN    = 256
BATCH      = 2
N_ITERS    = 10  # suficiente para capturar el patrón de kernels


def make_config():
    return ChimeraConfig(
        d_model=256, n_layers=4, expand=2, headdim=32,
        d_state=128, bus_dim=256, max_landmarks=512, sdtm_n_heads=4,
        sdtm_d_mem=0, lr=3e-4, warmup_steps=50, max_seq_len=512,
    )


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: d_model=256, n_layers=4, d_state=128, bus_dim=256")
    print(f"Batch={BATCH}, SeqLen={SEQ_LEN}, Iters={N_ITERS}")
    print()

    cfg = make_config()
    model = ChimeraLM(cfg, vocab_size=VOCAB_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.1f}M")
    print(f"VRAM post-init: {torch.cuda.memory_allocated()/1e6:.0f} MB")
    print()

    # ── Warmup (2 iters sin profiling real) ──────────────────────────────────
    for i in range(2):
        x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)
        logits, loss, _ = model(x, labels=x, aux_weight=0.01)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    print("Warmup done.")

    # ── Profiling region ─────────────────────────────────────────────────────
    # nsys captura TODOS los kernels, pero la región de interés es esta:
    torch.cuda.cudart().cudaProfilerStart()

    t0 = time.perf_counter()
    for i in range(N_ITERS):
        x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)
        logits, loss, loss_dict = model(x, labels=x, aux_weight=0.01)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i == 0:
            # Imprimir breakdown de losses en la primera iteración
            print(f"Iter 0 losses: ", end="")
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    print(f"{k}={v:.4f} ", end="")
                elif hasattr(v, 'item'):
                    print(f"{k}={v.item():.4f} ", end="")
            print()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    torch.cuda.cudart().cudaProfilerStop()

    total_tokens = N_ITERS * BATCH * SEQ_LEN
    elapsed = t1 - t0
    tok_per_s = total_tokens / elapsed

    print()
    print(f"{'='*60}")
    print(f"  {N_ITERS} iters × B={BATCH} × S={SEQ_LEN} = {total_tokens} tokens")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Throughput: {tok_per_s:.0f} tok/s (train fwd+bwd+opt)")
    print(f"  Per-iter: {elapsed/N_ITERS*1000:.1f} ms")
    print(f"  VRAM peak: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
