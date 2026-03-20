"""
quick_compile_timing.py — 30 steps individuales para separar compilación de steady-state
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
torch.set_float32_matmul_precision('high')

from chimera_config import ChimeraConfig
from chimera_lm import ChimeraLM

DEVICE = torch.device('cuda')
VOCAB, B, S = 4096, 2, 256

cfg = ChimeraConfig(
    d_model=256, n_layers=4, expand=2, headdim=32,
    d_state=128, bus_dim=256, max_landmarks=512, sdtm_n_heads=4,
    sdtm_d_mem=0, lr=3e-4, warmup_steps=50, max_seq_len=512,
)
model = ChimeraLM(cfg, vocab_size=VOCAB).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

print("Compiling...")
model.compile_for_training(mode='default')

print(f"\n{'Step':>4}  {'fwd ms':>8}  {'bwd ms':>8}  {'total ms':>9}  {'tok/s':>8}  note")
print("-" * 60)

for step in range(30):
    x = torch.randint(0, VOCAB, (B, S), device=DEVICE)

    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)

    e0.record()
    logits, loss, _ = model(x, labels=x, aux_weight=0.01)
    e1.record()
    loss.backward()
    opt.step(); opt.zero_grad()
    if hasattr(model, '_compiled') and model._compiled:
        model.post_compile_step()
    e2.record()
    torch.cuda.synchronize()

    fwd = e0.elapsed_time(e1)
    total = e0.elapsed_time(e2)
    bwd = total - fwd
    toks = B * S / (total / 1e3)
    note = " <-- compile" if total > 500 else ""
    print(f"{step:4d}  {fwd:8.1f}  {bwd:8.1f}  {total:9.1f}  {toks:8.0f}{note}")

# Summary: last 10 steps as steady-state
print("\n--- Steady-state (last 10 steps) ---")
