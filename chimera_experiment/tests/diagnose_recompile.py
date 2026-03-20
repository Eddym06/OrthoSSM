"""
diagnose_recompile.py — Detectar fuentes exactas de recompilación en torch.compile
==================================================================================
Ejecutar:
    cd /home/OrthoSSM/chimera_experiment
    /home/OrthoSSM/venv/bin/python tests/diagnose_recompile.py
"""
import sys, os, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch._dynamo

# ── Verbose recompilation logging ────────────────────────────────────────
torch._dynamo.config.verbose = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.repro_after = "dynamo"

# Set up logging to capture dynamo output
logging.basicConfig(level=logging.WARNING)
dynamo_log = logging.getLogger("torch._dynamo")
dynamo_log.setLevel(logging.DEBUG)

from chimera_config import ChimeraConfig
from chimera_lm import ChimeraLM

DEVICE = torch.device('cuda')
torch.set_float32_matmul_precision('high')
torch.manual_seed(42)

VOCAB = 4096
B, S = 2, 256

def main():
    cfg = ChimeraConfig(
        d_model=256, n_layers=4, expand=2, headdim=32,
        d_state=128, bus_dim=256, max_landmarks=512, sdtm_n_heads=4,
        sdtm_d_mem=0, lr=3e-4, warmup_steps=50, max_seq_len=512,
    )
    model = ChimeraLM(cfg, vocab_size=VOCAB).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print("=" * 60)
    print("  COMPILING...")
    print("=" * 60)
    model.compile_for_training(mode='default')

    print("\n" + "=" * 60)
    print("  RUNNING 5 STEPS — watching for recompilation logs")
    print("=" * 60)

    for step in range(5):
        x = torch.randint(0, VOCAB, (B, S), device=DEVICE)

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)

        t0.record()
        logits, loss, loss_dict = model(x, labels=x, aux_weight=0.01)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if hasattr(model, '_compiled') and model._compiled:
            model.post_compile_step()
        t1.record()
        torch.cuda.synchronize()

        ms = t0.elapsed_time(t1)
        toks = B * S / (ms / 1e3)
        print(f"  Step {step}: {ms:.1f} ms  ({toks:.0f} tok/s)")

    print("\n  Done. Check logs above for [RECOMPILE] or graph break messages.")


if __name__ == '__main__':
    main()
