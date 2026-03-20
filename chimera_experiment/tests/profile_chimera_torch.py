"""
profile_chimera_torch.py — Profiling detallado con torch.profiler
==================================================================

Captura timeline CUDA kernel-level de Chimera-1.3 usando torch.profiler.
Genera tabla de kernels top por tiempo GPU, con breakdown por subsistema.

Uso:
    cd /home/OrthoSSM/chimera_experiment
    /home/OrthoSSM/venv/bin/python tests/profile_chimera_torch.py

Output:
    - Tabla de top-N kernels por GPU time
    - Breakdown por categoría (Mamba2, SLR, TTT, SDTM, Archive, Bus, etc.)
    - Chrome trace en tests/perf_results/ (abrir con chrome://tracing)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

from chimera_config import ChimeraConfig
from chimera_lm import ChimeraLM

DEVICE = torch.device('cuda')
torch.set_float32_matmul_precision('high')
torch.manual_seed(42)

VOCAB_SIZE = 4096
SEQ_LEN    = 256
BATCH      = 2
N_WARMUP   = 3
N_ACTIVE   = 5

PERF_DIR = os.path.join(os.path.dirname(__file__), "perf_results")
os.makedirs(PERF_DIR, exist_ok=True)


def make_config():
    return ChimeraConfig(
        d_model=256, n_layers=4, expand=2, headdim=32,
        d_state=128, bus_dim=256, max_landmarks=512, sdtm_n_heads=4,
        sdtm_d_mem=0, lr=3e-4, warmup_steps=50, max_seq_len=512,
    )


def categorize_kernel(name: str) -> str:
    """Clasifica un kernel CUDA en subsistema de Chimera."""
    nl = name.lower()
    # Triton kernels tienen nombres específicos
    if 'diff_attn' in nl or 'flash_diff_slr' in nl or 'slr' in nl:
        return 'SLR'
    if 'lion_constrained' in nl or 'ttt_prediction' in nl or 'ttt' in nl:
        return 'TTT'
    # Mamba2 kernels
    if 'selective_scan' in nl or 'mamba' in nl or 'ssd' in nl or 'causal_conv1d' in nl:
        return 'Mamba2'
    # SDTM
    if 'sdtm' in nl or 'surprise' in nl:
        return 'SDTM'
    # Landmark archive
    if 'landmark' in nl or 'archive' in nl:
        return 'Archive'
    # Memory ops
    if 'memcpy' in nl or 'memset' in nl:
        return 'MemOps'
    # GEMM / matmul
    if any(k in nl for k in ['gemm', 'cutlass', 'cublas', 'matmul', 'ampere_', 'sm8']):
        return 'GEMM'
    # Elementwise / reduction
    if any(k in nl for k in ['elementwise', 'vectorized', 'reduce', 'softmax', 'layernorm', 'rmsnorm']):
        return 'Elementwise'
    # Optimizer
    if any(k in nl for k in ['adam', 'optimizer', 'fused_adam']):
        return 'Optimizer'
    # Triton (generic)
    if 'triton' in nl or 'kernel_' in nl:
        return 'Triton-Other'
    return 'Other'


def main():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Config: d_model=256, n_layers=4, d_state=128, bus_dim=256")
    print(f"Batch={BATCH}, SeqLen={SEQ_LEN}")
    print(f"Warmup={N_WARMUP}, Active={N_ACTIVE}")
    print()

    cfg = make_config()
    model = ChimeraLM(cfg, vocab_size=VOCAB_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.1f}M")

    # ── Warmup sin profiling ────────────────────────────────────────────────
    for _ in range(N_WARMUP):
        x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)
        logits, loss, _ = model(x, labels=x, aux_weight=0.01)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    print("Warmup done.")

    # ── Profile región activa ────────────────────────────────────────────────
    trace_path = os.path.join(PERF_DIR, "chimera_trace")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step_i in range(N_ACTIVE):
            x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)
            logits, loss, loss_dict = model(x, labels=x, aux_weight=0.01)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    torch.cuda.synchronize()

    # Exportar chrome trace
    prof.export_chrome_trace(os.path.join(PERF_DIR, "chimera_trace.json"))

    # ── Análisis: usar tabla nativa de PyTorch profiler ────────────────────
    print()
    print("=" * 80)
    print("  TOP CUDA KERNELS (sort_by=cuda_time_total)")
    print("=" * 80)

    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=40,
    ))

    print()
    print("=" * 80)
    print("  TOP SELF CUDA TIME (sort_by=self_cuda_time_total)")
    print("=" * 80)

    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=40,
    ))

    # ── Breakdown por categoría ──────────────────────────────────────────────
    print()
    print("=" * 80)
    print("  BREAKDOWN POR SUBSISTEMA CHIMERA")
    print("=" * 80)

    events = prof.key_averages()
    category_time = {}
    total_device_us = 0

    for ev in events:
        # Usamos cuda_time_total que incluye sub-calls
        t = ev.cuda_time_total  # microseconds
        if t > 0:
            cat = categorize_kernel(ev.key)
            category_time[cat] = category_time.get(cat, 0) + t
            total_device_us += t

    if total_device_us > 0:
        print(f"{'Category':<20}  {'Time (ms)':>12}  {'%':>8}")
        print("-" * 45)
        for cat, t_us in sorted(category_time.items(), key=lambda x: -x[1]):
            t_ms = t_us / 1e3
            pct = 100.0 * t_us / total_device_us
            bar = "#" * int(pct / 2)
            print(f"{cat:<20}  {t_ms:12.2f}  {pct:7.1f}%  {bar}")
        print(f"{'TOTAL':<20}  {total_device_us/1e3:12.2f}  {'100.0%':>8}")
    else:
        print("  (No CUDA timing data captured — profiler may not have CUDA events)")

    # ── Per-iteration timing ─────────────────────────────────────────────────
    print()
    if total_device_us > 0:
        total_ms = total_device_us / 1e3
        avg_iter_ms = total_ms / N_ACTIVE
        tokens_per_iter = BATCH * SEQ_LEN
        est_tok_per_s = tokens_per_iter / (avg_iter_ms / 1e3)
        print(f"Avg GPU time per training iter: {avg_iter_ms:.1f} ms")
        print(f"Est. throughput: {est_tok_per_s:.0f} tok/s (GPU-only)")

    # ── Top-N kernels con clasificación ──────────────────────────────────────
    print()
    print("=" * 80)
    print("  TOP-20 CUDA KERNELS CON CLASIFICACIÓN CHIMERA")
    print("=" * 80)

    events_with_cuda = [(ev, ev.cuda_time_total) for ev in events if ev.cuda_time_total > 0]
    events_with_cuda.sort(key=lambda x: -x[1])

    print(f"{'Rank':>4}  {'CUDA (ms)':>10}  {'Count':>6}  {'Category':<15}  {'Name'}")
    print("-" * 100)
    for i, (ev, t) in enumerate(events_with_cuda[:20], 1):
        cat = categorize_kernel(ev.key)
        name = ev.key[:65]
        print(f"{i:4d}  {t/1e3:10.3f}  {ev.count:6d}  {cat:<15}  {name}")

    # ── Bottleneck analysis ──────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("  BOTTLENECK ANALYSIS")
    print("=" * 80)

    if category_time:
        top3 = sorted(category_time.items(), key=lambda x: -x[1])[:3]
        for rank, (cat, t_us) in enumerate(top3, 1):
            pct = 100.0 * t_us / max(total_device_us, 1)
            print(f"  #{rank} bottleneck: {cat} ({pct:.1f}% del GPU time)")

        top_cat = top3[0][0]
        if top_cat == 'GEMM':
            print()
            print("  → GEMM domina: considerar FP8 (si H100+) o reducir d_model/expand")
            print("  → torch.compile(fullgraph=True) puede fusionar GEMMs pequeños")
        elif top_cat == 'SLR':
            print()
            print("  → SLR domina: verificar que el Flash-Diff kernel Triton está activo")
            print("  → Reducir sgr_top_k_frac o slr_window_size si no afecta calidad")
        elif top_cat == 'Mamba2':
            print()
            print("  → Mamba2 domina: esperado para d_state=128. Considerar chunk prefill")
            print("  → selective_scan_cuda ya es el path óptimo")
        elif top_cat == 'Elementwise':
            print()
            print("  → Elementwise domina: muchos kernels pequeños = overhead de launch")
            print("  → torch.compile puede fusionar estos en mega-kernels")
    else:
        print("  (No timing data available)")

    print()
    print(f"Chrome trace: {os.path.join(PERF_DIR, 'chimera_trace.json')}")
    print("  → Abrir con: chrome://tracing o Perfetto UI")
    print()


if __name__ == "__main__":
    main()
