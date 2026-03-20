#!/usr/bin/env python3
"""
CHIMERA Deep Benchmark Suite
==============================
Evaluación honesta y profunda de AdvancedChimeraLayer:

 T1.  Throughput vs Mamba2 puro — tokens/s, ms/fwd, VRAM por longitud
 T2.  Desglose de latencia por componente (overhead real de cada módulo)
 T3.  Latencia de decode token-by-token (step())
 T4.  Calidad de adaptación TTT — ¿cambia dt_bias? ¿cuánto?
 T5.  Estabilidad de routing — colapso de tiers, entropía H(probs)
 T6.  Tarea de copy/retrieval — señal SLR + archive vs Mamba2 base
 T7.  Flujo de gradientes — normas por componente, detectar vanish/explode
 T8.  Escalado de memoria — VRAM pico vs secuencia
 T9.  Comportamiento del warm-up — gates y lr durante 3 fases
 T10. Stress test — S=4096, B=4, backward completo; ¿OOM? ¿NaN?

Hardware objetivo: RTX 4050 6GB (SM89), CUDA 12.4, BF16
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, gc, json
from copy import deepcopy

# ─── Terminal color helpers ────────────────────────────────────────────────────
PASS  = "\033[92m✓\033[0m"
FAIL  = "\033[91m✗\033[0m"
WARN  = "\033[93m⚠\033[0m"
BOLD  = "\033[1m"
RESET = "\033[0m"
SEP   = "=" * 72

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Usamos float32 coherente con los pesos por defecto del modelo.
# En producción con AMP se usaría bfloat16 con autocast + model.half().
# Para esta evaluación honesta queremos evitar mismatches dtype silenciosos.
DTYPE  = torch.float32

results: dict = {}          # acumula todos los resultados para JSON final


# ─── Utilidades ───────────────────────────────────────────────────────────────

def gpu_reset():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def peak_mb() -> float:
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0

def current_mb() -> float:
    if DEVICE == "cuda":
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0

def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()

def banner(title: str):
    print(f"\n{SEP}\n{BOLD}{title}{RESET}\n{SEP}")

def ok(msg: str):   print(f"  {PASS} {msg}")
def warn(msg: str): print(f"  {WARN} {msg}")
def fail(msg: str): print(f"  {FAIL} {msg}")

def entropy(probs: torch.Tensor) -> float:
    """Shannon entropy H(p) en bits."""
    p = probs.clamp(min=1e-9)
    return -(p * p.log() / math.log(2)).sum(dim=-1).mean().item()


# ─── Mamba2 minimal (baseline limpio para comparación) ────────────────────────

class Mamba2Baseline(nn.Module):
    """Single Mamba2 layer con RMSNorm + residual — mismo param count aprox."""
    def __init__(self, d_model=256, expand=2, headdim=32):
        super().__init__()
        from mamba_ssm import Mamba2
        self.norm   = nn.RMSNorm(d_model)
        self.mamba2 = Mamba2(d_model=d_model, expand=expand,
                             headdim=headdim, d_state=64)

    def forward(self, x):
        return x + self.mamba2(self.norm(x))


# ─── Importar CHIMERA ─────────────────────────────────────────────────────────

print(f"{BOLD}Cargando AdvancedChimeraLayer...{RESET}")
from advanced_chimera import AdvancedChimeraLayer
from chimera_losses import ChimeraLosses
from chimera_scheduler import ChimeraWarmupScheduler
print("  OK\n")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Throughput vs Mamba2 puro
# ══════════════════════════════════════════════════════════════════════════════

def test_throughput():
    banner("T1 · THROUGHPUT vs MAMBA2  (tokens/s, ms/fwd, VRAM pico)")

    D         = 256
    B         = 2
    LENGTHS   = [256, 512, 1024, 2048, 4096]
    N_WARM    = 4
    N_ITER    = 10

    chimera  = AdvancedChimeraLayer(d_model=D, expand=2, headdim=32).to(DEVICE)
    mamba_bl = Mamba2Baseline(d_model=D, expand=2, headdim=32).to(DEVICE)
    chimera.eval()
    mamba_bl.eval()

    n_chimera = sum(p.numel() for p in chimera.parameters())
    n_mamba   = sum(p.numel() for p in mamba_bl.parameters())
    print(f"  Parámetros  CHIMERA: {n_chimera:,}   Mamba2: {n_mamba:,}")
    print(f"  Ratio: {n_chimera/n_mamba:.2f}x\n")

    t1_results = []
    fmt_header = f"  {'S':>5}  {'CHIMERA ms':>11}  {'Mamba2 ms':>10}  "
    fmt_header += f"{'overhead':>9}  {'CHI tok/s':>10}  {'MB2 tok/s':>10}  {'VRAM chim MB':>13}"
    print(fmt_header)
    print("  " + "-" * 80)

    for S in LENGTHS:
        x = torch.randn(B, S, D, device=DEVICE, dtype=DTYPE)

        # Warm-up
        with torch.no_grad():
            for _ in range(N_WARM):
                _ = chimera(x, bus_cache=None)
                _ = mamba_bl(x)
        sync()

        # CHIMERA
        gpu_reset()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            sync(); t0 = time.perf_counter()
            for _ in range(N_ITER):
                _ = chimera(x, bus_cache=None)
            sync(); t_chi = (time.perf_counter() - t0) / N_ITER * 1000
        vram_chi = peak_mb()

        # Mamba2
        gpu_reset()
        with torch.no_grad():
            sync(); t0 = time.perf_counter()
            for _ in range(N_ITER):
                _ = mamba_bl(x)
            sync(); t_mb2 = (time.perf_counter() - t0) / N_ITER * 1000

        overhead   = t_chi / max(t_mb2, 0.01)
        toks_chi   = B * S / (t_chi / 1000)
        toks_mb2   = B * S / (t_mb2 / 1000)

        row = dict(S=S, chimera_ms=round(t_chi,2), mamba2_ms=round(t_mb2,2),
                   overhead=round(overhead,2), chi_toks=int(toks_chi),
                   mb2_toks=int(toks_mb2), vram_chi_mb=round(vram_chi,1))
        t1_results.append(row)

        flag = PASS if overhead < 5.0 else WARN
        print(f"  {S:>5}  {t_chi:>10.2f}ms  {t_mb2:>9.2f}ms  "
              f"{overhead:>8.2f}x  {int(toks_chi):>10,}  {int(toks_mb2):>10,}  "
              f"{vram_chi:>12.1f}  {flag}")

    results["T1_throughput"] = t1_results
    del chimera, mamba_bl
    gpu_reset()


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Desglose de latencia por componente
# ══════════════════════════════════════════════════════════════════════════════

def test_latency_breakdown():
    banner("T2 · DESGLOSE DE LATENCIA POR COMPONENTE  (S=512, B=2)")

    D, B, S = 256, 2, 512
    N_ITER  = 20

    model = AdvancedChimeraLayer(d_model=D, expand=2, headdim=32).to(DEVICE)
    model.eval()

    x     = torch.randn(B, S, D, device=DEVICE, dtype=DTYPE)
    x_fp  = x.float()   # para componentes que necesitan fp32

    def time_block(fn, n=N_ITER):
        with torch.no_grad():
            for _ in range(3): fn()   # warm-up
            sync(); t0 = time.perf_counter()
            for _ in range(n): fn()
            sync()
        return (time.perf_counter() - t0) / n * 1000

    t_full = time_block(lambda: model(x, bus_cache=None))

    # Mamba2 solo
    t_m2 = time_block(lambda: model.mamba2(model.norm(x)))

    # Router
    t_router = time_block(lambda: model.router(model.norm(x)))

    # SLR (siempre activo en eval, lo forzamos)
    mamba_out = model.mamba2(model.norm(x))
    t_slr = time_block(lambda: model.slr(mamba_out, importance=None))

    # Bus (sin cache previo)
    t_bus_no = time_block(lambda: model.bus(mamba_out, bus_cache=None))

    # Bus con cache de 4 capas previas  (dtype=float32: bus.publish usa Linear fp32)
    fake_cache = torch.randn(B, 4, 128, device=DEVICE, dtype=torch.float32)
    mamba_out  = mamba_out.contiguous()   # asegurar layout correcto
    t_bus_4 = time_block(lambda: model.bus(mamba_out, bus_cache=fake_cache))

    # Archive retrieve
    t_arch = time_block(lambda: model.archive.retrieve(mamba_out))

    # RMSNorm
    t_norm = time_block(lambda: model.norm(x))

    print(f"  {'Componente':<28} {'ms/iter':>10}  {'%total':>8}")
    print("  " + "-" * 52)
    components = [
        ("Full forward (baseline)",  t_full),
        ("  Mamba2 SSD scan",        t_m2),
        ("  Router (GCP 3-tier)",    t_router),
        ("  SLR Diff-Attn",          t_slr),
        ("  Bus (sin cache)",        t_bus_no),
        ("  Bus (cache L=4)",        t_bus_4),
        ("  Archive retrieve",       t_arch),
        ("  RMSNorm",                t_norm),
    ]
    breakdown = {}
    for name, t in components:
        pct = t / t_full * 100
        print(f"  {name:<28} {t:>10.3f}ms  {pct:>7.1f}%")
        breakdown[name.strip()] = dict(ms=round(t,3), pct=round(pct,1))

    overhead_total = t_full - t_m2
    print(f"\n  Overhead TOTAL sobre Mamba2: {overhead_total:.3f} ms ({overhead_total/t_full*100:.1f}%)")
    results["T2_latency_breakdown"] = breakdown
    del model; gpu_reset()


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Latencia de decode token-by-token (step())
# ══════════════════════════════════════════════════════════════════════════════

def test_decode_latency():
    banner("T3 · DECODE LATENCIA  (step() token-by-token autoregresivo)")

    D, B = 256, 1
    N_DECODE = 200

    model = AdvancedChimeraLayer(d_model=D, expand=2, headdim=32).to(DEVICE)
    model.eval()

    cache = model.allocate_inference_cache(batch_size=B)

    # Warm-up: 10 tokens
    with torch.no_grad():
        for _ in range(10):
            tok = torch.randn(B, 1, D, device=DEVICE, dtype=DTYPE)
            _, cache = model.step(tok, cache)

    # Benchmark: N_DECODE tokens
    times = []
    with torch.no_grad():
        for _ in range(N_DECODE):
            tok = torch.randn(B, 1, D, device=DEVICE, dtype=DTYPE)
            sync(); t0 = time.perf_counter()
            _, cache = model.step(tok, cache)
            sync(); t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    times_t = torch.tensor(times)
    p50  = times_t.quantile(0.50).item()
    p95  = times_t.quantile(0.95).item()
    p99  = times_t.quantile(0.99).item()
    mean = times_t.mean().item()
    std  = times_t.std().item()
    tps  = 1000.0 / mean   # tokens per second

    print(f"  Tokens decodificados: {N_DECODE}")
    print(f"  Latencia por token:   mean={mean:.2f}ms  std={std:.2f}ms")
    print(f"  Percentiles:          p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms")
    print(f"  Throughput decode:    {tps:.1f} tok/s")

    flag = PASS if p99 < 50.0 else WARN
    print(f"  p99 < 50ms:           {flag}")

    results["T3_decode_latency"] = dict(
        mean_ms=round(mean,2), std_ms=round(std,2),
        p50=round(p50,2), p95=round(p95,2), p99=round(p99,2),
        toks_per_sec=round(tps,1), n_tokens=N_DECODE
    )
    del model; gpu_reset()


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — Calidad de adaptación TTT (¿cambia dt_bias? ¿cuánto?)
# ══════════════════════════════════════════════════════════════════════════════

def test_ttt_adaptation():
    banner("T4 · CALIDAD DE ADAPTACIÓN TTT  (dt_bias delta por forward)")

    D, B, S = 256, 2, 256
    N_STEPS = 30

    model = AdvancedChimeraLayer(d_model=D, expand=2, headdim=32).to(DEVICE)
    model.train()   # TTT solo activo en training
    model.ttt_lr = 1e-3   # asegurar lr activo

    dt_init = model.mamba2.dt_bias.data.clone()

    deltas  = []
    entropies_router = []

    for step in range(N_STEPS):
        x = torch.randn(B, S, D, device=DEVICE, dtype=DTYPE)
        out, cache, aux = model(x, bus_cache=None, return_aux=True)

        dt_now   = model.mamba2.dt_bias.data.clone()
        delta    = (dt_now - dt_init).abs().mean().item()
        dt_init  = dt_now.clone()
        deltas.append(delta)

        # entropía del router
        h = entropy(aux['routing_probs'])
        entropies_router.append(h)

    mean_delta  = sum(deltas) / len(deltas)
    max_delta   = max(deltas)
    zeros       = sum(1 for d in deltas if d < 1e-8)
    mean_H      = sum(entropies_router) / len(entropies_router)

    print(f"  Steps TTT evaluados:    {N_STEPS}")
    print(f"  Δdt_bias media/step:    {mean_delta:.6f}")
    print(f"  Δdt_bias máximo:        {max_delta:.6f}")
    print(f"  Steps con Δ=0 (TTT OFF): {zeros}/{N_STEPS}")
    print(f"  Entropía router media:  {mean_H:.4f} bits")
    print(f"  ttt_active en aux:      {aux['ttt_active']}")

    if max_delta < 1e-8:
        fail("dt_bias NUNCA cambió — TTT completamente inactivo")
    elif zeros > N_STEPS // 2:
        warn(f"TTT inactivo en {zeros}/{N_STEPS} pasos — revisar router/threshold")
    else:
        ok(f"TTT activo, Δdt_bias mean={mean_delta:.6f}")

    results["T4_ttt_adaptation"] = dict(
        mean_delta=round(mean_delta, 8), max_delta=round(max_delta, 8),
        zero_steps=zeros, total_steps=N_STEPS,
        mean_routing_entropy_bits=round(mean_H, 4),
    )
    del model; gpu_reset()


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — Estabilidad de routing (colapso de tiers, H(probs) a lo largo del tiempo)
# ══════════════════════════════════════════════════════════════════════════════

def test_routing_stability():
    banner("T5 · ESTABILIDAD DE ROUTING  (¿colapso de tiers?)")

    D, B, S = 256, 4, 512
    N_BATCHES = 80

    model = AdvancedChimeraLayer(d_model=D, expand=2, headdim=32).to(DEVICE)
    model.eval()

    hist_probs = []   # [N_BATCHES, 3]
    entropies  = []

    with torch.no_grad():
        for i in range(N_BATCHES):
            # Mixtura de secuencias: algunas con alta frecuencia, otras ruido puro
            if i % 3 == 0:
                x = torch.randn(B, S, D, device=DEVICE, dtype=DTYPE)
            elif i % 3 == 1:
                x = torch.sin(torch.linspace(0, 4*math.pi, S*D, device=DEVICE)
                               .view(1, S, D).expand(B, -1, -1).to(DTYPE)) * 0.5
            else:
                x = torch.zeros(B, S, D, device=DEVICE, dtype=DTYPE)
                x[:, ::8, :] = 1.0

            _, _, aux = model(x, bus_cache=None, return_aux=True)
            p = aux['routing_probs'].mean(dim=0)   # [3]
            hist_probs.append(p.tolist())
            entropies.append(entropy(aux['routing_probs']))

    probs_t   = torch.tensor(hist_probs)      # [N, 3]
    mean_p    = probs_t.mean(dim=0).tolist()
    std_p     = probs_t.std(dim=0).tolist()
    min_H     = min(entropies)
    max_H     = max(entropies)
    mean_H    = sum(entropies) / len(entropies)

    # Colapso: cualquier tier > 95% del tiempo
    collapse_fast   = mean_p[0] > 0.95
    collapse_hybrid = mean_p[1] > 0.95
    collapse_full   = mean_p[2] > 0.95
    any_collapse    = collapse_fast or collapse_hybrid or collapse_full

    print(f"  Batches evaluados: {N_BATCHES} (3 tipos de input)")
    print(f"\n  Probabilidad media por tier:")
    print(f"    FAST   (0): {mean_p[0]:.4f}  ±{std_p[0]:.4f}")
    print(f"    HYBRID (1): {mean_p[1]:.4f}  ±{std_p[1]:.4f}")
    print(f"    FULL   (2): {mean_p[2]:.4f}  ±{std_p[2]:.4f}")
    print(f"\n  Entropía H(probs):  mean={mean_H:.4f} bits  min={min_H:.4f}  max={max_H:.4f}")
    print(f"  Entropía máxima posible (3 tiers): {math.log2(3):.4f} bits")

    if any_collapse:
        fail(f"COLAPSO detectado — revisar ChimeraLosses routing_entropy_loss")
    elif mean_H < 0.5:
        warn(f"Entropía baja ({mean_H:.4f} bits) — router casi colapsado")
    else:
        ok(f"Router estable, entropía media={mean_H:.4f} bits")

    results["T5_routing_stability"] = dict(
        mean_probs=mean_p, std_probs=std_p,
        entropy_mean=round(mean_H,4), entropy_min=round(min_H,4),
        entropy_max=round(max_H,4), any_collapse=any_collapse
    )
    del model; gpu_reset()


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — Tarea de copy/retrieval (¿SLR + archive ayudan?)
# ══════════════════════════════════════════════════════════════════════════════

def test_copy_retrieval():
    banner("T6 · TAREA COPY/RETRIEVAL  (info en posición P, recuperación en pos Q)")

    D, B, S = 256, 1, 512

    # Señal: token especial en posición P → modelo debe "recordarlo" en posición Q>P
    # Métrica: norma de la diferencia entre la salida en Q con/sin la señal
    # (proxy de cuánto fluye la información a través del stack)

    chimera  = AdvancedChimeraLayer(d_model=D, expand=2, headdim=32).to(DEVICE)
    mamba_bl = Mamba2Baseline(d_model=D, expand=2, headdim=32).to(DEVICE)
    chimera.eval(); mamba_bl.eval()

    P, Q = 50, 400   # token insertado en P, leemos efecto en Q

    row_results = []
    print(f"  Posición señal P={P}, posición lectura Q={Q},  S={S}")
    print(f"\n  {'Amplitud':>8}  {'CHIMERA ΔQ':>12}  {'Mamba2 ΔQ':>12}  {'ratio CHI/MB2':>14}")
    print("  " + "-" * 55)

    for amp in [0.5, 1.0, 2.0, 4.0, 8.0]:
        x_base  = torch.randn(B, S, D, device=DEVICE, dtype=DTYPE)
        signal  = torch.randn(1, D, device=DEVICE, dtype=DTYPE) * amp
        x_sig   = x_base.clone()
        x_sig[0, P, :] = signal

        with torch.no_grad():
            chi_base, _ = chimera(x_base, bus_cache=None)
            chi_sig,  _ = chimera(x_sig,  bus_cache=None)
            mb2_base    = mamba_bl(x_base.float()).to(DTYPE)
            mb2_sig     = mamba_bl(x_sig.float()).to(DTYPE)

        delta_chi = (chi_sig[0, Q, :] - chi_base[0, Q, :]).norm().item()
        delta_mb2 = (mb2_sig[0, Q, :] - mb2_base[0, Q, :]).norm().item()
        ratio = delta_chi / (delta_mb2 + 1e-9)

        print(f"  {amp:>8.1f}  {delta_chi:>12.4f}  {delta_mb2:>12.4f}  {ratio:>14.3f}")
        row_results.append(dict(amp=amp, chi_delta=round(delta_chi,4),
                                mb2_delta=round(delta_mb2,4), ratio=round(ratio,3)))

    # También: retrieval exacto desde archivo — insertar landmark y comprobar que retrieve cambia salida
    chimera2 = AdvancedChimeraLayer(d_model=D, expand=2, headdim=32).to(DEVICE)
    chimera2.eval()
    with torch.no_grad():
        x_clean = torch.randn(B, S, D, device=DEVICE, dtype=DTYPE)
        # Forward 1: acumula landmarks (si tier FULL activo)
        out1, _ = chimera2(x_clean, bus_cache=None)
        n_lm1 = chimera2.archive.get_archive_info().get('n_landmarks', 0)

        # Forward 2: misma entrada, ahora retrieve puede encontrar landmarks del forward 1
        out2, _ = chimera2(x_clean, bus_cache=None)
        n_lm2 = chimera2.archive.get_archive_info().get('n_landmarks', 0)
        diff_out = (out2 - out1).norm().item()

    print(f"\n  LandmarkArchive: forward1 landmarks={n_lm1} → forward2 landmarks={n_lm2}")
    print(f"  Diferencia output fwd1 vs fwd2:  {diff_out:.4f}")
    if diff_out > 1e-4:
        ok(f"Archive inyecta señal (diff={diff_out:.4f})")
    else:
        warn(f"Archive no diferencia output (diff={diff_out:.4f}) — tier FULL inactivo")

    results["T6_copy_retrieval"] = dict(retrieval_rows=row_results,
                                         archive_diff=round(diff_out,4),
                                         n_landmarks_after_fwd1=n_lm1,
                                         n_landmarks_after_fwd2=n_lm2)
    del chimera, mamba_bl, chimera2; gpu_reset()


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7 — Flujo de gradientes por componente
# ══════════════════════════════════════════════════════════════════════════════

def test_gradient_flow():
    banner("T7 · FLUJO DE GRADIENTES  (normas por componente, vanish/explode)")

    D, B, S = 256, 2, 512

    model = AdvancedChimeraLayer(d_model=D, expand=2, headdim=32).to(DEVICE)
    model.train()

    x = torch.randn(B, S, D, device=DEVICE, dtype=DTYPE, requires_grad=False)
    out, cache, aux = model(x, bus_cache=None, return_aux=True)

    # Loss simple: combinar LM proxy + routing entropy
    losses_acc = ChimeraLosses(routing_weight=0.01, ttt_pred_weight=0.05)
    losses_acc.add_routing_probs(aux['routing_probs'])
    aux_loss = losses_acc.compute()
    loss = out.mean() + aux_loss['total']
    loss.backward()

    # Recolectar normas por grupo
    groups = {
        "mamba2.dt_bias":      [model.mamba2.dt_bias],
        "mamba2.A_log":        [model.mamba2.A_log],
        "mamba2.in_proj":      [model.mamba2.in_proj.weight],
        "mamba2.out_proj":     [model.mamba2.out_proj.weight],
        "ttt_U":               [model.ttt_U],
        "ttt_V":               [model.ttt_V],
        "ttt_full_scale":      [model.ttt_full_scale],
        "slr.lam_logit":       [model.slr.lam_logit],
        "slr.merge_gate":      [model.slr.merge_gate],
        "bus.publish":         list(model.bus.publish.parameters()),
        "bus.gate":            [model.bus.gate],
        "router.mlp":          list(model.router.mlp.parameters()),
        "norm.weight":         list(model.norm.parameters()),
    }

    print(f"  {'Componente':<24} {'grad_norm':>12}  {'param_norm':>12}  {'ratio':>10}  Estado")
    print("  " + "-" * 72)

    grad_report = {}
    all_ok = True
    for name, params in groups.items():
        p_norm  = sum(p.data.norm().item()**2 for p in params)**0.5
        g_norms = [p.grad.norm().item() if p.grad is not None else 0.0 for p in params]
        g_norm  = sum(g**2 for g in g_norms)**0.5
        ratio   = g_norm / (p_norm + 1e-12)

        if g_norm == 0.0:
            estado = f"{FAIL} GRAD=0 (dead)"
            all_ok = False
        elif g_norm > 1e3:
            estado = f"{WARN} EXPLODE"
            all_ok = False
        elif g_norm < 1e-8:
            estado = f"{WARN} VANISH"
            all_ok = False
        else:
            estado = f"{PASS}"

        print(f"  {name:<24} {g_norm:>12.2e}  {p_norm:>12.2e}  {ratio:>10.2e}  {estado}")
        grad_report[name] = dict(grad_norm=g_norm, param_norm=p_norm, ratio=ratio)

    if all_ok:
        ok("Todos los componentes tienen gradientes finitos y no nulos")
    else:
        warn("Algunos componentes con gradientes problemáticos — ver tabla")

    # Verificar que ttt_U/V reciben gradiente (clave para aprendizaje)
    u_grad = model.ttt_U.grad
    v_grad = model.ttt_V.grad
    if u_grad is not None and v_grad is not None:
        ok(f"TTT-Full U/V gradientes: ‖∇U‖={u_grad.norm().item():.2e}  ‖∇V‖={v_grad.norm().item():.2e}")
    else:
        fail("TTT-Full U/V NO reciben gradiente — check requires_grad")

    results["T7_gradient_flow"] = grad_report
    del model; gpu_reset()


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8 — Escalado de memoria VRAM vs longitud de secuencia
# ══════════════════════════════════════════════════════════════════════════════

def test_memory_scaling():
    banner("T8 · ESCALADO DE MEMORIA  (VRAM pico vs S, fwd + bwd)")

    D, B = 256, 2
    LENGTHS = [256, 512, 1024, 2048, 4096]

    print(f"  {'S':>5}  {'fwd MB':>10}  {'bwd MB':>10}  {'bwd/fwd':>9}  {'MB/token':>10}")
    print("  " + "-" * 55)

    mem_results = []
    for S in LENGTHS:
        # Forward only
        model_f = AdvancedChimeraLayer(d_model=D, expand=2, headdim=32).to(DEVICE)
        model_f.eval()
        x = torch.randn(B, S, D, device=DEVICE, dtype=DTYPE)
        gpu_reset()
        with torch.no_grad():
            _ = model_f(x, bus_cache=None)
        vram_fwd = peak_mb()
        del model_f; gpu_reset()

        # Forward + Backward
        try:
            model_b = AdvancedChimeraLayer(d_model=D, expand=2, headdim=32).to(DEVICE)
            model_b.train()
            x2 = torch.randn(B, S, D, device=DEVICE, dtype=DTYPE)
            gpu_reset()
            out, _, _ = model_b(x2, bus_cache=None, return_aux=True)
            out.mean().backward()
            sync()
            vram_bwd = peak_mb()
            del model_b; gpu_reset()
            ratio_bwd = vram_bwd / (vram_fwd + 1e-6)
            mb_per_tok = vram_bwd / (B * S) * 1024  # KB/token → convertir a bytes
            status = PASS if vram_bwd < 5500 else FAIL
            print(f"  {S:>5}  {vram_fwd:>10.1f}  {vram_bwd:>10.1f}  {ratio_bwd:>9.2f}x  {mb_per_tok:>9.4f}KB  {status}")
            mem_results.append(dict(S=S, fwd_mb=round(vram_fwd,1),
                                    bwd_mb=round(vram_bwd,1), ratio=round(ratio_bwd,2),
                                    kb_per_token=round(mb_per_tok,4)))
        except torch.cuda.OutOfMemoryError:
            fail(f"  {S:>5}  OOM durante backward")
            mem_results.append(dict(S=S, fwd_mb=round(vram_fwd,1), bwd_mb="OOM"))
            gpu_reset()

    results["T8_memory_scaling"] = mem_results


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9 — Comportamiento del warm-up (gates y lr durante 3 fases)
# ══════════════════════════════════════════════════════════════════════════════

def test_warmup_phases():
    banner("T9 · WARM-UP ESCALONADO  (gates, ttt_lr por fase)")

    D = 256
    layers = [AdvancedChimeraLayer(d_model=D, expand=2, headdim=32).to(DEVICE)
              for _ in range(3)]
    sched  = ChimeraWarmupScheduler(layers, warm1=100, warm2=300)

    probe_steps = [0, 50, 100, 150, 200, 300, 400]
    print(f"  {'step':>5}  {'phase':>6}  {'ttt_lr':>10}  {'slr_gate[0]':>13}  {'bus_gate[0]':>13}  {'tff_scale[0]':>14}")
    print("  " + "-" * 70)

    phase_results = []
    for step in probe_steps:
        sched._global_step = step
        sched.step(step)
        lr   = layers[0].ttt_lr
        sg   = layers[0].slr.merge_gate.data.item()
        bg   = layers[0].bus.gate.data.item()
        tff  = layers[0].ttt_full_scale.data.item()
        ph   = "1" if step < 100 else ("2" if step < 300 else "3")
        print(f"  {step:>5}  {ph:>6}  {lr:>10.6f}  {sg:>13.4f}  {bg:>13.4f}  {tff:>14.4f}")
        phase_results.append(dict(step=step, phase=int(ph), ttt_lr=lr,
                                  slr_gate=round(sg,4), bus_gate=round(bg,4),
                                  tff_scale=round(tff,4)))

    # Verificar que Fase 1 tiene ttt_lr=0
    sched._global_step = 0; sched.step(0)
    early_lr = layers[0].ttt_lr
    if early_lr == 0.0:
        ok("Fase 1: ttt_lr=0 correctamente")
    else:
        fail(f"Fase 1: ttt_lr={early_lr} (esperado 0)")

    # Verificar que Fase 3 tiene ttt_lr máximo
    sched._global_step = 500; sched.step(500)
    late_lr = layers[0].ttt_lr
    if late_lr >= 9e-4:
        ok(f"Fase 3: ttt_lr={late_lr:.6f} (≈target 1e-3)")
    else:
        warn(f"Fase 3: ttt_lr={late_lr:.6f} (bajo esperado)")

    results["T9_warmup_phases"] = phase_results
    del layers; gpu_reset()


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10 — Stress test S=4096, B=4, backward completo
# ══════════════════════════════════════════════════════════════════════════════

def test_stress():
    banner("T10 · STRESS TEST  (S=4096 B=4, backward completo)")

    D, B, S = 256, 4, 4096

    model = AdvancedChimeraLayer(d_model=D, expand=2, headdim=32).to(DEVICE)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    try:
        gpu_reset()
        x = torch.randn(B, S, D, device=DEVICE, dtype=DTYPE)

        sync(); t0 = time.perf_counter()
        out, bus_cache, aux = model(x, bus_cache=None, return_aux=True)
        sync(); t_fwd = (time.perf_counter() - t0) * 1000

        losses_acc = ChimeraLosses(routing_weight=0.01, ttt_pred_weight=0.05)
        losses_acc.add_routing_probs(aux['routing_probs'])
        aux_loss = losses_acc.compute()
        loss = out.mean() + aux_loss['total']

        sync(); t0 = time.perf_counter()
        loss.backward()
        sync(); t_bwd = (time.perf_counter() - t0) * 1000

        vram = peak_mb()

        # Grad clip check
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # NaN/Inf check
        has_nan = not out.isfinite().all().item()
        has_nan_grad = any(
            (p.grad is not None and not p.grad.isfinite().all().item())
            for p in model.parameters()
        )

        opt.step()

        print(f"  B={B}  S={S}  D={D}")
        print(f"  Forward:           {t_fwd:.1f} ms")
        print(f"  Backward:          {t_bwd:.1f} ms")
        print(f"  Total:             {t_fwd+t_bwd:.1f} ms")
        print(f"  VRAM pico:         {vram:.1f} MB  ({vram/6000*100:.1f}% de 6GB)")
        print(f"  Grad total norm:   {total_norm:.4f}")
        print(f"  Output NaN/Inf:    {has_nan}")
        print(f"  Grad NaN/Inf:      {has_nan_grad}")
        print(f"  aux_loss total:    {aux_loss['total'].item():.6f}")

        if not has_nan and not has_nan_grad:
            ok("Stress test PASADO — sin NaN/Inf")
        else:
            fail("NaN/Inf detectados durante stress")

        if vram < 5500:
            ok(f"VRAM dentro del límite ({vram:.0f} MB < 5500 MB)")
        else:
            warn(f"VRAM al límite: {vram:.0f} MB — riesgo OOM con B mayor")

        results["T10_stress"] = dict(
            B=B, S=S, fwd_ms=round(t_fwd,1), bwd_ms=round(t_bwd,1),
            vram_mb=round(vram,1), grad_norm=round(total_norm.item(),4),
            has_nan=has_nan, has_nan_grad=has_nan_grad,
            aux_total_loss=round(aux_loss['total'].item(),6),
        )

    except torch.cuda.OutOfMemoryError:
        fail(f"OOM en S={S} B={B} — reducir batch o activar gradient checkpointing")
        results["T10_stress"] = dict(B=B, S=S, oom=True)
    finally:
        del model; gpu_reset()


# ══════════════════════════════════════════════════════════════════════════════
# BONUS — Resumen estadístico de componentes internas
# ══════════════════════════════════════════════════════════════════════════════

def test_component_health():
    banner("BONUS · SALUD DE COMPONENTES INTERNOS")

    D, B, S = 256, 2, 512
    model = AdvancedChimeraLayer(d_model=D, expand=2, headdim=32).to(DEVICE)
    model.eval()

    x = torch.randn(B, S, D, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        out, bus_cache, aux = model(x, bus_cache=None, return_aux=True)

    # 1. Output stats
    print(f"\n  Output [B,S,D]:")
    print(f"    shape={tuple(out.shape)}  dtype={out.dtype}")
    print(f"    mean={out.mean().item():.4f}  std={out.std().item():.4f}")
    print(f"    finite={out.isfinite().all().item()}")
    print(f"    min={out.min().item():.4f}  max={out.max().item():.4f}")

    # 2. Bus cache
    print(f"\n  Bus cache: shape={tuple(bus_cache.shape)}")

    # 3. Router
    p = aux['routing_probs']
    print(f"\n  Router probs (media batch):")
    print(f"    FAST={p[:,0].mean().item():.4f}  HYBRID={p[:,1].mean().item():.4f}  FULL={p[:,2].mean().item():.4f}")
    print(f"    Entropía: {entropy(p):.4f} bits  (max={math.log2(3):.4f})")

    # 4. Archive
    ai = model.archive.get_archive_info()
    print(f"\n  NativeLandmarkArchive: {ai}")

    # 5. TTT params
    print(f"\n  TTT-Full:")
    print(f"    U shape: {tuple(model.ttt_U.shape)}  norm={model.ttt_U.norm().item():.4f}")
    print(f"    V shape: {tuple(model.ttt_V.shape)}  norm={model.ttt_V.norm().item():.4f}")
    print(f"    scale: {model.ttt_full_scale.data.item():.4f} → sigmoid={torch.sigmoid(model.ttt_full_scale).item():.4f}")

    # 6. SLR
    print(f"\n  SLR:")
    print(f"    λ logit: {model.slr.lam_logit.item():.4f} → σ={torch.sigmoid(model.slr.lam_logit).item():.4f}")
    print(f"    merge_gate: {model.slr.merge_gate.item():.4f} → σ={torch.sigmoid(model.slr.merge_gate).item():.4f}")

    # 7. Multi-scale A
    A_vals = (-model.mamba2.A_log.exp()).detach().flatten()
    print(f"\n  Multi-scale A (Mamba2):")
    print(f"    min={A_vals.min().item():.4f}  max={A_vals.max().item():.4f}  std={A_vals.std().item():.4f}")

    health = dict(
        out_mean=round(out.mean().item(),4), out_std=round(out.std().item(),4),
        out_finite=out.isfinite().all().item(),
        bus_cache_shape=list(bus_cache.shape),
        router_mean_probs=[round(p[:,i].mean().item(),4) for i in range(3)],
        router_entropy=round(entropy(p),4),
        archive_info=ai,
        ttt_U_norm=round(model.ttt_U.norm().item(),4),
        ttt_V_norm=round(model.ttt_V.norm().item(),4),
        ttt_full_scale=round(torch.sigmoid(model.ttt_full_scale).item(),4),
        slr_lambda=round(torch.sigmoid(model.slr.lam_logit).item(),4),
        slr_merge_gate=round(torch.sigmoid(model.slr.merge_gate).item(),4),
        A_min=round(A_vals.min().item(),4), A_max=round(A_vals.max().item(),4),
    )
    results["BONUS_component_health"] = health
    del model; gpu_reset()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(SEP)
    print(f"{BOLD}  CHIMERA DEEP BENCHMARK SUITE{RESET}")
    print(SEP)

    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:   {props.name}")
        print(f"  VRAM:  {props.total_memory/1024**3:.1f} GB")
        print(f"  SM:    {props.major}{props.minor}")
        print(f"  DTYPE: {DTYPE} (modelos en fp32 nativo)")
    else:
        print(f"  AVISO: ejecutando en CPU — resultados de latencia no representativos")
    print()

    # Ejecutar todos los tests en orden
    TESTS = [
        ("T1",    test_throughput),
        ("T2",    test_latency_breakdown),
        ("T3",    test_decode_latency),
        ("T4",    test_ttt_adaptation),
        ("T5",    test_routing_stability),
        ("T6",    test_copy_retrieval),
        ("T7",    test_gradient_flow),
        ("T8",    test_memory_scaling),
        ("T9",    test_warmup_phases),
        ("T10",   test_stress),
        ("BONUS", test_component_health),
    ]

    t_global = time.perf_counter()
    failed_tests = []
    for tag, fn in TESTS:
        try:
            fn()
        except Exception as e:
            fail(f"{tag} lanzó excepción inesperada: {e}")
            import traceback; traceback.print_exc()
            failed_tests.append(tag)
            gpu_reset()

    t_total = time.perf_counter() - t_global

    # Guardar resultados JSON
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_chimera_deep_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{SEP}")
    print(f"{BOLD}  BENCHMARK COMPLETADO{RESET}  —  {t_total:.1f}s  |  resultados: {out_path}")
    if failed_tests:
        print(f"  {FAIL} Tests fallidos: {', '.join(failed_tests)}")
    else:
        print(f"  {PASS} Todos los tests completados")
    print(SEP)
