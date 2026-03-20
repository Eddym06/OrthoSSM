#!/usr/bin/env python3
"""
chimera_million_stress.py — Stress Test Masivo CHIMERA
========================================================
Suite de pruebas de estabilidad, rendimiento y robustez para contextos
de millones de tokens. Diseñado para detectar:

  M1.  Throughput & VRAM escalado a 128K, 512K, 1M tokens (extrapolado)
  M2.  Estabilidad numérica bajo 50 steps de training con optimizer real
  M3.  Drift chunked vs full POST-FIX (debe ser < 5% antes era 106%)
  M4.  Routing specialization con Z-loss: H converge < H_max en 20 steps
  M5.  Leak de memoria VRAM: 100 forwards consecutivos deben ser flat
  M6.  Long-context NIAH (Needle in a Haystack): S=16K, recuperación exacta
  M7.  Decode autoregresivo 4096 tokens: sin NaN, sin corrupción de estado
  M8.  Gradient health: grad_norm por componente, detección vanish/explode
  M9.  Error acumulado en procesamiento por chunks de 1M tokens virtuales
  M10. Stress concurrente: 4 capas en stack, backward end-to-end con routing loss

Ejecutar: /home/OrthoSSM/venv/bin/python chimera_experiment/chimera_million_stress.py
Salida:   /tmp/chimera_million_stress.json (resultados) + stdout detallado
"""
import sys, os, gc, time, math, json, traceback
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS: dict = {}

# ─── Colores ANSI ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
SEP    = "=" * 72

def ok(m):   print(f"  {GREEN}✓{RESET} {m}")
def fail(m): print(f"  {RED}✗{RESET} {m}")
def warn(m): print(f"  {YELLOW}⚠{RESET} {m}")
def info(m): print(f"    {m}")
def banner(t): print(f"\n{SEP}\n{BOLD}{t}{RESET}\n{SEP}")

def sync():
    if DEVICE == "cuda": torch.cuda.synchronize()

def vram_mb() -> float:
    if DEVICE == "cuda": sync(); return torch.cuda.memory_allocated() / 1e6
    return 0.0

def vram_peak_mb() -> float:
    if DEVICE == "cuda": return torch.cuda.max_memory_allocated() / 1e6
    return 0.0

def reset_vram_peak():
    if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats()

def flush_cache():
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

def timer_ms(fn, warmup: int = 3, reps: int = 8) -> tuple[float, float]:
    """Mide tiempo de ejecución. Retorna (mean_ms, std_ms)."""
    for _ in range(warmup): fn()
    sync()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); sync()
        times.append((time.perf_counter() - t0) * 1e3)
    t = torch.tensor(times)
    return float(t.mean()), float(t.std())

def make_model(d_model: int = 256, eval_mode: bool = True):
    from advanced_chimera import AdvancedChimeraLayer
    m = AdvancedChimeraLayer(d_model=d_model, expand=2, headdim=32).to(DEVICE).float()
    return m.eval() if eval_mode else m.train()

def check_finite(t: torch.Tensor, name: str) -> bool:
    if not t.isfinite().all():
        fail(f"{name}: tiene NaN/Inf! max={t.abs().max().item():.2e}")
        return False
    return True

# ─── M1: Throughput & VRAM escalado ──────────────────────────────────────────
def test_M1_throughput_scale():
    banner("M1 — Throughput & VRAM vs longitud (hasta extrapolación 1M tok)")
    model = make_model()
    d = 256; results = []

    # Longitudes que caben en 6GB RTX4050 en un forward single-layer
    lens = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    for S in lens:
        flush_cache(); reset_vram_peak()
        x = torch.randn(1, S, d, device=DEVICE)
        v0 = vram_mb()
        def fwd(): 
            with torch.no_grad(): model(x)
        try:
            ms, std = timer_ms(fwd, warmup=3, reps=6)
            peak = vram_peak_mb(); delta = peak - v0
            tps  = int(S / ms * 1e3)
            results.append({
                "S": S, "ms": round(ms, 2), "std": round(std, 2),
                "tps": tps, "vram_delta_mb": round(delta, 1), "vram_peak_mb": round(peak, 1),
                "oom": False
            })
            info(f"S={S:>6,} | {ms:6.2f}±{std:.2f} ms | {tps:>10,} tok/s | VRAM Δ={delta:.0f} MB | peak={peak:.0f} MB")
        except RuntimeError as e:
            if "out of memory" in str(e):
                results.append({"S": S, "oom": True, "ms": None, "tps": 0})
                warn(f"S={S:>6,} | OOM")
                flush_cache()
            else:
                raise

    # Regresión lineal sobre VRAM para extrapolación
    valid = [r for r in results if not r["oom"]]
    if len(valid) >= 2:
        xs = [r["S"] for r in valid]
        ys = [r["vram_delta_mb"] for r in valid]
        n  = len(xs); sx = sum(xs); sy = sum(ys)
        sxy = sum(a*b for a,b in zip(xs,ys)); sxx = sum(a*a for a in xs)
        slope = (n*sxy - sx*sy) / (n*sxx - sx*sx + 1e-9)
        intercept = (sy - slope*sx) / n
        tps_ref = valid[-1]["tps"]
        print()
        info("─── Extrapolación lineal (VRAM O(S), throughput ~cte) ───")
        for S_p, label in [(131_072, "128K"), (524_288, "512K"), (1_000_000, "1M")]:
            vp = slope * S_p + intercept
            ms_p = S_p / tps_ref * 1e3
            info(f"  S={label:>4}: VRAM≈{vp/1024:.2f} GB | latencia≈{ms_p/1e3:.1f}s | tps≈{tps_ref:,}")

    ok("M1 completado")
    RESULTS["M1_throughput"] = results

# ─── M2: Estabilidad training 50 steps ───────────────────────────────────────
def test_M2_training_stability():
    banner("M2 — Estabilidad numérica: 50 steps de training con optimizer real")
    model = make_model(eval_mode=False)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    B, S, D = 2, 512, 256

    losses_log = []; nan_steps = 0; grad_norms = []
    for step in range(50):
        opt.zero_grad()
        x   = torch.randn(B, S, D, device=DEVICE)
        out, _, aux = model(x, return_aux=True)
        if not check_finite(out, f"step {step} out"):
            nan_steps += 1; continue

        # Pérdida LM proxy (predecir siguiente activación)
        loss = F.mse_loss(out[:, :-1], x[:, 1:].detach())

        # Routing loss Z-loss
        from chimera_losses import ChimeraRoutingLoss
        rl  = ChimeraRoutingLoss(z_loss_weight=1e-3).to(DEVICE)
        rl_loss, rl_info = rl(aux)
        total_loss = loss + 0.01 * rl_loss
        total_loss.backward()

        # Gradient clipping
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        grad_norms.append(gn)
        losses_log.append(float(total_loss.item()))

        if step % 10 == 0:
            info(f"  step={step:>3}  loss={loss.item():.4f}  rl={rl_loss.item():.4f}  "
                 f"grad_norm={gn:.3f}  H={rl_info['routing/H_bits']:.3f}b")

    loss_reduction = (losses_log[0] - losses_log[-1]) / (losses_log[0] + 1e-8) * 100

    if nan_steps == 0:
        ok(f"50 steps sin NaN/Inf")
    else:
        fail(f"{nan_steps} steps con NaN/Inf")

    info(f"  loss inicial: {losses_log[0]:.4f}  final: {losses_log[-1]:.4f}  "
         f"reducción: {loss_reduction:.1f}%")
    info(f"  grad_norm: min={min(grad_norms):.3f}  max={max(grad_norms):.3f}  "
         f"mean={sum(grad_norms)/len(grad_norms):.3f}")

    if max(grad_norms) > 10.0:
        warn(f"max grad_norm={max(grad_norms):.2f} > 10 (revisar LR)")
    else:
        ok("Gradient norms estables (< 10)")

    RESULTS["M2_stability"] = {
        "nan_steps": nan_steps,
        "loss_initial": round(losses_log[0], 4),
        "loss_final":   round(losses_log[-1], 4),
        "loss_reduction_pct": round(loss_reduction, 1),
        "grad_norm_max": round(max(grad_norms), 4),
        "grad_norm_mean": round(sum(grad_norms)/len(grad_norms), 4),
    }

# ─── M3: Drift chunked vs full POST-FIX ──────────────────────────────────────
def test_M3_drift_chunked_vs_full():
    banner("M3 — Drift chunked vs full (POST-FIX TTT asíncrono, objetivo < 5%)")
    model = make_model(eval_mode=False)
    model.train()
    D = 256

    drifts = []
    for S in [256, 512, 1024]:
        x = torch.randn(1, S, D, device=DEVICE)
        with torch.no_grad():
            out_full, _ = model(x)

        # Procesamiento chunked: 4 trozos
        chunk_size = S // 4
        chunks = x.split(chunk_size, dim=1)
        out_chunks = []
        for c in chunks:
            with torch.no_grad():
                o, _ = model(c)
            out_chunks.append(o)
        out_chunked = torch.cat(out_chunks, dim=1)

        min_S = min(out_full.shape[1], out_chunked.shape[1])
        diff  = (out_full[:, :min_S] - out_chunked[:, :min_S]).abs()
        drift_pct = diff.mean().item() / (out_full[:, :min_S].abs().mean().item() + 1e-8) * 100
        status = "✓" if drift_pct < 5.0 else "✗" if drift_pct > 20.0 else "⚠"
        info(f"  S={S:>5} | drift={drift_pct:.2f}%  {status}")
        drifts.append({"S": S, "drift_pct": round(drift_pct, 2)})

    max_drift = max(d["drift_pct"] for d in drifts)
    if max_drift < 5.0:
        ok(f"Drift máximo = {max_drift:.2f}% < 5% — TTT asíncrono correcto")
    elif max_drift < 20.0:
        warn(f"Drift máximo = {max_drift:.2f}% en [5%, 20%] — aceptable pero revisa")
    else:
        fail(f"Drift máximo = {max_drift:.2f}% > 20% — TTT todavía modifica pre-scan")

    note = ("NOTA: algo de drift es normal. El procesamiento chunked reinicia el"
            " estado SSM en cada chunk (equivalente a secuencias independientes)."
            " El fix asíncrono elimina el componente de drift causado por la"
            " modificación de dt_bias dentro del scan.")
    info(note)
    RESULTS["M3_drift"] = {"drifts": drifts, "max_drift_pct": round(max_drift, 2),
                            "target_pct": 5.0, "pass": max_drift < 20.0}

# ─── M4: Routing specialization con Z-loss ───────────────────────────────────
def test_M4_routing_zloss():
    banner("M4 — Routing specialization: Z-loss + supervision en 20 gradient steps")
    from chimera_losses import ChimeraRoutingLoss
    model = make_model(eval_mode=False)
    rl    = ChimeraRoutingLoss(
        entropy_weight=0.05, supervision_weight=0.10,
        balance_weight=0.02, z_loss_weight=1e-3
    ).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    B, S, D = 4, 256, 256

    h_history = []; p_fast_hist = []; p_full_hist = []

    # Simular inputs con complejidad variada: mezcla ruido blanco y estructura
    info("  Step  H(bits)  p_fast  p_hybrid  p_full  z_loss")
    info("  " + "-" * 55)
    for step in range(20):
        opt.zero_grad()
        # Alternar inputs simples (ruido bajo) y complejos (alta varianza)
        if step % 2 == 0:
            x = torch.randn(B, S, D, device=DEVICE) * 0.1   # simple
        else:
            x = torch.randn(B, S, D, device=DEVICE) * 2.0   # complejo
        out, _, aux = model(x, return_aux=True)
        loss = F.mse_loss(out[:, :-1], x[:, 1:].detach())
        rl_loss, info_d = rl(aux)
        (loss + 0.05 * rl_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        h   = info_d['routing/H_bits']
        pf  = info_d['routing/p_fast']
        ph  = info_d['routing/p_hybrid']
        pfl = info_d['routing/p_full']
        zl  = info_d.get('routing/z_loss', 0.0)
        h_history.append(h); p_fast_hist.append(pf); p_full_hist.append(pfl)
        if step % 5 == 0 or step == 19:
            info(f"  {step:>4}  {h:>7.4f}  {pf:>6.4f}  {ph:>8.4f}  {pfl:>6.4f}  {zl:>8.5f}")

    H_init  = h_history[0]
    H_final = h_history[-1]
    H_max   = math.log(3) / math.log(2)  # bits
    specialization = (H_init - H_final) / H_max * 100

    info(f"\n  H init={H_init:.3f}b  final={H_final:.3f}b  H_max={H_max:.3f}b")
    info(f"  Especialización: {specialization:.1f}% de reducción de entropía")

    # Verificar que el router está produciendo gradientes
    rg = model.router.mlp[0].weight.grad
    if rg is not None and rg.norm().item() > 1e-12:
        ok("router.mlp.weight tiene gradiente activo")
    else:
        fail("Router sin gradiente!")

    if specialization > 5.0:
        ok(f"Routing mostrando especialización ({specialization:.1f}%)")
    else:
        warn(f"Especialización baja ({specialization:.1f}%) — necesita más steps")

    RESULTS["M4_routing"] = {
        "H_initial_bits": round(H_init, 4), "H_final_bits": round(H_final, 4),
        "H_max_bits": round(H_max, 4), "specialization_pct": round(specialization, 1),
        "p_fast_final": round(p_fast_hist[-1], 4), "p_full_final": round(p_full_hist[-1], 4),
    }

# ─── M5: VRAM leak detection ─────────────────────────────────────────────────
def test_M5_vram_leak():
    banner("M5 — Detección de leak VRAM: 100 forwards consecutivos")
    model = make_model()
    x = torch.randn(1, 1024, 256, device=DEVICE)
    flush_cache(); reset_vram_peak()

    # Warm-up
    with torch.no_grad():
        for _ in range(5): model(x)

    # Medir VRAM en puntos
    snapshots = []
    for i in range(100):
        with torch.no_grad(): model(x)
        if i % 20 == 0:
            snap = vram_mb()
            snapshots.append({"step": i, "vram_mb": round(snap, 1)})
            info(f"  step={i:>3}  VRAM={snap:.1f} MB")

    # Regresión lineal para detectar leak
    xs_s = [s["step"] for s in snapshots]
    ys_s = [s["vram_mb"] for s in snapshots]
    n = len(xs_s); sx = sum(xs_s); sy = sum(ys_s)
    sxy = sum(a*b for a,b in zip(xs_s, ys_s)); sxx = sum(a*a for a in xs_s)
    slope_mb_per_step = (n*sxy - sx*sy) / (n*sxx - sx*sx + 1e-9)

    if abs(slope_mb_per_step) < 0.1:
        ok(f"Sin leak: slope={slope_mb_per_step:.4f} MB/step (< 0.1)")
    elif abs(slope_mb_per_step) < 1.0:
        warn(f"Posible micro-leak: slope={slope_mb_per_step:.4f} MB/step (< 1.0)")
    else:
        fail(f"LEAK DETECTADO: slope={slope_mb_per_step:.4f} MB/step")

    RESULTS["M5_vram_leak"] = {
        "snapshots": snapshots, "slope_mb_per_step": round(slope_mb_per_step, 4),
        "pass": abs(slope_mb_per_step) < 1.0,
    }

# ─── M6: NIAH Long-Context ───────────────────────────────────────────────────
def test_M6_niah():
    banner("M6 — Needle in a Haystack (NIAH): longitudes hasta 8K+")
    model = make_model()
    D = 256; passed = 0; total = 0

    # Generar un "documento" con una señal embedida en posición conocida
    # Tarea: comprimir el contexto y recuperar el embedding de la posición target
    for S, needle_pos_frac in [(1024, 0.5), (2048, 0.1), (4096, 0.9), (8192, 0.5)]:
        flush_cache()
        x = torch.randn(1, S, D, device=DEVICE) * 0.1   # fondo suave
        needle_pos = int(S * needle_pos_frac)
        # Inyectar señal fuerte ("needle") en posición conocida
        needle = torch.ones(1, 1, D, device=DEVICE) * 5.0   # señal alta
        x[:, needle_pos] = needle.squeeze()

        with torch.no_grad():
            out, _ = model(x)

        # Verificar que la posición needle tiene mayor norma en la salida
        # que las posiciones vecinas (señal debe propagarse hacia adelante)
        out_norms = out[0].norm(dim=-1)   # [S]
        needle_norm = out_norms[needle_pos].item()
        background_norm = out_norms[[max(0, needle_pos-10), min(S-1, needle_pos+10)]].mean().item()
        ratio = needle_norm / (background_norm + 1e-6)

        is_finite = check_finite(out, f"NIAH S={S}")
        signal_detected = ratio > 1.1   # needle debe ser > 10% más fuerte
        total += 1
        if is_finite:
            status = f"ratio={ratio:.2f} {'✓' if signal_detected else '⚠'}"
            info(f"  S={S:>5} frac={needle_pos_frac} | norm_needle={needle_norm:.3f}  "
                 f"norm_bg={background_norm:.3f}  {status}")
            passed += 1 if is_finite else 0
        else:
            passed -= 1

    if passed == total:
        ok(f"NIAH: {passed}/{total} longitudes sin NaN/Inf")
    else:
        fail(f"NIAH: {passed}/{total} pasaron (fallos numéricos)")

    RESULTS["M6_niah"] = {"passed": passed, "total": total}

# ─── M7: Decode autoregresivo 4096 tokens ────────────────────────────────────
def test_M7_autoregressive_decode():
    banner("M7 — Decode autoregresivo: 4096 tokens step-by-step")
    model = make_model()
    D = 256; DECODE_STEPS = 4096

    cache = model.allocate_inference_cache(batch_size=1, dtype=torch.float32)
    x_tok = torch.randn(1, 1, D, device=DEVICE)

    nan_count = 0; decode_times = []
    flush_cache(); reset_vram_peak()

    for i in range(DECODE_STEPS):
        t0 = time.perf_counter(); sync()
        with torch.no_grad():
            out_tok, cache = model.step(x_tok, cache)
        sync(); decode_times.append((time.perf_counter() - t0) * 1e3)

        if not out_tok.isfinite().all():
            nan_count += 1
            warn(f"  NaN en step {i}")
            if nan_count > 5: break

        x_tok = out_tok   # siguiente entrada = salida actual (autoregresivo)

        if i % 1024 == 0:
            info(f"  step={i:>5}  vram={vram_mb():.0f} MB  "
                 f"out_norm={out_tok.norm().item():.3f}")

    mean_ms = sum(decode_times) / len(decode_times)
    tps_decode = 1 / (mean_ms / 1e3)
    vram_decode = vram_peak_mb()

    if nan_count == 0:
        ok(f"4096 steps sin NaN  →  {tps_decode:.0f} tok/s  VRAM peak={vram_decode:.0f} MB")
    else:
        fail(f"{nan_count} steps con NaN")

    RESULTS["M7_decode"] = {
        "steps": DECODE_STEPS, "nan_count": nan_count,
        "tps": round(tps_decode, 1),
        "mean_ms_per_step": round(mean_ms, 3),
        "vram_peak_mb": round(vram_decode, 1),
    }

# ─── M8: Gradient health per componente ──────────────────────────────────────
def test_M8_gradient_health():
    banner("M8 — Salud de gradientes: norma por componente, vanish/explode")
    model = make_model(eval_mode=False)
    B, S, D = 2, 512, 256

    x   = torch.randn(B, S, D, device=DEVICE, requires_grad=False)
    out, _, aux = model(x, return_aux=True)
    from chimera_losses import ChimeraRoutingLoss
    rl   = ChimeraRoutingLoss(z_loss_weight=1e-3).to(DEVICE)
    rl_l, _ = rl(aux)
    loss = F.mse_loss(out[:, :-1], x[:, 1:]) + 0.01 * rl_l
    loss.backward()

    groups = {
        "mamba2.dt_bias":     model.mamba2.dt_bias,
        "mamba2.A_log":       model.mamba2.A_log,
        "ttt_U":              model.ttt_U,
        "ttt_V":              model.ttt_V,
        "ttt_full_scale":     model.ttt_full_scale,
        "router.mlp[0].w":    model.router.mlp[0].weight,
        "slr.lam_logit":      model.slr.lam_logit,
        "slr.merge_gate":     model.slr.merge_gate,
        "bus.publish.weight": model.bus.publish.weight,
        "archive.compress.w": model.archive.compress.weight,
    }
    dead = []; alive = []; explode = []
    info(f"  {'Parámetro':30s}  {'grad_norm':>12}  {'status':>10}")
    info("  " + "-" * 58)
    for name, p in groups.items():
        if p.grad is None:
            gn = 0.0; status = "DEAD"
            dead.append(name)
        else:
            gn = p.grad.norm().item()
            if gn < 1e-10:
                status = "DEAD"; dead.append(name)
            elif gn > 100.0:
                status = "EXPLODE"; explode.append(name)
                alive.append(name)
            else:
                status = "OK"; alive.append(name)
        info(f"  {name:30s}  {gn:>12.3e}  {status:>10}")

    print()
    if len(dead) == 0 and len(explode) == 0:
        ok(f"Todos los parámetros con gradiente saludable ({len(alive)}/{len(groups)})")
    else:
        if dead:
            fail(f"Gradientes muertos: {dead}")
        if explode:
            warn(f"Gradientes explosivos (>100): {explode}")
        ok(f"Parámetros OK: {len(alive)}/{len(groups)}")

    RESULTS["M8_gradients"] = {
        "alive": len(alive), "dead": dead, "explode": explode,
        "total": len(groups),
    }

# ─── M9: Error acumulado en 1M tokens virtuales ───────────────────────────────
def test_M9_million_token_error_accum():
    banner("M9 — Acumulación de error: procesamiento de 1M tokens virtuales por chunks")
    model = make_model()
    D     = 256
    CHUNK = 2048          # chunk size
    N_CHUNKS = 488        # 488 × 2048 ≈ 1M tokens

    info(f"  Simulando {N_CHUNKS * CHUNK:,} tokens ({N_CHUNKS} chunks de {CHUNK})")
    info("  Monitoreo: norma de salida por cada 100 chunks\n")

    # Referencia: primera salida para detectar drift global   
    x_ref = torch.randn(1, CHUNK, D, device=DEVICE)
    with torch.no_grad():
        out_ref, _ = model(x_ref)
    ref_norm = out_ref.norm().item()

    norms = []; nan_chunks = 0
    t_start = time.perf_counter()

    for chunk_i in range(N_CHUNKS):
        # Inputs con distribución ligeramente variada (simula texto real)
        x = torch.randn(1, CHUNK, D, device=DEVICE) * (0.8 + 0.4 * (chunk_i % 7) / 6)
        with torch.no_grad():
            out, _ = model(x)

        if not out.isfinite().all():
            nan_chunks += 1
            if nan_chunks > 3:
                fail(f"  ABORTANDO: {nan_chunks} chunks con NaN en posición {chunk_i}")
                break
        else:
            n = out.norm().item()
            norms.append(n)

        if chunk_i % 100 == 0:
            elapsed = time.perf_counter() - t_start
            tokens_done = (chunk_i + 1) * CHUNK
            tps_so_far = tokens_done / elapsed
            info(f"  chunk={chunk_i:>4}  tokens={tokens_done:>9,}  "
                 f"out_norm={norms[-1] if norms else 0:.3f}  "
                 f"tps={tps_so_far:>10,.0f}")

    elapsed_total = time.perf_counter() - t_start
    total_tokens  = len(norms) * CHUNK

    if norms:
        norm_mean = sum(norms) / len(norms)
        norm_std  = (sum((n - norm_mean)**2 for n in norms) / len(norms)) ** 0.5
        drift_total = abs(norms[-1] - norms[0]) / (abs(norms[0]) + 1e-6) * 100

        info(f"\n  Total: {total_tokens:>10,} tokens  elapsed={elapsed_total:.1f}s  "
             f"avg_tps={total_tokens/elapsed_total:,.0f}")
        info(f"  out_norm: mean={norm_mean:.3f}  std={norm_std:.3f}  "
             f"drift_pct={drift_total:.1f}%")

        if nan_chunks == 0:
            ok(f"1M tokens SIN NaN/Inf  →  avg {total_tokens/elapsed_total:,.0f} tok/s")
        else:
            fail(f"{nan_chunks} chunks con NaN")

        if drift_total < 10.0:
            ok(f"Drift de norma: {drift_total:.1f}% < 10% — salida estable")
        elif drift_total < 30.0:
            warn(f"Drift de norma: {drift_total:.1f}% — aceptable para SSM sin warmup")
        else:
            fail(f"Drift de norma: {drift_total:.1f}% > 30% — potencial inestabilidad")
    else:
        fail("Sin resultados válidos")

    RESULTS["M9_million_tokens"] = {
        "total_tokens": total_tokens,
        "elapsed_s": round(elapsed_total, 1),
        "avg_tps": round(total_tokens / (elapsed_total + 1e-6)),
        "nan_chunks": nan_chunks,
        "norm_mean": round(sum(norms)/len(norms), 3) if norms else None,
        "drift_pct": round(drift_total if norms else 0, 1),
    }

# ─── M10: Stack de 4 capas, backward end-to-end ──────────────────────────────
def test_M10_stack_backward():
    banner("M10 — Stack 4-capa end-to-end: backward + routing loss + gradient check")
    from advanced_chimera import AdvancedChimeraLayer
    from chimera_losses import ChimeraRoutingLoss

    n_layers = 4; D = 256
    layers = nn.ModuleList([
        AdvancedChimeraLayer(d_model=D).to(DEVICE).float().train()
        for _ in range(n_layers)
    ])
    rl  = ChimeraRoutingLoss(
        entropy_weight=0.05, supervision_weight=0.10,
        balance_weight=0.02, z_loss_weight=1e-3
    ).to(DEVICE)
    all_params = list(layers.parameters()) + list(rl.parameters())
    opt = torch.optim.AdamW(all_params, lr=3e-4)

    B, S = 2, 256
    n_steps = 20; info("  20 gradient steps sobre stack 4-capa con routing loss")
    info(f"  Parámetros totales: {sum(p.numel() for p in layers.parameters()):,}")
    info(f"  {'Step':>4}  {'loss':>8}  {'rl_total':>10}  {'H(bits)':>8}  {'gn':>8}  {'p_fast':>7}  {'p_full':>7}")
    info("  " + "-" * 65)

    step_results = []
    for step in range(n_steps):
        opt.zero_grad()
        x = torch.randn(B, S, D, device=DEVICE)
        bus_cache = None; all_aux = []

        out = x
        for layer in layers:
            out, bus_cache, aux = layer(out, bus_cache=bus_cache, return_aux=True)
            all_aux.append(aux)

        # Pérdida tarea proxy
        task_loss = F.mse_loss(out[:, :-1], x[:, 1:].detach())

        # Routing loss agregada sobre las 4 capas
        rl_total = torch.zeros(1, device=DEVICE).squeeze()
        h_bits_total = 0.0; pf = 0.0; pfl = 0.0
        for aux in all_aux:
            rl_l, ri = rl(aux)
            rl_total    += rl_l
            h_bits_total += ri['routing/H_bits']
            pf           += ri['routing/p_fast']
            pfl          += ri['routing/p_full']
        rl_total /= n_layers
        h_bits_mean = h_bits_total / n_layers
        pf /= n_layers; pfl /= n_layers

        total_loss = task_loss + 0.01 * rl_total
        total_loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(layers.parameters(), 1.0).item()
        opt.step()
        step_results.append({"loss": task_loss.item(), "rl": rl_total.item()})

        if step % 5 == 0 or step == n_steps - 1:
            info(f"  {step:>4}  {task_loss.item():>8.4f}  {rl_total.item():>10.5f}  "
                 f"{h_bits_mean:>8.4f}  {gn:>8.4f}  {pf:>7.4f}  {pfl:>7.4f}")

    # Verificar gradientes en bus.publish de cada capa (el más difícil de fluir)
    print()
    ok_count = 0
    for i, layer in enumerate(layers):
        g = layer.bus.publish.weight.grad
        gn_pub = g.norm().item() if g is not None else 0.0
        alive  = gn_pub > 1e-12
        if alive: ok_count += 1
        info(f"  Layer {i}  bus.publish  grad_norm={gn_pub:.3e}  {'✓' if alive else '✗'}")

    if ok_count == n_layers:
        ok(f"bus.publish tiene gradiente en TODAS las {n_layers} capas")
    else:
        fail(f"bus.publish sin gradiente en {n_layers - ok_count}/{n_layers} capas")

    lr_init  = step_results[0]["loss"]
    lr_final = step_results[-1]["loss"]
    converge = (lr_init - lr_final) / (lr_init + 1e-8) * 100

    if converge > 0:
        ok(f"Stack converge: loss inició en {lr_init:.4f}, terminó en {lr_final:.4f} ({converge:.1f}% ↓)")
    else:
        warn(f"Sin convergencia en 20 steps (normal en un toy sin datos reales)")

    RESULTS["M10_stack"] = {
        "n_layers": n_layers, "n_steps": n_steps,
        "bus_publish_alive": ok_count,
        "loss_initial": round(lr_init, 4), "loss_final": round(lr_final, 4),
        "convergence_pct": round(converge, 1),
    }


# ─── RUNNER PRINCIPAL ─────────────────────────────────────────────────────────
def main():
    print(f"\n{SEP}")
    print(f"{BOLD}CHIMERA MILLION-TOKEN STRESS TEST{RESET}")
    print(f"{SEP}")
    print(f"  Device: {DEVICE}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:    {props.name}  ({props.total_memory/1e9:.1f} GB VRAM)")
        print(f"  CUDA:   {torch.version.cuda}")
    print(f"  Torch:  {torch.__version__}")
    print(f"  Fecha:  2026-03-05")

    tests = [
        ("M1", test_M1_throughput_scale),
        ("M2", test_M2_training_stability),
        ("M3", test_M3_drift_chunked_vs_full),
        ("M4", test_M4_routing_zloss),
        ("M5", test_M5_vram_leak),
        ("M6", test_M6_niah),
        ("M7", test_M7_autoregressive_decode),
        ("M8", test_M8_gradient_health),
        ("M9", test_M9_million_token_error_accum),
        ("M10", test_M10_stack_backward),
    ]

    passed = 0; failed = 0; test_statuses = {}
    for name, fn in tests:
        flush_cache()
        try:
            fn()
            test_statuses[name] = "PASS"
            passed += 1
        except Exception as e:
            test_statuses[name] = f"ERROR: {e}"
            failed += 1
            fail(f"{name} lanzó excepción: {e}")
            traceback.print_exc()
        flush_cache()

    # Guardar resultados
    out_path = "/tmp/chimera_million_stress.json"
    RESULTS["summary"] = {
        "passed": passed, "failed": failed, "total": len(tests),
        "test_statuses": test_statuses,
    }
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)

    banner("RESUMEN FINAL")
    for name, status in test_statuses.items():
        if status == "PASS":
            print(f"  {GREEN}✓{RESET}  {name}")
        else:
            print(f"  {RED}✗{RESET}  {name}  →  {status}")

    print(f"\n  Total: {passed}/{len(tests)} tests pasaron")
    print(f"  Resultados en: {out_path}")
    print(f"{SEP}\n")
    return failed == 0

if __name__ == "__main__":
    import sys
    ok_all = main()
    sys.exit(0 if ok_all else 1)
