"""
profile_chimera_components.py — Profiling por componente con CUDA Events
=========================================================================

Instrumenta cada subsistema de AdvancedChimeraLayer individualmente:
  Mamba2, Router, TTT, SLR, Archive, SDTM, Bus, CrossAttn, Norms

Usa torch.cuda.Event(enable_timing=True) para medición precisa kernel-level.

Ejecutar:
    cd /home/OrthoSSM/chimera_experiment
    /home/OrthoSSM/venv/bin/python tests/profile_chimera_components.py
"""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from chimera_config import ChimeraConfig
from chimera_lm import ChimeraLM

DEVICE = torch.device('cuda')
torch.set_float32_matmul_precision('high')
# TF32 explicit (necesario en algunas builds de PyTorch donde default es False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(42)

VOCAB_SIZE = 4096
SEQ_LEN    = 256
BATCH      = 2
N_WARMUP   = 5
N_MEASURE  = 15
N_COMPILE_WARMUP = 20  # torch.compile needs ~15 steps to compile all subgraphs


def make_config():
    return ChimeraConfig(
        d_model=256, n_layers=4, expand=2, headdim=32,
        d_state=128, bus_dim=256, max_landmarks=512, sdtm_n_heads=4,
        sdtm_d_mem=0, lr=3e-4, warmup_steps=50, max_seq_len=512,
    )


def cuda_time_ms(start_event, end_event):
    """Tiempo en ms entre dos CUDA events."""
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)


def profile_full_forward_backward():
    """Fase 1: Timing total de fwd + bwd + optimizer."""
    print("=" * 70)
    print("  FASE 1: Timing total fwd + bwd + optimizer step")
    print("=" * 70)

    cfg = make_config()
    model = ChimeraLM(cfg, vocab_size=VOCAB_SIZE).to(DEVICE)
    # fused=True: kernel CUDA unificado para el paso de optimizer (~15% más rápido)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.1f}M")
    print(f"  VRAM post-init: {torch.cuda.memory_allocated()/1e6:.0f} MB")

    # Warmup
    for _ in range(N_WARMUP):
        x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)
        logits, loss, _ = model(x, labels=x, aux_weight=0.01)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Measure
    fwd_times = []
    bwd_times = []
    opt_times = []
    total_times = []

    for i in range(N_MEASURE):
        x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)

        e_start = torch.cuda.Event(enable_timing=True)
        e_fwd   = torch.cuda.Event(enable_timing=True)
        e_bwd   = torch.cuda.Event(enable_timing=True)
        e_opt   = torch.cuda.Event(enable_timing=True)

        e_start.record()
        logits, loss, loss_dict = model(x, labels=x, aux_weight=0.01)
        e_fwd.record()
        loss.backward()
        e_bwd.record()
        optimizer.step()
        optimizer.zero_grad()
        e_opt.record()

        torch.cuda.synchronize()
        fwd_times.append(e_start.elapsed_time(e_fwd))
        bwd_times.append(e_fwd.elapsed_time(e_bwd))
        opt_times.append(e_bwd.elapsed_time(e_opt))
        total_times.append(e_start.elapsed_time(e_opt))

    avg = lambda lst: sum(lst) / len(lst)
    tokens_per_iter = BATCH * SEQ_LEN

    print(f"\n  {'Phase':<15}  {'Avg (ms)':>10}  {'Min':>8}  {'Max':>8}  {'%':>6}")
    print(f"  {'-'*55}")
    total_avg = avg(total_times)
    for name, times in [("Forward", fwd_times), ("Backward", bwd_times), ("Optimizer", opt_times), ("TOTAL", total_times)]:
        a = avg(times)
        pct = 100 * a / total_avg
        print(f"  {name:<15}  {a:10.2f}  {min(times):8.2f}  {max(times):8.2f}  {pct:5.1f}%")

    print(f"\n  Throughput: {tokens_per_iter / (total_avg / 1e3):.0f} tok/s (train)")
    print(f"  VRAM peak: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")

    return model


def profile_layer_components(model):
    """Fase 2: Profiling por componente dentro de una layer."""
    print()
    print("=" * 70)
    print("  FASE 2: Breakdown por componente dentro de AdvancedChimeraLayer")
    print("=" * 70)

    # Tomamos layer 0 para profiling
    layer = model.stack.layers[0]
    d_model = layer.d_model
    bus_dim = layer.bus.bus_dim if hasattr(layer, 'bus') and layer.bus is not None else 256

    # Input — float32 para evitar problemas de dtype en componentes aislados
    x = torch.randn(BATCH, SEQ_LEN, d_model, device=DEVICE)
    bus_in = torch.zeros(BATCH, SEQ_LEN, bus_dim, device=DEVICE)

    def timed(fn, warmup=3, reps=N_MEASURE):
        """Medir fn con CUDA events, retorna avg ms."""
        for _ in range(warmup):
            try:
                fn()
            except Exception:
                return -1.0
        torch.cuda.synchronize()

        times = []
        for _ in range(reps):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1))
        return sum(times) / len(times)

    components = {}

    # ── Full layer forward (baseline) ────────────────────────────────────────
    components["Full Layer"] = timed(lambda: layer(x, bus_cache=None, return_aux=True))

    # ── RMSNorm ──────────────────────────────────────────────────────────────
    components["RMSNorm"] = timed(lambda: layer.norm(x))

    # ── Mamba2 (SSM) ────────────────────────────────────────────────────────
    h = layer.norm(x)
    components["Mamba2 SSM"] = timed(lambda: layer.mamba2(h))

    # ── Router ───────────────────────────────────────────────────────────────
    components["Router"] = timed(lambda: layer.router(h))

    # ── SLR DiffAttn ─────────────────────────────────────────────────────────
    if hasattr(layer, 'slr') and layer.slr is not None:
        scan_out = layer.mamba2(h)
        importance = scan_out.detach().norm(dim=-1)
        components["SLR DiffAttn"] = timed(lambda: layer.slr(scan_out, importance))
    else:
        components["SLR DiffAttn"] = 0.0

    # ── Archive ──────────────────────────────────────────────────────────────
    if hasattr(layer, 'archive') and layer.archive is not None:
        scan_out = layer.mamba2(h)
        components["Archive"] = timed(lambda: layer.archive.maybe_archive(scan_out, step=0))
    else:
        components["Archive"] = 0.0

    # ── SDTM ─────────────────────────────────────────────────────────────────
    if hasattr(layer, 'sdtm') and layer.sdtm is not None:
        scan_out = layer.mamba2(h)
        surprise = torch.randn(BATCH, SEQ_LEN, device=DEVICE).abs()
        def sdtm_fn():
            layer.sdtm.update_memory_inplace(scan_out.detach(), surprise)
            return layer.sdtm.query(scan_out[:, -1:, :])
        components["SDTM"] = timed(sdtm_fn)
    else:
        components["SDTM"] = 0.0

    # ── Bus ──────────────────────────────────────────────────────────────────
    if hasattr(layer, 'bus') and layer.bus is not None:
        scan_out = layer.mamba2(h)
        components["Bus R/W"] = timed(lambda: layer.bus(scan_out, bus_in))
    else:
        components["Bus R/W"] = 0.0

    # Report
    print(f"\n  {'Component':<25}  {'Avg (ms)':>10}  {'%':>7}  Bar")
    print(f"  {'-'*65}")
    # Excluir "Full Layer" del total para descomponer partes
    parts = {k: v for k, v in components.items() if k != "Full Layer" and v > 0}
    parts_total = sum(parts.values()) or 1.0

    # Imprimir Full Layer primero
    full = components.get("Full Layer", 0)
    print(f"  {'Full Layer (ref)':<25}  {full:10.3f}  {'---':>7}")
    print(f"  {'-'*65}")

    for name, t in sorted(parts.items(), key=lambda x: -x[1]):
        pct = 100 * t / parts_total
        bar = "#" * int(pct / 2)
        print(f"  {name:<25}  {t:10.3f}  {pct:6.1f}%  {bar}")
    print(f"  {'Parts SUM':<25}  {parts_total:10.3f}  100.0%")

    # Overhead: full layer - parts sum
    overhead = full - parts_total
    if full > 0:
        print(f"  {'Overhead/misc':<25}  {overhead:10.3f}  {100*overhead/full:6.1f}%")

    return components


def profile_full_forward_components(model):
    """Fase 3: Instrumentar forward completo separando embedding/layers/head."""
    print()
    print("=" * 70)
    print("  FASE 3: Forward breakdown (embedding → layers × N → lm_head)")
    print("=" * 70)

    x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)

    # Warmup
    for _ in range(3):
        model(x, labels=x, aux_weight=0.01)
    torch.cuda.synchronize()

    embed_times = []
    layer_times = [[] for _ in range(len(model.stack.layers))]
    head_times = []
    loss_times = []

    for _ in range(N_MEASURE):
        # Embedding
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        h = model.embedding(x)
        e1.record()
        torch.cuda.synchronize()
        embed_times.append(e0.elapsed_time(e1))

        # Layers
        bus = None
        for li, layer in enumerate(model.stack.layers):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            h, bus, aux_list = layer(h, bus_cache=bus, return_aux=True)
            e1.record()
            torch.cuda.synchronize()
            layer_times[li].append(e0.elapsed_time(e1))

        # Final norm
        h = model.norm_f(h.float())

        # LM head
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        logits = model.lm_head(h)
        e1.record()
        torch.cuda.synchronize()
        head_times.append(e0.elapsed_time(e1))

        # Loss
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
        e1.record()
        torch.cuda.synchronize()
        loss_times.append(e0.elapsed_time(e1))

    avg = lambda lst: sum(lst) / len(lst)

    total = avg(embed_times)
    for lt in layer_times:
        total += avg(lt)
    total += avg(head_times) + avg(loss_times)

    print(f"\n  {'Stage':<25}  {'Avg (ms)':>10}  {'%':>7}")
    print(f"  {'-'*50}")
    print(f"  {'Embedding':<25}  {avg(embed_times):10.3f}  {100*avg(embed_times)/total:6.1f}%")
    for i, lt in enumerate(layer_times):
        a = avg(lt)
        print(f"  {'Layer '+str(i):<25}  {a:10.3f}  {100*a/total:6.1f}%")
    print(f"  {'LM Head':<25}  {avg(head_times):10.3f}  {100*avg(head_times)/total:6.1f}%")
    print(f"  {'CE Loss':<25}  {avg(loss_times):10.3f}  {100*avg(loss_times)/total:6.1f}%")
    print(f"  {'TOTAL (fwd only)':<25}  {total:10.3f}  100.0%")

    return total


def profile_kernel_launch_overhead():
    """Fase 4: Medir overhead de kernel launch vs compute."""
    print()
    print("=" * 70)
    print("  FASE 4: Kernel launch overhead analysis")
    print("=" * 70)

    cfg = make_config()
    model = ChimeraLM(cfg, vocab_size=VOCAB_SIZE).to(DEVICE)

    # Count total ops in one forward
    x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)

    # Medir wall-clock vs CUDA time para detectar CPU-bound
    torch.cuda.synchronize()

    # Warmup
    for _ in range(3):
        logits, loss, _ = model(x, labels=x, aux_weight=0.01)
        loss.backward()
    torch.cuda.synchronize()

    # Measure wall-clock
    wall_times = []
    cuda_times = []
    for _ in range(N_MEASURE):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)

        t0 = time.perf_counter()
        e0.record()
        logits, loss, _ = model(x, labels=x, aux_weight=0.01)
        loss.backward()
        e1.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        wall_times.append((t1 - t0) * 1000)
        cuda_times.append(e0.elapsed_time(e1))

    avg_wall = sum(wall_times) / len(wall_times)
    avg_cuda = sum(cuda_times) / len(cuda_times)
    overhead_pct = 100 * (avg_wall - avg_cuda) / avg_wall if avg_wall > 0 else 0

    print(f"  Avg wall-clock: {avg_wall:.2f} ms")
    print(f"  Avg CUDA time:  {avg_cuda:.2f} ms")
    print(f"  CPU overhead:   {avg_wall - avg_cuda:.2f} ms ({overhead_pct:.1f}%)")

    if overhead_pct > 30:
        print("  ⚠ CPU-bound: >30% overhead. Python/PyTorch dispatch dominates.")
        print("    → torch.compile(fullgraph=True) puede reducir overhead de dispatch")
        print("    → CUDA Graphs pueden eliminar overhead de launch completamente")
    elif overhead_pct > 15:
        print("  ⚡ Moderate overhead. torch.compile puede ayudar.")
    else:
        print("  ✓ GPU-bound: overhead de launch es bajo.")


def profile_bf16_training(eager_fwd_ms: float = 0.0, eager_throughput: float = 0.0):
    """
    Parte C: Mide el potencial de BF16 Mixed Precision.

    BF16 Tensor Cores en Ada Lovelace (RTX 4050) ofrecen hasta 2× más TFLOPS
    que FP32. Esto se materializa en todas las operaciones matriciales (Linear,
    Mamba2, SLR, Bus) pero NO en operaciones elementales (RMSNorm, gating).
    Esperado: 1.3-1.8× speedup en training throughput sobre FP32 compiled.
    """
    print()
    print("=" * 70)
    print("  PARTE C: BF16 compiled training vs FP32 compiled")
    print("=" * 70)

    cfg = make_config()
    model = ChimeraLM(cfg, vocab_size=VOCAB_SIZE).to(DEVICE).bfloat16()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.1f}M  dtype=bfloat16")
    print(f"  VRAM post-init: {torch.cuda.memory_allocated()/1e6:.0f} MB")

    print("  Compilando modelo BF16 con torch.compile(mode='default')...")
    try:
        model.compile_for_training(mode='default')
        print("  ✓ Compilación BF16 exitosa")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return

    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    torch.cuda.reset_peak_memory_stats()

    print(f"  Warmup {N_COMPILE_WARMUP} steps BF16...")
    for _ in range(N_COMPILE_WARMUP):
        x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)
        logits, loss, _ = model(x, labels=x, aux_weight=0.01)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if getattr(model, '_compiled', False):
            model.post_compile_step()
    torch.cuda.synchronize()
    print("  ✓ Warmup BF16 completo")

    fwd_times = []
    bwd_times = []
    total_times = []
    for _ in range(N_MEASURE):
        x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e0.record()
        logits, loss, _ = model(x, labels=x, aux_weight=0.01)
        e1.record()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if getattr(model, '_compiled', False):
            model.post_compile_step()
        e2.record()
        torch.cuda.synchronize()
        fwd_times.append(e0.elapsed_time(e1))
        bwd_times.append(e1.elapsed_time(e2))
        total_times.append(e0.elapsed_time(e2))

    avg = lambda lst: sum(lst) / len(lst)
    tokens_per_iter = BATCH * SEQ_LEN
    bf16_total_avg  = avg(total_times)
    bf16_fwd_avg    = avg(fwd_times)
    bf16_throughput = tokens_per_iter / (bf16_total_avg / 1e3)

    print(f"\n  {'Phase':<15}  {'Avg (ms)':>10}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*50}")
    for name, times in [("Forward BF16", fwd_times), ("Bwd+Opt BF16", bwd_times), ("TOTAL BF16", total_times)]:
        a = avg(times)
        print(f"  {name:<15}  {a:10.2f}  {min(times):8.2f}  {max(times):8.2f}")

    print(f"\n  Throughput BF16: {bf16_throughput:.0f} tok/s")
    print(f"  VRAM peak (BF16): {torch.cuda.max_memory_allocated()/1e6:.0f} MB")

    if eager_fwd_ms > 0:
        print(f"\n  {'Métrica':<35}  {'FP32 Eager':>12}  {'BF16 Compiled':>14}  {'Speedup':>10}")
        print(f"  {'-'*75}")
        print(f"  {'Fwd total (ms)':<35}  {eager_fwd_ms:12.2f}  {bf16_fwd_avg:14.2f}  {eager_fwd_ms/max(bf16_fwd_avg,0.01):9.2f}×")
        print(f"  {'Train throughput (tok/s)':<35}  {eager_throughput:12.0f}  {bf16_throughput:14.0f}  {bf16_throughput/max(eager_throughput,1):9.2f}×")

    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e6
    print(f"GPU: {gpu_name} ({vram_total:.0f} MB)")
    print(f"PyTorch: {torch.__version__}")
    print(f"B={BATCH}, S={SEQ_LEN}, vocab={VOCAB_SIZE}")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    # PARTE A: BASELINE (eager mode — sin torch.compile)
    # ══════════════════════════════════════════════════════════════════════════
    print("╔" + "═" * 68 + "╗")
    print("║  PARTE A: BASELINE (eager mode)                                   ║")
    print("╚" + "═" * 68 + "╝")

    model = profile_full_forward_backward()
    print()
    components = profile_layer_components(model)
    print()
    fwd_total_eager = profile_full_forward_components(model)
    print()
    profile_kernel_launch_overhead()

    # Guardar métricas baseline
    eager_throughput = BATCH * SEQ_LEN / (fwd_total_eager / 1e3) if fwd_total_eager > 0 else 0
    parts_eager = {k: v for k, v in components.items() if k != "Full Layer" and v > 0}
    parts_sum_eager = sum(parts_eager.values()) or 1
    full_layer_eager = components.get("Full Layer", 0)
    overhead_eager = full_layer_eager - parts_sum_eager

    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ══════════════════════════════════════════════════════════════════════════
    # PARTE B: CON torch.compile (reduce-overhead mode)
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  PARTE B: torch.compile(mode='default')                            ║")
    print("╚" + "═" * 68 + "╝")

    model_compiled = profile_compiled_training()
    if model_compiled is not None:
        print()
        fwd_total_compiled = profile_full_forward_components(model_compiled)
        compiled_throughput = BATCH * SEQ_LEN / (fwd_total_compiled / 1e3) if fwd_total_compiled > 0 else 0
        del model_compiled
    else:
        fwd_total_compiled = fwd_total_eager
        compiled_throughput = eager_throughput
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # RESUMEN COMPARATIVO
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("  RESUMEN COMPARATIVO: EAGER vs COMPILED")
    print("=" * 70)

    if components:
        parts = {k: v for k, v in components.items() if k != "Full Layer" and v > 0}
        if parts:
            bottleneck = max(parts, key=parts.get)
            parts_total = sum(parts.values())
            bottleneck_pct = 100 * parts[bottleneck] / parts_total if parts_total > 0 else 0
            print(f"  Bottleneck principal (per-layer): {bottleneck} ({bottleneck_pct:.0f}%)")
            full = components.get("Full Layer", 0)
            overhead = full - parts_total
            if overhead > 0 and full > 0:
                overhead_pct = 100 * overhead / full
                print(f"  Overhead (routing/TTT/cross-attn/residual): {overhead:.1f} ms ({overhead_pct:.0f}%)")

    print(f"\n  {'Metric':<30}  {'Eager':>12}  {'Compiled':>12}  {'Speedup':>10}")
    print(f"  {'-'*68}")
    print(f"  {'Fwd total (ms)':<30}  {fwd_total_eager:12.2f}  {fwd_total_compiled:12.2f}  {fwd_total_eager/max(fwd_total_compiled,0.01):9.2f}×")
    print(f"  {'Fwd throughput (tok/s)':<30}  {eager_throughput:12.0f}  {compiled_throughput:12.0f}  {compiled_throughput/max(eager_throughput,1):9.2f}×")
    print(f"  {'Overhead eager (ms)':<30}  {overhead_eager:12.2f}  {'N/A':>12}")

    if fwd_total_compiled < fwd_total_eager * 0.9:
        print(f"\n  ✓ torch.compile dio speedup de {fwd_total_eager/fwd_total_compiled:.1f}×")
    elif fwd_total_compiled > fwd_total_eager * 1.1:
        print(f"\n  ⚠ torch.compile fue más lento (overhead de compilación o graph breaks)")
    else:
        print(f"\n  ≈ Diferencia marginal — el compilador no pudo optimizar significativamente")

    # ══════════════════════════════════════════════════════════════════════════
    # PARTE C: BF16 — cuantifica el potencial de precisión reducida
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  PARTE C: BF16 Mixed Precision (mayor throughput computacional)    ║")
    print("╚" + "═" * 68 + "╝")
    profile_bf16_training(fwd_total_eager, eager_throughput)


def profile_compiled_training():
    """Fase COMPILE: Training con torch.compile(mode='reduce-overhead')."""
    print()
    print("=" * 70)
    print("  FASE COMPILE: Timing total fwd + bwd + optimizer (compiled)")
    print("=" * 70)

    cfg = make_config()
    model = ChimeraLM(cfg, vocab_size=VOCAB_SIZE).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.1f}M")

    # Aplicar torch.compile — mode='default' para estabilidad con custom autograd funcs
    print("  Compilando modelo con torch.compile(mode='default')...")
    try:
        model.compile_for_training(mode='default')
        print("  ✓ Compilación exitosa")
    except Exception as e:
        print(f"  ✗ Error de compilación: {e}")
        return None

    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Warmup extendido — con ~16+ subgrafos (4 layers × graph breaks en mamba2/SLR/etc)
    # se necesitan ~15 steps para compilar todos los subgrafos via Inductor
    print(f"  Warmup {N_COMPILE_WARMUP} steps (compilación JIT de ~16+ subgrafos)...")
    for i in range(N_COMPILE_WARMUP):
        x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)
        logits, loss, _ = model(x, labels=x, aux_weight=0.01)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if hasattr(model, '_compiled') and model._compiled:
            model.post_compile_step()
    torch.cuda.synchronize()
    print("  ✓ Warmup completo (todos los subgrafos compilados)")

    print(f"  VRAM post-warmup: {torch.cuda.memory_allocated()/1e6:.0f} MB")

    # Measure
    fwd_times = []
    bwd_times = []
    opt_times = []
    total_times = []

    for i in range(N_MEASURE):
        x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)

        e_start = torch.cuda.Event(enable_timing=True)
        e_fwd = torch.cuda.Event(enable_timing=True)
        e_bwd = torch.cuda.Event(enable_timing=True)
        e_opt = torch.cuda.Event(enable_timing=True)

        e_start.record()
        logits, loss, loss_dict = model(x, labels=x, aux_weight=0.01)
        e_fwd.record()
        loss.backward()
        e_bwd.record()
        optimizer.step()
        optimizer.zero_grad()
        if hasattr(model, '_compiled') and model._compiled:
            model.post_compile_step()
        e_opt.record()

        torch.cuda.synchronize()
        fwd_times.append(e_start.elapsed_time(e_fwd))
        bwd_times.append(e_fwd.elapsed_time(e_bwd))
        opt_times.append(e_bwd.elapsed_time(e_opt))
        total_times.append(e_start.elapsed_time(e_opt))

    avg = lambda lst: sum(lst) / len(lst)
    tokens_per_iter = BATCH * SEQ_LEN

    print(f"\n  {'Phase':<15}  {'Avg (ms)':>10}  {'Min':>8}  {'Max':>8}  {'%':>6}")
    print(f"  {'-'*55}")
    total_avg = avg(total_times)
    for name, times in [("Forward", fwd_times), ("Backward", bwd_times), ("Optimizer", opt_times), ("TOTAL", total_times)]:
        a = avg(times)
        pct = 100 * a / total_avg
        print(f"  {name:<15}  {a:10.2f}  {min(times):8.2f}  {max(times):8.2f}  {pct:5.1f}%")

    print(f"\n  Throughput: {tokens_per_iter / (total_avg / 1e3):.0f} tok/s (compiled train)")
    print(f"  VRAM peak: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")

    return model


if __name__ == '__main__':
    main()
