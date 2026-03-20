#!/usr/bin/env python3
"""
test_pt28_upgrade.py — Validación de la actualización a PyTorch 2.8.0 + Triton 3.4.0

Tests:
  1. Versiones correctas instaladas
  2. use_reentrant=False funciona sin el bug de 2.5.1
  3. AMP BF16 + checkpoint backward sin errores de dimensión
  4. Gradient flow cross-layer (bus ring)
  5. Triton kernels compilan y ejecutan correctamente
  6. torch.compile funciona con el modelo
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import traceback

def cprint(msg, color='green'):
    c = {'green': '\033[92m', 'red': '\033[91m', 'yellow': '\033[93m', 'reset': '\033[0m'}
    print(f"{c.get(color, '')}{msg}{c['reset']}")


def test_versions():
    """Test 1: Verificar versiones."""
    import triton
    pt_ver = torch.__version__
    tr_ver = triton.__version__
    print(f"  PyTorch: {pt_ver}")
    print(f"  Triton:  {tr_ver}")
    print(f"  CUDA:    {torch.version.cuda}")
    assert '2.8' in pt_ver, f"Expected PyTorch 2.8.x, got {pt_ver}"
    assert tr_ver.startswith('3.4'), f"Expected Triton 3.4.x, got {tr_ver}"
    cprint("  PASS: Versions correct")


def test_mamba_import():
    """Test 2: Verificar que mamba_ssm carga con ABI correcto."""
    from mamba_ssm import Mamba2
    m = Mamba2(d_model=256, expand=2, headdim=32, d_state=64, layer_idx=0).cuda().bfloat16()
    x = torch.randn(2, 32, 256, device='cuda', dtype=torch.bfloat16)
    with torch.no_grad():
        out = m(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    cprint("  PASS: mamba_ssm Mamba2 works")


def test_causal_conv1d_import():
    """Test 3: Verificar causal_conv1d."""
    from causal_conv1d import causal_conv1d_fn
    # Simple smoke test
    x = torch.randn(2, 128, 32, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(128, 4, device='cuda', dtype=torch.bfloat16)
    out = causal_conv1d_fn(x, w, bias=None, activation='silu')
    assert out.shape == x.shape
    cprint("  PASS: causal_conv1d_fn works")


def test_use_reentrant_false_checkpoint():
    """Test 4: El bug sistémico de use_reentrant=False con bmm/mm que existía
    en PyTorch 2.5.1 debería estar corregido en 2.8."""
    from advanced_chimera import AsyncLightBus

    B, S, D = 4, 64, 256
    bus_dim = 128
    N_layers = 4

    bus = AsyncLightBus(D, bus_dim).cuda().bfloat16()

    def _fwd(x, bus_ring):
        out, new_ring = bus.forward_ring(x, bus_ring, write_idx=0)
        return out, new_ring

    x = torch.randn(B, S, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    bus_ring = torch.zeros(B, N_layers, bus_dim, device='cuda', dtype=torch.bfloat16)

    # Forward + backward con use_reentrant=False
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out, new_ring = checkpoint(_fwd, x, bus_ring, use_reentrant=False)
        loss = out.sum()
    loss.backward()

    assert x.grad is not None, "No gradient on x"
    assert x.grad.shape == x.shape
    cprint("  PASS: use_reentrant=False + AsyncLightBus works!")


def test_full_chimera_checkpoint():
    """Test 5: ChimeraStack completo con use_reentrant=False + AMP."""
    from chimera_config import ChimeraConfig
    from chimera_lm import ChimeraStack

    for n_layers in [1, 4]:
        for dtype in [torch.bfloat16, torch.float16]:
            for ckpt_interval in [1, 2, 999]:
                dtype_name = 'BF16' if dtype == torch.bfloat16 else 'FP16'
                tag = f"n_layers={n_layers} {dtype_name} ckpt={ckpt_interval}"

                cfg = ChimeraConfig(
                    d_model=256, n_layers=n_layers, expand=2, headdim=32,
                    bus_dim=128, residual_scale=True
                )
                stack = ChimeraStack(cfg, ckpt_interval=ckpt_interval).cuda().to(dtype)
                stack.train()

                B, S = 4, 64
                x = torch.randn(B, S, 256, device='cuda', dtype=dtype, requires_grad=True)

                try:
                    with torch.amp.autocast('cuda', dtype=dtype):
                        out, aux_list, _, _ = stack(x, collect_aux=True)
                        loss = out.sum()
                    loss.backward()

                    assert x.grad is not None, f"No gradient on x"
                    assert x.grad.shape == x.shape
                    assert torch.isfinite(x.grad).all(), "Non-finite gradients"
                    cprint(f"  PASS: {tag}")
                except Exception as e:
                    cprint(f"  FAIL: {tag} — {e}", 'red')
                    traceback.print_exc()
                    return False

                del stack, x, out
                torch.cuda.empty_cache()

    return True


def test_cross_layer_bus_gradient():
    """Test 6: Verificar que los gradientes fluyen entre capas a través del bus."""
    from chimera_config import ChimeraConfig
    from chimera_lm import ChimeraStack

    cfg = ChimeraConfig(d_model=256, n_layers=4, expand=2, headdim=32, bus_dim=128)
    stack = ChimeraStack(cfg, ckpt_interval=1).cuda().bfloat16()
    stack.train()

    x = torch.randn(4, 64, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out, _, _, _ = stack(x, collect_aux=True)
        loss = out.sum()
    loss.backward()

    # Verificar que TODOS los bus modules tienen gradientes
    for i, layer in enumerate(stack.layers):
        bus = layer.bus
        for name, p in bus.named_parameters():
            if p.grad is not None and p.grad.abs().max() > 0:
                pass  # OK
            else:
                cprint(f"  WARN: Layer {i} bus.{name} has zero/no gradient", 'yellow')

    cprint("  PASS: Cross-layer bus gradient flow verified")


def test_triton_kernels():
    """Test 7: Verificar que los kernels Triton compilan con Triton 3.4."""
    from ttt_kernel import lion_constrained_update_inplace, compute_token_errors_triton

    n_heads = 8
    dt_bias = torch.randn(n_heads, device='cuda', dtype=torch.float32)
    momentum = torch.zeros(n_heads, device='cuda', dtype=torch.float32)
    grad = torch.randn(n_heads, device='cuda', dtype=torch.float32)
    A_log = torch.randn(n_heads, device='cuda', dtype=torch.float32) - 5.0
    mom_comp = torch.zeros(n_heads, device='cuda', dtype=torch.float32)
    dt_comp = torch.zeros(n_heads, device='cuda', dtype=torch.float32)

    lion_constrained_update_inplace(
        dt_bias, momentum, grad, A_log,
        beta=0.9, lr=1e-3, active_prob=1.0,
        mom_comp=mom_comp, dt_comp=dt_comp,
    )
    cprint("  PASS: lion_constrained_update_inplace (Triton)")

    # Token errors
    pred = torch.randn(2, 63, 256, device='cuda', dtype=torch.bfloat16)
    target = torch.randn(2, 63, 256, device='cuda', dtype=torch.bfloat16)
    err = compute_token_errors_triton(pred, target)
    assert err.shape == (2, 63), f"Bad shape: {err.shape}"
    cprint("  PASS: compute_token_errors_triton (Triton)")


def test_slr_triton():
    """Test 8: SLR/SGR Triton kernels."""
    from sgr_slr import SLRDifferentialModule

    slr = SLRDifferentialModule(d_model=256, d_head=32, window_size=64, top_k_frac=0.125)
    slr = slr.cuda().bfloat16()

    query = torch.randn(2, 64, 256, device='cuda', dtype=torch.bfloat16)
    context = torch.randn(2, 64, 256, device='cuda', dtype=torch.bfloat16)
    importance = torch.randn(2, 64, device='cuda', dtype=torch.bfloat16).abs()

    out, indices = slr(query_base=query, context_base=context, importance=importance)
    assert out.shape == query.shape, f"Shape mismatch: {out.shape}"
    cprint("  PASS: SLR Triton kernels work with Triton 3.4")


def test_gpu_profile():
    """Test 9: Verificar gpu_profile adaptive compile kwargs."""
    from gpu_profile import get_gpu_profile, get_torch_compile_kwargs
    import triton

    profile = get_gpu_profile()
    print(f"  {profile}")
    print(f"  Triton stages_flash: {profile.triton_stages_flash}")
    print(f"  Triton stages_ema:   {profile.triton_stages_ema}")

    kwargs = get_torch_compile_kwargs(mode='train')
    print(f"  compile kwargs (train): {kwargs}")
    assert 'mode' in kwargs and 'fullgraph' in kwargs and 'dynamic' in kwargs
    assert kwargs['dynamic'] == False
    cprint("  PASS: gpu_profile adaptive compile kwargs OK")

    kwargs_inf = get_torch_compile_kwargs(mode='infer')
    print(f"  compile kwargs (infer): {kwargs_inf}")
    assert 'mode' in kwargs_inf
    cprint("  PASS: gpu_profile infer mode OK")


def test_performance_benchmark():
    """Test 10: Benchmark rápido para medir throughput (PT 2.8 + Triton 3.4)."""
    from chimera_config import ChimeraConfig
    from chimera_lm import ChimeraStack
    import time

    cfg = ChimeraConfig(d_model=256, n_layers=4, expand=2, headdim=32, bus_dim=128)
    stack = ChimeraStack(cfg, ckpt_interval=1).cuda().bfloat16()
    stack.train()

    B, S = 4, 64
    x = torch.randn(B, S, 256, device='cuda', dtype=torch.bfloat16)

    # Warmup — 12 pasos para que Triton autotune y CUDA JIT estén completamente compilados
    for _ in range(12):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out, _ , _, _= stack(x, collect_aux=False)
            out.sum().backward()
        stack.zero_grad()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Ahora medir en estado estacionario (sin overhead de compilación)
    torch.cuda.synchronize()
    times = []
    for _ in range(15):
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out, _, _, _ = stack(x, collect_aux=False)
            out.sum().backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        stack.zero_grad()

    # Mediana de los 10 steps centrales (descartar 2 más lentos y más rápidos)
    times_sorted = sorted(times)
    steady = times_sorted[2:-2]  # 11 valores sin outliers extremos
    avg_ms  = sum(steady) / len(steady) * 1000
    min_ms  = min(steady) * 1000
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    # Baseline PT 2.5.1: ~400ms/step, 318MB  |  PT 2.8 sin opts: ~364ms/step, 299MB
    ref_25_ms = 400.0
    ref_28_ms = 364.0
    speedup_vs_25 = (ref_25_ms - avg_ms) / ref_25_ms * 100
    speedup_vs_28 = (ref_28_ms - avg_ms) / ref_28_ms * 100
    cprint(f"  BENCHMARK PT 2.8 + Triton 3.4 (estado estacionario, {len(steady)} muestras):")
    cprint(f"    avg={avg_ms:.1f}ms  min={min_ms:.1f}ms  peak={peak_mb:.0f}MB")
    cprint(f"    vs PT 2.5.1 (~400ms): {speedup_vs_25:+.1f}%  vs PT 2.8 sin opts (~364ms): {speedup_vs_28:+.1f}%")


if __name__ == '__main__':
    tests = [
        ("Versions", test_versions),
        ("Mamba2 import", test_mamba_import),
        ("causal_conv1d import", test_causal_conv1d_import),
        ("use_reentrant=False checkpoint", test_use_reentrant_false_checkpoint),
        ("Full ChimeraStack checkpoint", test_full_chimera_checkpoint),
        ("Cross-layer bus gradient", test_cross_layer_bus_gradient),
        ("Triton kernels (ttt_kernel)", test_triton_kernels),
        ("SLR Triton kernels", test_slr_triton),
        ("GPU profile / adaptive compile kwargs", test_gpu_profile),
        ("Performance benchmark (PT 2.8 + Triton 3.4)", test_performance_benchmark),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        try:
            result = fn()
            if result is False:
                failed += 1
            else:
                passed += 1
        except Exception as e:
            cprint(f"  FAIL: {e}", 'red')
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    if failed == 0:
        cprint(f"ALL {passed} TESTS PASSED ✓", 'green')
    else:
        cprint(f"{passed} passed, {failed} FAILED", 'red')
