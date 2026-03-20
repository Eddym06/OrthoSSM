"""
tests/test_triton_perf.py — Benchmarks de rendimiento Triton vs PyTorch
========================================================================

Nivel 1 de análisis profundo de Chimera:
  - diff_attn_v2_triton vs PyTorch cuBLAS reference
  - lion_constrained_update_inplace vs PyTorch puro
  - compute_token_errors_triton vs PyTorch puro
  - FlashDiffSLRFunction (batched 2D) vs PyTorch bmm reference

Usa triton.testing.perf_report para generar reportes de throughput (GB/s)
y latency (ms) con barrido paramétrico automático.

Ejecutar:
    cd /home/OrthoSSM/chimera_experiment
    python tests/test_triton_perf.py

Resultados: tablas + plots (si matplotlib disponible) en tests/perf_results/
"""

import sys
import os
import math

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_CHIMERA = os.path.dirname(_HERE)
if _CHIMERA not in sys.path:
    sys.path.insert(0, _CHIMERA)

import torch
import torch.nn.functional as F
import triton
import triton.testing

from sgr_slr import diff_attn_v2_triton, FlashDiffSLRFunction
from ttt_kernel import lion_constrained_update_inplace, compute_token_errors_triton


DEVICE = "cuda"
DTYPE  = torch.float32

# ── Output dir para plots ────────────────────────────────────────────────────
PERF_DIR = os.path.join(_HERE, "perf_results")
os.makedirs(PERF_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark 1: diff_attn_v2_triton vs PyTorch (unbatched, [K,d] × [W,d])
# ═══════════════════════════════════════════════════════════════════════════════

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["K_size"],
        x_vals=[64, 128, 256, 512, 1024, 2048],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="GB/s",
        plot_name="diff_attn_v2_throughput",
        args={"W_size": 64, "d_head": 32},
    )
)
def bench_diff_attn_v2(K_size, W_size, d_head, provider):
    """Throughput de diff_attn_v2: Triton flash kernel vs PyTorch softmax+bmm."""
    Q1 = torch.randn(K_size, d_head, device=DEVICE, dtype=DTYPE)
    Q2 = torch.randn(K_size, d_head, device=DEVICE, dtype=DTYPE)
    K1 = torch.randn(W_size, d_head, device=DEVICE, dtype=DTYPE)
    K2 = torch.randn(W_size, d_head, device=DEVICE, dtype=DTYPE)
    V  = torch.randn(W_size, d_head, device=DEVICE, dtype=DTYPE)
    lam = 0.5

    if provider == "triton":
        fn = lambda: diff_attn_v2_triton(Q1, Q2, K1, K2, V, lam)
    else:
        scale = 1.0 / math.sqrt(d_head)
        def torch_ref():
            s1 = (Q1.float() @ K1.float().T) * scale
            s2 = (Q2.float() @ K2.float().T) * scale
            a1 = torch.softmax(s1, dim=-1)
            a2 = torch.softmax(s2, dim=-1)
            return a1 @ V.float() - lam * (a2 @ V.float())
        fn = torch_ref

    ms = triton.testing.do_bench(fn, warmup=50, rep=200)
    # Memoria total leída+escrita (approx):
    #   Read:  Q1,Q2[K,d] + K1,K2,V[W,d] = (2K+3W)*d*4 bytes
    #   Write: Out[K,d] = K*d*4 bytes
    total_bytes = ((2 * K_size + 3 * W_size + K_size) * d_head * 4)
    gbps = total_bytes / (ms * 1e-3) / 1e9
    return gbps


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark 2: diff_attn_v2_triton scaling con d_head
# ═══════════════════════════════════════════════════════════════════════════════

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["d_head"],
        x_vals=[16, 32, 64, 128],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="ms",
        plot_name="diff_attn_v2_latency_vs_dhead",
        args={"K_size": 512, "W_size": 64},
    )
)
def bench_diff_attn_v2_dhead(K_size, W_size, d_head, provider):
    """Latencia de diff_attn_v2 variando d_head."""
    Q1 = torch.randn(K_size, d_head, device=DEVICE, dtype=DTYPE)
    Q2 = torch.randn(K_size, d_head, device=DEVICE, dtype=DTYPE)
    K1 = torch.randn(W_size, d_head, device=DEVICE, dtype=DTYPE)
    K2 = torch.randn(W_size, d_head, device=DEVICE, dtype=DTYPE)
    V  = torch.randn(W_size, d_head, device=DEVICE, dtype=DTYPE)
    lam = 0.5

    if provider == "triton":
        fn = lambda: diff_attn_v2_triton(Q1, Q2, K1, K2, V, lam)
    else:
        scale = 1.0 / math.sqrt(d_head)
        def torch_ref():
            s1 = (Q1.float() @ K1.float().T) * scale
            s2 = (Q2.float() @ K2.float().T) * scale
            a1 = torch.softmax(s1, dim=-1)
            a2 = torch.softmax(s2, dim=-1)
            return a1 @ V.float() - lam * (a2 @ V.float())
        fn = torch_ref

    ms = triton.testing.do_bench(fn, warmup=50, rep=200)
    return ms


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark 3: FlashDiffSLRFunction batched vs PyTorch bmm
# ═══════════════════════════════════════════════════════════════════════════════

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["K_size"],
        x_vals=[32, 64, 128, 256, 512],
        line_arg="provider",
        line_vals=["triton_flash", "torch_bmm"],
        line_names=["Triton Flash 2D", "PyTorch BMM"],
        styles=[("green", "-"), ("orange", "--")],
        ylabel="ms",
        plot_name="flash_diff_slr_batched_latency",
        args={"B": 2, "W_size": 64, "d_head": 32},
    )
)
def bench_flash_diff_slr(B, K_size, W_size, d_head, provider):
    """Batched Flash-Diff SLR: Triton 2D kernel vs PyTorch cuBLAS."""
    Q1 = torch.randn(B, K_size, d_head, device=DEVICE, dtype=DTYPE, requires_grad=False)
    Q2 = torch.randn(B, K_size, d_head, device=DEVICE, dtype=DTYPE, requires_grad=False)
    K1 = torch.randn(B, W_size, d_head, device=DEVICE, dtype=DTYPE, requires_grad=False)
    K2 = torch.randn(B, W_size, d_head, device=DEVICE, dtype=DTYPE, requires_grad=False)
    V  = torch.randn(B, W_size, d_head, device=DEVICE, dtype=DTYPE, requires_grad=False)
    lam_logit = torch.zeros(1, device=DEVICE, dtype=DTYPE, requires_grad=False)

    if provider == "triton_flash":
        fn = lambda: FlashDiffSLRFunction.apply(Q1, Q2, K1, K2, V, lam_logit)
    else:
        scale = 1.0 / math.sqrt(d_head)
        lam = torch.sigmoid(lam_logit).item()
        def torch_bmm_ref():
            s1 = torch.bmm(Q1.float(), K1.float().transpose(-1, -2)) * scale
            s2 = torch.bmm(Q2.float(), K2.float().transpose(-1, -2)) * scale
            A1 = torch.softmax(s1, dim=-1)
            A2 = torch.softmax(s2, dim=-1)
            return torch.bmm(A1 - lam * A2, V.float())
        fn = torch_bmm_ref

    ms = triton.testing.do_bench(fn, warmup=50, rep=200)
    return ms


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark 4: lion_constrained_update_inplace vs PyTorch puro
# ═══════════════════════════════════════════════════════════════════════════════

def _lion_pytorch_ref(dt_bias, momentum, grad, A_abs, beta, lr, active_prob):
    """Implementación PyTorch pura equivalente al kernel Triton Lion."""
    new_mom = beta * momentum + (1.0 - beta) * grad
    sign_step = torch.sign(new_mom)
    raw_upd = lr * sign_step * active_prob
    max_delta = 0.1 * A_abs
    clamped = torch.clamp(raw_upd, -max_delta, max_delta)
    dt_bias.sub_(clamped)
    momentum.copy_(new_mom)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["nheads"],
        x_vals=[8, 16, 32, 64, 128, 256, 512, 1024],
        line_arg="provider",
        line_vals=["triton", "triton_kahan", "torch"],
        line_names=["Triton", "Triton+Kahan", "PyTorch"],
        styles=[("blue", "-"), ("cyan", "-."), ("red", "--")],
        ylabel="us",
        plot_name="lion_update_latency",
        args={},
    )
)
def bench_lion_update(nheads, provider):
    """Latencia de Lion constrained update variando número de heads."""
    dt_bias  = torch.randn(nheads, device=DEVICE, dtype=DTYPE)
    momentum = torch.zeros(nheads, device=DEVICE, dtype=DTYPE)
    grad     = torch.randn(nheads, device=DEVICE, dtype=DTYPE) * 0.1
    A_log    = torch.linspace(-6, -1, nheads, device=DEVICE)
    A_abs    = torch.exp(A_log).abs()
    mom_comp = torch.zeros(nheads, device=DEVICE, dtype=DTYPE)
    dt_comp  = torch.zeros(nheads, device=DEVICE, dtype=DTYPE)

    if provider == "triton":
        fn = lambda: lion_constrained_update_inplace(
            dt_bias.clone(), momentum.clone(), grad, A_log,
            beta=0.9, lr=1e-3, active_prob=0.7,
        )
    elif provider == "triton_kahan":
        fn = lambda: lion_constrained_update_inplace(
            dt_bias.clone(), momentum.clone(), grad, A_log,
            beta=0.9, lr=1e-3, active_prob=0.7,
            mom_comp=mom_comp.clone(), dt_comp=dt_comp.clone(),
        )
    else:
        fn = lambda: _lion_pytorch_ref(
            dt_bias.clone(), momentum.clone(), grad, A_abs,
            beta=0.9, lr=1e-3, active_prob=0.7,
        )

    ms = triton.testing.do_bench(fn, warmup=100, rep=500)
    return ms * 1e3  # devolver en microsegundos


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark 5: compute_token_errors_triton vs PyTorch L2 norm
# ═══════════════════════════════════════════════════════════════════════════════

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[64, 128, 256, 512, 1024, 2048],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton 2D", "PyTorch"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="GB/s",
        plot_name="token_errors_throughput",
        args={"B": 2, "D": 256},
    )
)
def bench_token_errors(B, seq_len, D, provider):
    """Throughput de compute_token_errors: Triton 2D grid vs torch.norm."""
    Sm1 = seq_len - 1
    pred   = torch.randn(B, Sm1, D, device=DEVICE, dtype=DTYPE)
    target = torch.randn(B, Sm1, D, device=DEVICE, dtype=DTYPE)

    if provider == "triton":
        fn = lambda: compute_token_errors_triton(pred, target)
    else:
        fn = lambda: torch.norm(pred - target, dim=-1)

    ms = triton.testing.do_bench(fn, warmup=50, rep=200)
    # Read: 2 × B × Sm1 × D × 4 bytes; Write: B × Sm1 × 4 bytes
    total_bytes = (2 * B * Sm1 * D + B * Sm1) * 4
    gbps = total_bytes / (ms * 1e-3) / 1e9
    return gbps


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark 6: Correctness — Triton vs PyTorch (no es perf, pero lo necesitamos)
# ═══════════════════════════════════════════════════════════════════════════════

def test_diff_attn_correctness():
    """Verifica que Triton y PyTorch producen el mismo resultado (atol<1e-3)."""
    print("=" * 70)
    print("  CORRECTNESS: diff_attn_v2_triton vs PyTorch")
    print("=" * 70)

    for K, W, d in [(64, 32, 32), (256, 64, 32), (512, 128, 64), (1024, 256, 32)]:
        torch.manual_seed(42)
        Q1 = torch.randn(K, d, device=DEVICE, dtype=DTYPE)
        Q2 = torch.randn(K, d, device=DEVICE, dtype=DTYPE)
        K1 = torch.randn(W, d, device=DEVICE, dtype=DTYPE)
        K2 = torch.randn(W, d, device=DEVICE, dtype=DTYPE)
        V  = torch.randn(W, d, device=DEVICE, dtype=DTYPE)
        lam = 0.5

        out_triton = diff_attn_v2_triton(Q1, Q2, K1, K2, V, lam)

        scale = 1.0 / math.sqrt(d)
        s1 = (Q1.float() @ K1.float().T) * scale
        s2 = (Q2.float() @ K2.float().T) * scale
        a1 = torch.softmax(s1, dim=-1)
        a2 = torch.softmax(s2, dim=-1)
        out_ref = (a1 @ V.float() - lam * (a2 @ V.float()))

        max_err = (out_triton.float() - out_ref).abs().max().item()
        status = "PASS" if max_err < 1e-2 else "FAIL"
        print(f"  K={K:4d}  W={W:3d}  d={d:2d}  max_err={max_err:.2e}  [{status}]")


def test_lion_correctness():
    """Verifica que Triton Lion produce resultado idéntico a PyTorch."""
    print("=" * 70)
    print("  CORRECTNESS: lion_constrained_update_inplace vs PyTorch")
    print("=" * 70)

    for N in [8, 16, 64, 128]:
        torch.manual_seed(42)
        dt_t  = torch.randn(N, device=DEVICE, dtype=DTYPE)
        mom_t = torch.zeros(N, device=DEVICE, dtype=DTYPE)
        grad  = torch.randn(N, device=DEVICE, dtype=DTYPE) * 0.1
        A_log = torch.linspace(-6, -1, N, device=DEVICE)
        A_abs = torch.exp(A_log).abs()

        # PyTorch reference
        dt_ref  = dt_t.clone()
        mom_ref = mom_t.clone()
        _lion_pytorch_ref(dt_ref, mom_ref, grad, A_abs, 0.9, 1e-3, 0.7)

        # Triton
        dt_tri  = dt_t.clone()
        mom_tri = mom_t.clone()
        lion_constrained_update_inplace(dt_tri, mom_tri, grad, A_log,
                                         beta=0.9, lr=1e-3, active_prob=0.7)

        dt_err  = (dt_tri - dt_ref).abs().max().item()
        mom_err = (mom_tri - mom_ref).abs().max().item()
        status = "PASS" if dt_err < 1e-5 and mom_err < 1e-5 else "FAIL"
        print(f"  N={N:4d}  dt_err={dt_err:.2e}  mom_err={mom_err:.2e}  [{status}]")


def test_token_errors_correctness():
    """Verifica compute_token_errors_triton vs torch.norm."""
    print("=" * 70)
    print("  CORRECTNESS: compute_token_errors_triton vs torch.norm")
    print("=" * 70)

    for B, S, D in [(1, 64, 128), (2, 256, 256), (4, 512, 64)]:
        torch.manual_seed(42)
        Sm1 = S - 1
        pred   = torch.randn(B, Sm1, D, device=DEVICE, dtype=DTYPE)
        target = torch.randn(B, Sm1, D, device=DEVICE, dtype=DTYPE)

        out_triton = compute_token_errors_triton(pred, target)
        out_ref    = torch.norm(pred - target, dim=-1)

        max_err = (out_triton - out_ref).abs().max().item()
        status = "PASS" if max_err < 1e-3 else "FAIL"
        print(f"  B={B}  S={S}  D={D}  max_err={max_err:.2e}  [{status}]")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA requerido para benchmarks Triton"

    gpu_name = torch.cuda.get_device_name(0)
    vram_mb  = torch.cuda.get_device_properties(0).total_memory / 1e6
    print(f"GPU: {gpu_name} ({vram_mb:.0f} MB)")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print()

    # ── Fase 1: Correctness ──────────────────────────────────────────────────
    test_diff_attn_correctness()
    print()
    test_lion_correctness()
    print()
    test_token_errors_correctness()
    print()

    # ── Fase 2: Performance benchmarks ───────────────────────────────────────
    print("=" * 70)
    print("  PERFORMANCE BENCHMARKS")
    print("=" * 70)
    print()

    print(">>> Benchmark 1: diff_attn_v2 throughput (K sweep, W=64, d=32)")
    bench_diff_attn_v2.run(
        print_data=True,
        save_path=os.path.join(PERF_DIR, "diff_attn_v2_throughput"),
    )
    print()

    print(">>> Benchmark 2: diff_attn_v2 latency vs d_head (K=512, W=64)")
    bench_diff_attn_v2_dhead.run(
        print_data=True,
        save_path=os.path.join(PERF_DIR, "diff_attn_v2_latency_vs_dhead"),
    )
    print()

    print(">>> Benchmark 3: Flash-Diff SLR batched (B=2, W=64, d=32)")
    bench_flash_diff_slr.run(
        print_data=True,
        save_path=os.path.join(PERF_DIR, "flash_diff_slr_batched"),
    )
    print()

    print(">>> Benchmark 4: Lion update latency (nheads sweep)")
    bench_lion_update.run(
        print_data=True,
        save_path=os.path.join(PERF_DIR, "lion_update_latency"),
    )
    print()

    print(">>> Benchmark 5: Token errors throughput (seq_len sweep, B=2, D=256)")
    bench_token_errors.run(
        print_data=True,
        save_path=os.path.join(PERF_DIR, "token_errors_throughput"),
    )
    print()

    print("=" * 70)
    print("  BENCHMARKS COMPLETADOS — resultados en tests/perf_results/")
    print("=" * 70)
