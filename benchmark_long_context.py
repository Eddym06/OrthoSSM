"""
OrthoSSM V9 — Ultra Long Context Benchmark
============================================
Tests OrthoSSM kernel at extreme sequence lengths:
  128K, 256K, 512K, 1M tokens

Measures:
  - Forward pass time (ms)
  - Backward pass time (ms) 
  - VRAM peak (MB)
  - Numerical stability (output finite, bounded)
  - Throughput (tokens/sec)
  - Memory per token (bytes)

RTX 4050 6GB constraints:
  - B=1, D=64 (minimal model) to maximize sequence length
  - Forward-only for very long sequences (backward needs 2x memory)
  - Incremental: try each length, report OOM gracefully
"""

import torch
import time
import sys
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

def gpu_mem_mb():
    if DEVICE == "cuda":
        return torch.cuda.memory_allocated() / 1e6
    return 0

def gpu_peak_mb():
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    return 0

def gpu_reset_peak():
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

def format_time(ms):
    if ms < 1:
        return f"{ms*1000:.0f}µs"
    elif ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms/1000:.2f}s"

def format_mem(mb):
    if mb < 1:
        return f"{mb*1024:.0f}KB"
    elif mb < 1024:
        return f"{mb:.1f}MB"
    else:
        return f"{mb/1024:.2f}GB"


print("=" * 70)
print("  OrthoSSM V9 — ULTRA LONG CONTEXT BENCHMARK")
print("=" * 70)

if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory / 1e9
    print(f"  GPU: {props.name}")
    print(f"  VRAM: {total_vram:.1f} GB")
    print(f"  Compute: SM{props.major}{props.minor}")
else:
    print("  WARNING: No CUDA device. Running on CPU (very slow).")
    total_vram = 0

print("=" * 70)

# Import kernel
try:
    import sdpc_kernel as K
    print(f"  {PASS} Kernel V9 loaded")
except Exception as e:
    print(f"  {FAIL} Cannot load kernel: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Minimal model dimensions to maximize sequence length on 6GB GPU
B = 1
D = 64
NH = 8
hD = D // NH  # 8
deg = 8

# Context lengths to test
FORWARD_LENGTHS = [
    (128 * 1024,   "128K"),
    (256 * 1024,   "256K"),
    (512 * 1024,   "512K"),
    (1024 * 1024,  "1M"),
]

# Smaller set for backward (needs ~2x memory)
BACKWARD_LENGTHS = [
    (128 * 1024,   "128K"),
    (256 * 1024,   "256K"),
]

results_table = []

# ─────────────────────────────────────────────────────────────────────────────
# FORWARD BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("  FORWARD PASS BENCHMARK (B={}, D={}, nH={})".format(B, D, NH))
print(f"{'─'*70}")
print(f"  {'Length':>8s}  {'Time':>10s}  {'VRAM Peak':>10s}  {'tok/s':>12s}  {'B/tok':>8s}  {'Status':>8s}")
print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*8}")

for seq_len, label in FORWARD_LENGTHS:
    gpu_reset_peak()
    
    # Estimate memory needed: x[B,S,D]*4 + y[B,S,D]*2 + out[B,S,D]*2 + misc
    est_bytes = B * seq_len * D * (4 + 2 + 2 + 2)  # FP32 input + BF16 intermediates
    est_mb = est_bytes / 1e6
    
    if est_mb > (total_vram * 1024 * 0.85):  # 85% of VRAM
        print(f"  {label:>8s}  {'—':>10s}  {'—':>10s}  {'—':>12s}  {'—':>8s}  {WARN} SKIP (est {format_mem(est_mb)})")
        results_table.append((label, "forward", None, None, None, "SKIP"))
        continue
    
    try:
        torch.manual_seed(42)
        x = torch.randn(B, seq_len, D, device=DEVICE)
        coeffs   = K.init_chebyshev_coefficients(B, NH, deg, hD, DEVICE)
        momentum = torch.zeros_like(coeffs)
        
        # Warmup
        with torch.no_grad():
            out, *_ = K.apply_cheby_rkv_core(
                x, coeffs.clone(), momentum.clone(),
                n_heads=NH, ema_momentum=0.9
            )
        torch.cuda.synchronize()
        del out
        torch.cuda.empty_cache()
        
        # Benchmark
        gpu_reset_peak()
        N_RUNS = 3
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_RUNS):
            with torch.no_grad():
                out, *_ = K.apply_cheby_rkv_core(
                    x, coeffs.clone(), momentum.clone(),
                    n_heads=NH, ema_momentum=0.9
                )
            torch.cuda.synchronize()
        t_total = (time.perf_counter() - t0) / N_RUNS * 1000  # ms
        
        peak = gpu_peak_mb()
        toks_per_sec = (B * seq_len) / (t_total / 1000)
        bytes_per_tok = (peak * 1e6) / (B * seq_len)
        
        # Stability checks
        is_finite = out.isfinite().all().item()
        is_bounded = out.abs().max().item() <= 1.0 + 1e-5
        stable = is_finite and is_bounded
        
        status = PASS if stable else FAIL
        status_txt = "OK" if stable else f"UNSTABLE(fin={is_finite},bnd={is_bounded})"
        
        print(f"  {label:>8s}  {format_time(t_total):>10s}  {format_mem(peak):>10s}  "
              f"{toks_per_sec:>12,.0f}  {bytes_per_tok:>7.1f}B  {status} {status_txt}")
        
        results_table.append((label, "forward", t_total, peak, toks_per_sec, status_txt))
        
        # Cleanup
        del x, out, coeffs, momentum
        torch.cuda.empty_cache()
        gc.collect()
        
    except torch.cuda.OutOfMemoryError:
        print(f"  {label:>8s}  {'—':>10s}  {'—':>10s}  {'—':>12s}  {'—':>8s}  {FAIL} OOM")
        results_table.append((label, "forward", None, None, None, "OOM"))
        torch.cuda.empty_cache()
        gc.collect()
        # If OOM at this length, skip longer ones
        break
    except Exception as e:
        print(f"  {label:>8s}  {'—':>10s}  {'—':>10s}  {'—':>12s}  {'—':>8s}  {FAIL} {str(e)[:40]}")
        results_table.append((label, "forward", None, None, None, str(e)[:40]))
        torch.cuda.empty_cache()
        gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("  BACKWARD PASS BENCHMARK (B={}, D={}, nH={})".format(B, D, NH))
print(f"{'─'*70}")
print(f"  {'Length':>8s}  {'Fwd':>8s}  {'Bwd':>8s}  {'Total':>8s}  {'VRAM':>10s}  {'Status':>8s}")
print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}")

for seq_len, label in BACKWARD_LENGTHS:
    gpu_reset_peak()
    
    # Backward needs ~3x forward memory (save for backward + gradients)
    est_bytes = B * seq_len * D * (4 + 2 + 2 + 4 + 4 + 4)  # generous estimate
    est_mb = est_bytes / 1e6
    
    if est_mb > (total_vram * 1024 * 0.85):
        print(f"  {label:>8s}  {'—':>8s}  {'—':>8s}  {'—':>8s}  {'—':>10s}  {WARN} SKIP")
        results_table.append((label, "backward", None, None, None, "SKIP"))
        continue
    
    try:
        torch.manual_seed(42)
        x = torch.randn(B, seq_len, D, device=DEVICE, requires_grad=True)
        coeffs = K.init_chebyshev_coefficients(B, NH, deg, hD, DEVICE)
        coeffs.requires_grad_(True)
        momentum = torch.zeros(B, NH, deg, hD, device=DEVICE)
        
        # Warmup
        out, *_ = K.apply_cheby_rkv_core(
            x, coeffs.clone(), momentum.clone(),
            n_heads=NH, ema_momentum=0.9
        )
        out.sum().backward()
        torch.cuda.synchronize()
        
        # Benchmark
        gpu_reset_peak()
        
        x_bm = x.detach().requires_grad_(True)
        c_bm = coeffs.detach().requires_grad_(True)
        
        # Forward timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, *_ = K.apply_cheby_rkv_core(
            x_bm, c_bm, momentum.clone(),
            n_heads=NH, ema_momentum=0.9
        )
        torch.cuda.synchronize()
        t_fwd = (time.perf_counter() - t0) * 1000
        
        # Backward timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out.sum().backward()
        torch.cuda.synchronize()
        t_bwd = (time.perf_counter() - t0) * 1000
        
        peak = gpu_peak_mb()
        t_total = t_fwd + t_bwd
        
        # Gradient stability
        grad_ok = (x_bm.grad is not None and 
                   x_bm.grad.isfinite().all().item() and
                   c_bm.grad is not None and
                   c_bm.grad.isfinite().all().item())
        
        status = PASS if grad_ok else FAIL
        status_txt = "OK" if grad_ok else "UNSTABLE"
        
        print(f"  {label:>8s}  {format_time(t_fwd):>8s}  {format_time(t_bwd):>8s}  "
              f"{format_time(t_total):>8s}  {format_mem(peak):>10s}  {status} {status_txt}")
        
        results_table.append((label, "backward", t_total, peak, None, status_txt))
        
        del x, out, coeffs, momentum, x_bm, c_bm
        torch.cuda.empty_cache()
        gc.collect()
        
    except torch.cuda.OutOfMemoryError:
        print(f"  {label:>8s}  {'—':>8s}  {'—':>8s}  {'—':>8s}  {'—':>10s}  {FAIL} OOM")
        results_table.append((label, "backward", None, None, None, "OOM"))
        torch.cuda.empty_cache()
        gc.collect()
        break
    except Exception as e:
        print(f"  {label:>8s}  {'—':>8s}  {'—':>8s}  {'—':>8s}  {'—':>10s}  {FAIL} {str(e)[:40]}")
        results_table.append((label, "backward", None, None, None, str(e)[:40]))
        torch.cuda.empty_cache()
        gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STABILITY STRESS TEST — progressive sequence lengths
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("  NUMERICAL STABILITY STRESS TEST")
print(f"{'─'*70}")

STABILITY_LENGTHS = [1024, 4096, 16384, 65536, 131072, 262144, 524288, 1048576]
print(f"  {'Length':>8s}  {'max|out|':>10s}  {'mean|out|':>10s}  {'%finite':>8s}  {'Status'}")
print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*12}")

max_stable_len = 0

for seq_len in STABILITY_LENGTHS:
    gpu_reset_peak()
    label = f"{seq_len//1024}K" if seq_len >= 1024 else str(seq_len)
    
    try:
        torch.manual_seed(42)
        x = torch.randn(B, seq_len, D, device=DEVICE)
        coeffs   = K.init_chebyshev_coefficients(B, NH, deg, hD, DEVICE)
        momentum = torch.zeros_like(coeffs)
        
        with torch.no_grad():
            out, *_ = K.apply_cheby_rkv_core(
                x, coeffs, momentum,
                n_heads=NH, ema_momentum=0.9
            )
        
        max_out = out.abs().max().item()
        mean_out = out.abs().mean().item()
        pct_finite = out.isfinite().float().mean().item() * 100
        bounded = max_out <= 1.0 + 1e-5
        stable = pct_finite == 100.0 and bounded
        
        status = PASS if stable else FAIL
        detail = "STABLE" if stable else f"EXCEEDED(max={max_out:.2f})"
        
        print(f"  {label:>8s}  {max_out:>10.6f}  {mean_out:>10.6f}  {pct_finite:>7.1f}%  {status} {detail}")
        
        if stable:
            max_stable_len = seq_len
        
        del x, out, coeffs, momentum
        torch.cuda.empty_cache()
        gc.collect()
        
    except torch.cuda.OutOfMemoryError:
        print(f"  {label:>8s}  {'—':>10s}  {'—':>10s}  {'—':>8s}  {WARN} OOM at {format_mem(gpu_peak_mb())}")
        torch.cuda.empty_cache()
        gc.collect()
        break
    except Exception as e:
        print(f"  {label:>8s}  {'—':>10s}  {'—':>10s}  {'—':>8s}  {FAIL} {str(e)[:50]}")
        torch.cuda.empty_cache()
        gc.collect()
        break

# ─────────────────────────────────────────────────────────────────────────────
# SCALING ANALYSIS: Memory & Time vs Context Length
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("  SCALING ANALYSIS — Memory & Time vs Context Length")
print(f"{'─'*70}")

SCALE_LENGTHS = [1024, 4096, 16384, 65536, 131072]
scale_data = []

print(f"  {'Length':>8s}  {'Time(ms)':>10s}  {'VRAM(MB)':>10s}  {'MB/token':>10s}  {'time/S':>10s}")
print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

for seq_len in SCALE_LENGTHS:
    gpu_reset_peak()
    label = f"{seq_len//1024}K" if seq_len >= 1024 else str(seq_len)
    
    try:
        torch.manual_seed(42)
        x = torch.randn(B, seq_len, D, device=DEVICE)
        coeffs   = K.init_chebyshev_coefficients(B, NH, deg, hD, DEVICE)
        momentum = torch.zeros_like(coeffs)
        
        # Warmup
        with torch.no_grad():
            K.apply_cheby_rkv_core(x, coeffs.clone(), momentum.clone(),
                                    n_heads=NH, ema_momentum=0.9)
        torch.cuda.synchronize()
        
        # Measure
        gpu_reset_peak()
        N = 3
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            with torch.no_grad():
                out, *_ = K.apply_cheby_rkv_core(
                    x, coeffs.clone(), momentum.clone(),
                    n_heads=NH, ema_momentum=0.9)
            torch.cuda.synchronize()
        t_ms = (time.perf_counter() - t0) / N * 1000
        
        peak = gpu_peak_mb()
        mb_per_tok = peak / (B * seq_len) * 1e6  # bytes
        time_per_tok = t_ms / seq_len * 1000  # µs per token
        
        print(f"  {label:>8s}  {t_ms:>10.2f}  {peak:>10.1f}  {mb_per_tok:>9.1f}B  {time_per_tok:>9.3f}µs")
        scale_data.append((seq_len, t_ms, peak, mb_per_tok, time_per_tok))
        
        del x, out, coeffs, momentum
        torch.cuda.empty_cache()
        gc.collect()
        
    except torch.cuda.OutOfMemoryError:
        print(f"  {label:>8s}  {'OOM':>10s}  {'—':>10s}  {'—':>10s}  {'—':>10s}")
        torch.cuda.empty_cache()
        gc.collect()
        break
    except Exception as e:
        print(f"  {label:>8s}  {str(e)[:40]:>10s}")
        break

# Check O(S) scaling
if len(scale_data) >= 3:
    # Time should scale linearly with S → time_per_tok should be ~constant
    times_per_tok = [d[4] for d in scale_data]
    ratio = times_per_tok[-1] / times_per_tok[0] if times_per_tok[0] > 0 else float('inf')
    is_linear = ratio < 3.0  # Allow up to 3x variance (cache effects)
    print(f"\n  Scaling check: time/token ratio (longest/shortest) = {ratio:.2f}")
    print(f"  {'✓' if is_linear else '✗'} {'O(S) linear scaling confirmed' if is_linear else 'Sub-linear or super-linear scaling detected'}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  SUMMARY")
print(f"{'='*70}")

max_fwd_ok = 0
max_bwd_ok = 0
for label, mode, t, peak, tps, status in results_table:
    val = int(label.replace("K", "")) * 1024 if "K" in label else (int(label.replace("M", "")) * 1024 * 1024 if "M" in label else 0)
    if status == "OK":
        if mode == "forward":
            max_fwd_ok = max(max_fwd_ok, val)
        else:
            max_bwd_ok = max(max_bwd_ok, val)

def fmt_len(n):
    if n >= 1048576:
        return f"{n // 1048576}M"
    elif n >= 1024:
        return f"{n // 1024}K"
    return str(n)

print(f"  Max stable forward:   {fmt_len(max_fwd_ok)} tokens" if max_fwd_ok > 0 else "  Max stable forward:   N/A")
print(f"  Max stable backward:  {fmt_len(max_bwd_ok)} tokens" if max_bwd_ok > 0 else "  Max stable backward:  N/A")
print(f"  Max stable numerical: {fmt_len(max_stable_len)} tokens" if max_stable_len > 0 else "  Max stable numerical: N/A")

# O(S) memory check
if scale_data:
    first_mb_per_tok = scale_data[0][3]
    last_mb_per_tok = scale_data[-1][3]
    mem_ratio = last_mb_per_tok / first_mb_per_tok if first_mb_per_tok > 0 else float('inf')
    print(f"  Memory scaling:       {mem_ratio:.2f}x (1.0 = perfect O(S))")

print(f"\n  OrthoSSM V9: Spectral state-space model with O(S) memory, O(S) time,")
print(f"  and constant-size recurrent state. No attention matrix, no O(S²) bottleneck.")
print(f"{'='*70}")
