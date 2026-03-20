"""
OrthoSSM V11 — Precision Tests
================================
Verifica la equivalencia numérica entre:
  1. Mega-kernel (Clenshaw+EMA+TTTGrad+Lion fused) vs referencia secuencial
  2. Triton SLR kernel vs fallback PyTorch chunked
  3. BF16 guard (LUT error con/sin el guard coeffs.float())
"""
import sys, math
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VENV_PY = "/home/OrthoSSM/venv/bin/python3"
torch.manual_seed(42)

print(f"=== OrthoSSM V11 Precision Tests ===")
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

PASS, FAIL = [], []

def check(name, ok, detail=""):
    sym = "✓" if ok else "✗"
    print(f"  {sym}  {name}  [{detail}]")
    (PASS if ok else FAIL).append(name)

# ─────────────────────────────────────────────────────────────────────────────
# P1: Mega-kernel: verifica que las salidas EMA son correctas
# ─────────────────────────────────────────────────────────────────────────────
print("\n── P1: Mega-kernel EMA output precision ──")
from sdpc_kernel import apply_cheby_rkv_v10, _fused_clenshaw_ema_kernel, get_chebyshev_lut, LUT_TABLE_SIZE
import triton

B, S, D, nH, deg = 2, 256, 64, 8, 4  # S ≤ chunk único para comparación directa
x   = torch.randn(B, S, D, device=DEVICE)
coeffs0 = torch.randn(B, nH, deg, D // nH, device=DEVICE)
mom0    = torch.zeros(B, nH, deg, D // nH, device=DEVICE)

# Mega-kernel run (chunk_size=512 → única llamada a FusedChebyRKVv10, sin detach)
out_mega, coeffs_mega, mom_mega = apply_cheby_rkv_v10(
    x, coeffs0.clone(), mom0.clone(),
    n_heads=nH, use_lut=True, seq_threshold=0, chunk_size=512,
)

# Referencia: EMA pura sin TTT (solo Clenshaw+EMA, sin Lion) via _fused_clenshaw_ema_kernel
x_bf16   = x.contiguous().to(torch.bfloat16)
coeff_c  = coeffs0.contiguous()
lut_c    = get_chebyshev_lut(DEVICE).contiguous()
x_norm_ref = torch.empty(B, S, D, device=DEVICE, dtype=torch.float32)
tanh_y_ref = torch.empty(B, S, D, device=DEVICE, dtype=torch.float32)
out_ref    = torch.empty(B, S, D, device=DEVICE, dtype=torch.float32)
ema_init   = torch.zeros(B, D, device=DEVICE, dtype=torch.float32)
ema_final  = torch.empty(B, D, device=DEVICE, dtype=torch.float32)

BLOCK_HD = min(32, triton.next_power_of_2(D // nH))
n_hd_blocks = triton.cdiv(D // nH, BLOCK_HD)
grid_f   = (B * nH * n_hd_blocks,)
_fused_clenshaw_ema_kernel[grid_f](
    x_bf16, coeff_c, lut_c,
    x_norm_ref, tanh_y_ref, out_ref,
    ema_init, ema_final,
    S, D, D // nH, nH, deg, LUT_TABLE_SIZE, 0.9,
    x_bf16.stride(0),   x_bf16.stride(1),   x_bf16.stride(2),
    coeff_c.stride(0),  coeff_c.stride(1),  coeff_c.stride(2), coeff_c.stride(3),
    lut_c.stride(0),    lut_c.stride(1),
    x_norm_ref.stride(0), x_norm_ref.stride(1), x_norm_ref.stride(2),
    tanh_y_ref.stride(0), tanh_y_ref.stride(1), tanh_y_ref.stride(2),
    out_ref.stride(0),  out_ref.stride(1),  out_ref.stride(2),
    ema_init.stride(0), ema_init.stride(1),
    BLOCK_HD=BLOCK_HD, BLOCK_S=128, USE_LUT=True,
)
out_ref_f = out_ref.float()
out_mega_f = out_mega.float()

# La salida EMA debe coincidir (pre-TTT Lion)
# Los coeficientes usados en ambos son los mismos (pre-Lion)
cos_sim  = F.cosine_similarity(out_mega_f.flatten(), out_ref_f.flatten(), dim=0).item()
# Usar norma relativa al valor medio (robusta a near-zeros)
mae = (out_mega_f - out_ref_f).abs().mean().item()
sig = out_ref_f.abs().mean().item() + 1e-9
norm_err = mae / sig
check("EMA output cosine≥0.99", cos_sim >= 0.99, f"cosine={cos_sim:.5f}")
check("EMA output normerr≤0.01", norm_err <= 0.01, f"norm_err={norm_err:.5f}")

# Coefficients deben haber cambiado (Lion update)
coeffs_changed = not torch.allclose(coeffs_mega.float(), coeffs0.float(), atol=1e-6)
check("Coefficients updated by Lion", coeffs_changed, f"changed={coeffs_changed}")

# ─────────────────────────────────────────────────────────────────────────────
# P2: BF16 guard — LUT error con coeficientes BF16 vs F32
# ─────────────────────────────────────────────────────────────────────────────
print("\n── P2: BF16 coefficients guard ──")
coeffs_f32 = torch.randn(B, nH, deg, D // nH, device=DEVICE, dtype=torch.float32)
coeffs_bf16_roundtrip = coeffs_f32.to(torch.bfloat16).to(torch.float32)

out_f32,  *_ = apply_cheby_rkv_v10(x, coeffs_f32.clone(),              mom0.clone(), n_heads=nH, seq_threshold=0)
out_bf16, *_ = apply_cheby_rkv_v10(x, coeffs_bf16_roundtrip.clone(),   mom0.clone(), n_heads=nH, seq_threshold=0)

rel_err_bf16 = (out_f32.float() - out_bf16.float()).abs().max().item() / (out_f32.float().abs().max().item() + 1e-9)
# Con el guard, los coeficientes pasan por .float() al entrar → error bajo
check("BF16 guard: rel_err≤0.015", rel_err_bf16 <= 0.015, f"rel_err={rel_err_bf16:.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# P3: SLR Triton kernel vs fallback PyTorch
# ─────────────────────────────────────────────────────────────────────────────
print("\n── P3: SLR Triton diff-flash-attn vs PyTorch chunked ──")
from slr_module import SpectralLocalRefiner, slr_diff_flash_attn, _TRITON_OK

if _TRITON_OK:
    slr = SpectralLocalRefiner(d_model=64, n_heads=4, window_size=128).to(DEVICE)
    K = 128
    nH_slr, hD_slr = 4, 16
    B_slr = 2

    # Genera q1,q2,k,v y posiciones aleatorias (ordenadas para ser válidas)
    q1 = torch.randn(B_slr, nH_slr, K, hD_slr, device=DEVICE)
    q2 = torch.randn(B_slr, nH_slr, K, hD_slr, device=DEVICE)
    k  = torch.randn(B_slr, nH_slr, K, hD_slr, device=DEVICE)
    v  = torch.randn(B_slr, nH_slr, K, hD_slr, device=DEVICE)

    # Posiciones: índices ordenados en [0, 512)
    positions = torch.stack([
        torch.sort(torch.randperm(512, device=DEVICE)[:K])[0]
        for _ in range(B_slr)
    ])   # [B, K]

    diff_lambda_logit = slr.diff_lambda_logit

    # Triton kernel
    out_triton = slr_diff_flash_attn(
        q1.contiguous(), q2.contiguous(),
        k.contiguous(),  v.contiguous(),
        positions, diff_lambda_logit, slr.window_size,
    )   # [B, nH, K, hD]

    # Referencia PyTorch: _chunked_sparse_windowed_attn + diff_lambda
    out1_py, out2_py = slr._chunked_sparse_windowed_attn(q1, q2, k, v, positions)
    lamb_py = torch.sigmoid(diff_lambda_logit).view(1, nH_slr, 1, 1)
    out_py = out1_py - lamb_py * out2_py

    cos_slr = F.cosine_similarity(
        out_triton.flatten().unsqueeze(0),
        out_py.flatten().unsqueeze(0)
    ).item()
    rel_slr = (out_triton - out_py).abs().max().item() / (out_py.abs().max().item() + 1e-9)

    check("SLR Triton cosine≥0.98", cos_slr >= 0.98, f"cosine={cos_slr:.4f}")
    check("SLR Triton rel_err≤0.05", rel_slr <= 0.05, f"rel_err={rel_slr:.4f}")
else:
    check("SLR Triton NOT AVAILABLE", False, "Triton not installed")

# ─────────────────────────────────────────────────────────────────────────────
# P4: Mega-kernel gradientes
# ─────────────────────────────────────────────────────────────────────────────
print("\n── P4: Mega-kernel gradient flow ──")
# S ≤ chunk_size para que coeffs NO sean detachadas durante el loop de chunks
x_grad = torch.randn(B, 256, D, device=DEVICE, requires_grad=True)
coeffs_grad = torch.randn(B, nH, deg, D // nH, device=DEVICE, requires_grad=True)
mom_grad    = torch.zeros(B, nH, deg, D // nH, device=DEVICE)

out_g, _, _ = apply_cheby_rkv_v10(
    x_grad, coeffs_grad, mom_grad,
    n_heads=nH, seq_threshold=0, chunk_size=512,  # única pasada sin detach
)
loss = out_g.sum()
loss.backward()

grad_x_ok = x_grad.grad is not None and x_grad.grad.isfinite().all().item()
grad_c_ok = coeffs_grad.grad is not None and coeffs_grad.grad.isfinite().all().item()
check("Gradient x finite", grad_x_ok, f"max={x_grad.grad.abs().max().item():.4f}" if grad_x_ok else "None")
check("Gradient coeffs finite", grad_c_ok, f"max={coeffs_grad.grad.abs().max().item():.4f}" if grad_c_ok else "None")

# ─────────────────────────────────────────────────────────────────────────────
# P5: Latencia — mega-kernel vs separate kernels comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n── P5: Latencia mega-kernel vs baseline ──")
if DEVICE == "cuda":
    import time
    N_WARM, N_BENCH = 5, 20

    x_lat = torch.randn(2, 1024, 128, device=DEVICE)
    c_lat = torch.randn(2, 8, 4, 16, device=DEVICE)
    m_lat = torch.zeros(2, 8, 4, 16, device=DEVICE)

    for _ in range(N_WARM):
        apply_cheby_rkv_v10(x_lat, c_lat.clone(), m_lat.clone(), n_heads=8, seq_threshold=0)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        apply_cheby_rkv_v10(x_lat, c_lat.clone(), m_lat.clone(), n_heads=8, seq_threshold=0)
    torch.cuda.synchronize()
    t_mega = (time.perf_counter() - t0) / N_BENCH * 1000

    check("Mega-kernel latencia≤3ms", t_mega <= 3.0, f"{t_mega:.2f}ms/forward (B=2,S=1024,D=128)")
else:
    check("Latencia (CPU skip)", True, "no GPU")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Precision Tests: {len(PASS)} passed, {len(FAIL)} failed out of {len(PASS)+len(FAIL)}")
if FAIL:
    print(f"\n  Failed: {', '.join(FAIL)}")
print(f"{'='*60}")
sys.exit(0 if not FAIL else 1)
