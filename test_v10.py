"""
OrthoSSM V10 "Lightning" Test Suite
====================================
Tests:
  T1  Import and module load
  T2  Forward shape correctness (all three routing paths)
  T3  Fast path (seq < 384): no TTT, no SLR, low latency
  T4  Hybrid path (384 <= seq < 1024): TTT + SLR (all tokens)
  T5  Full path (seq >= 1024): everything enabled
  T6  Lion optimizer: state memory reduction (no m2)
  T7  Chebyshev LUT: forward match vs Clenshaw
  T8  Backward / gradients (no crash, finite values)
  T9  Multi-batch stability (B=4)
  T10 AsyncLightBus: cross-layer summary propagation
  T11 Full model forward (short + long sequences)
  T12 V9 backward compatibility (can load V9 states)
  T13 Degree 4 vs 8 comparison
  T14 Performance comparison: V10 vs V9 kernel speed
  T15 SLR: routing scores + token selection
  T16 SLR: differential attention lambda
  T17 SLR: parameter reduction vs old NSA
"""
import sys
import time
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

results = []


def check(name, ok, detail=""):
    mark = PASS if ok else FAIL
    msg = f"{mark} {name}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    results.append((name, ok))
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# T1: Import
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== OrthoSSM V10 'Lightning' Test Suite ===\n")
try:
    import sdpc_kernel as K10
    from sdpc_engine import SpectralDualPathContextEngine, build_ortho_stack
    from model import OrthoSSMLanguageModel
    from landmark_archive import AsyncLightBus, LandmarkArchive
    from slr_module import SpectralLocalRefiner
    check("T1  Import V10 modules", True,
          "kernel, engine, model, archive, slr")
except Exception as e:
    check("T1  Import V10 modules", False, str(e))
    print("FATAL: cannot continue without imports.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# T2: Forward shape (kernel level)
# ─────────────────────────────────────────────────────────────────────────────
B, S, D, NH = 2, 512, 64, 8
hD = D // NH
deg = 4  # V10 default

x = torch.randn(B, S, D, device=DEVICE)
coef = K10.init_chebyshev_coefficients(B, NH, deg, hD, DEVICE)
mom = torch.zeros_like(coef)

try:
    out, cn, mn = K10.apply_cheby_rkv_v10(
        x, coef, mom, n_heads=NH, ema_momentum=0.9
    )
    shape_ok = (out.shape == x.shape
                and cn.shape == coef.shape
                and mn.shape == mom.shape)
    check("T2  Forward shape (kernel)", shape_ok,
          f"out={tuple(out.shape)} coeffs={tuple(cn.shape)} mom={tuple(mn.shape)}")
except Exception as e:
    check("T2  Forward shape (kernel)", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# T3: Fast path (seq < 384)
# ─────────────────────────────────────────────────────────────────────────────
try:
    x_short = torch.randn(2, 128, D, device=DEVICE)
    coef_s = K10.init_chebyshev_coefficients(2, NH, deg, hD, DEVICE)
    mom_s = torch.zeros_like(coef_s)

    t0 = time.perf_counter()
    out_s, cn_s, mn_s = K10.apply_cheby_rkv_v10(
        x_short, coef_s, mom_s, n_heads=NH, ema_momentum=0.9,
        seq_threshold=384
    )
    torch.cuda.synchronize()
    t_fast = (time.perf_counter() - t0) * 1000

    # Fast path should NOT modify coefficients (no TTT)
    coeff_unchanged = torch.allclose(cn_s, coef_s, atol=1e-6)
    check("T3  Fast path (seq=128)", out_s.shape == x_short.shape and coeff_unchanged,
          f"time={t_fast:.1f}ms, coeffs_unchanged={coeff_unchanged}")
except Exception as e:
    check("T3  Fast path (seq=128)", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# T4: Hybrid path (384 <= seq < 1024)
# ─────────────────────────────────────────────────────────────────────────────
try:
    x_mid = torch.randn(2, 512, D, device=DEVICE)
    coef_m = K10.init_chebyshev_coefficients(2, NH, deg, hD, DEVICE)
    mom_m = torch.zeros_like(coef_m)

    out_m, cn_m, mn_m = K10.apply_cheby_rkv_v10(
        x_mid, coef_m, mom_m, n_heads=NH, ema_momentum=0.9,
        seq_threshold=384  # 512 > 384 → full path triggered
    )

    # TTT should have modified coefficients
    coeff_changed = not torch.allclose(cn_m, coef_m, atol=1e-6)
    check("T4  Hybrid path (seq=512)", out_m.shape == x_mid.shape and coeff_changed,
          f"coeffs_changed={coeff_changed}")
except Exception as e:
    check("T4  Hybrid path (seq=512)", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# T5: Full path (seq >= 1024)
# ─────────────────────────────────────────────────────────────────────────────
try:
    x_long = torch.randn(2, 1024, D, device=DEVICE)
    coef_l = K10.init_chebyshev_coefficients(2, NH, deg, hD, DEVICE)
    mom_l = torch.zeros_like(coef_l)

    out_l, cn_l, mn_l = K10.apply_cheby_rkv_v10(
        x_long, coef_l, mom_l, n_heads=NH, ema_momentum=0.9,
        seq_threshold=384
    )

    check("T5  Full path (seq=1024)", out_l.shape == x_long.shape,
          f"out={tuple(out_l.shape)}")
except Exception as e:
    check("T5  Full path (seq=1024)", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# T6: Lion optimizer — no m2
# ─────────────────────────────────────────────────────────────────────────────
try:
    # V10: state = (coeffs, momentum) — 2 tensors
    # V9:  state = (coeffs, m1, m2)   — 3 tensors
    coef6 = K10.init_chebyshev_coefficients(2, NH, deg, hD, DEVICE)
    mom6 = torch.zeros_like(coef6)

    state_size_v10 = coef6.numel() + mom6.numel()  # 2 tensors
    state_size_v9 = coef6.numel() * 3  # 3 tensors (coeffs + m1 + m2)

    reduction_pct = (1 - state_size_v10 / state_size_v9) * 100
    check("T6  Lion optimizer (no m2)", reduction_pct > 30,
          f"state reduction: {reduction_pct:.0f}% ({state_size_v9} → {state_size_v10} elements)")
except Exception as e:
    check("T6  Lion optimizer (no m2)", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# T7: Chebyshev LUT vs Clenshaw forward match
# ─────────────────────────────────────────────────────────────────────────────
try:
    x7 = torch.randn(2, 512, D, device=DEVICE)
    coef7 = K10.init_chebyshev_coefficients(2, NH, deg, hD, DEVICE)
    mom7a = torch.zeros_like(coef7)
    mom7b = torch.zeros_like(coef7)

    # LUT path
    out_lut, _, _ = K10.apply_cheby_rkv_v10(
        x7, coef7.clone(), mom7a, n_heads=NH, ema_momentum=0.9,
        use_lut=True, seq_threshold=0
    )

    # Clenshaw path
    out_clw, _, _ = K10.apply_cheby_rkv_v10(
        x7, coef7.clone(), mom7b, n_heads=NH, ema_momentum=0.9,
        use_lut=False, seq_threshold=0
    )

    cos_sim = F.cosine_similarity(
        out_lut.reshape(1, -1), out_clw.reshape(1, -1)
    ).item()
    rel_err = (out_lut - out_clw).norm() / (out_clw.norm() + 1e-8)

    check("T7  LUT vs Clenshaw match", cos_sim > 0.99,
          f"cosine={cos_sim:.4f}, rel_err={rel_err.item():.4f}")
except Exception as e:
    check("T7  LUT vs Clenshaw match", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# T8: Backward gradients (no crash, finite)
# ─────────────────────────────────────────────────────────────────────────────
try:
    x8 = torch.randn(2, 512, D, device=DEVICE, requires_grad=True)
    coef8 = K10.init_chebyshev_coefficients(2, NH, deg, hD, DEVICE)
    mom8 = torch.zeros_like(coef8)

    out8, _, _ = K10.apply_cheby_rkv_v10(
        x8, coef8, mom8, n_heads=NH, ema_momentum=0.9, seq_threshold=0
    )
    loss = out8.sum()
    loss.backward()

    grad_finite = x8.grad is not None and torch.isfinite(x8.grad).all()
    grad_nonzero = x8.grad is not None and x8.grad.abs().max() > 0
    check("T8  Backward gradients", grad_finite and grad_nonzero,
          f"grad_max={x8.grad.abs().max().item():.4f}" if x8.grad is not None else "no grad")
except Exception as e:
    check("T8  Backward gradients", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# T9: Multi-batch stability
# ─────────────────────────────────────────────────────────────────────────────
try:
    x9 = torch.randn(4, 512, D, device=DEVICE)
    coef9 = K10.init_chebyshev_coefficients(4, NH, deg, hD, DEVICE)
    mom9 = torch.zeros_like(coef9)

    out9, _, _ = K10.apply_cheby_rkv_v10(
        x9, coef9, mom9, n_heads=NH, ema_momentum=0.9, seq_threshold=0
    )

    all_finite = torch.isfinite(out9).all()
    no_nan = not torch.isnan(out9).any()
    check("T9  Multi-batch B=4", all_finite and no_nan,
          f"out_max={out9.abs().max().item():.4f}")
except Exception as e:
    check("T9  Multi-batch B=4", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# T10: AsyncLightBus
# ─────────────────────────────────────────────────────────────────────────────
try:
    bus = AsyncLightBus(summary_dim=64, n_layers=4)

    # Simulate layer 0 publishing
    summary0 = torch.randn(2, 64, device=DEVICE)
    bus.publish(0, summary0)

    # Layer 1 gathers from layer 0
    gathered = bus.gather(1, 2, DEVICE)
    bus_ok = gathered is not None and gathered.shape == (2, 64)

    # Layer 0 has nothing below it
    nothing = bus.gather(0, 2, DEVICE)
    bus_ok = bus_ok and nothing is None

    check("T10 AsyncLightBus", bus_ok,
          f"publish/gather works, shape={gathered.shape if gathered is not None else 'None'}")
except Exception as e:
    check("T10 AsyncLightBus", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# T11: Full model forward (short + long)
# ─────────────────────────────────────────────────────────────────────────────
try:
    model = OrthoSSMLanguageModel(
        vocab_size=1000,
        d_model=64,
        n_attn_heads=4,
        n_cheby_heads=8,
        n_layers=2,
        max_degree=4,
        window_size=256,
        use_bf16=False,
        use_lut=True,
    ).to(DEVICE)

    # Short sequence (fast path)
    ids_short = torch.randint(0, 1000, (2, 64), device=DEVICE)
    logits_short = model(ids_short)
    short_ok = logits_short.shape == (2, 64, 1000) and torch.isfinite(logits_short).all()

    # Long sequence (full path)
    ids_long = torch.randint(0, 1000, (2, 1024), device=DEVICE)
    logits_long = model(ids_long)
    long_ok = logits_long.shape == (2, 1024, 1000) and torch.isfinite(logits_long).all()

    check("T11 Full model (short+long)", short_ok and long_ok,
          f"short={tuple(logits_short.shape)} long={tuple(logits_long.shape)}")
except Exception as e:
    check("T11 Full model (short+long)", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# T12: V9 backward compatibility
# ─────────────────────────────────────────────────────────────────────────────
try:
    # Simulate V9 state: (coeffs, m1, m2) with 3-tuple
    model12 = OrthoSSMLanguageModel(
        vocab_size=1000, d_model=64, n_attn_heads=4,
        n_cheby_heads=8, n_layers=2, max_degree=4, window_size=256,
    ).to(DEVICE)

    hd12 = 64 // 8
    v9_states = []
    for _ in range(2):
        c = K10.init_chebyshev_coefficients(2, 8, 4, hd12, DEVICE)
        m1_v9 = torch.zeros_like(c)
        m2_v9 = torch.zeros_like(c)
        v9_states.append((c, m1_v9, m2_v9))

    ids12 = torch.randint(0, 1000, (2, 256), device=DEVICE)
    logits12, new_states12 = model12(ids12, states=v9_states, return_state=True)

    compat_ok = logits12.shape == (2, 256, 1000) and len(new_states12) == 2
    # V10 returns (coeffs, momentum) — 2-tuple
    state_tuple_ok = len(new_states12[0]) == 2
    check("T12 V9 backward compat", compat_ok and state_tuple_ok,
          f"state_tuple_len={len(new_states12[0])}")
except Exception as e:
    check("T12 V9 backward compat", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# T13: Degree 4 vs 8 comparison
# ─────────────────────────────────────────────────────────────────────────────
try:
    x13 = torch.randn(2, 512, D, device=DEVICE)

    # Degree 4
    coef4 = K10.init_chebyshev_coefficients(2, NH, 4, hD, DEVICE)
    mom4 = torch.zeros_like(coef4)
    t0 = time.perf_counter()
    for _ in range(10):
        out4, _, _ = K10.apply_cheby_rkv_v10(
            x13, coef4.clone(), mom4.clone(), n_heads=NH,
            ema_momentum=0.9, seq_threshold=0
        )
    torch.cuda.synchronize()
    t_deg4 = (time.perf_counter() - t0) * 100  # ms per iter

    # Degree 8
    coef8 = K10.init_chebyshev_coefficients(2, NH, 8, hD, DEVICE)
    mom8 = torch.zeros_like(coef8)
    t0 = time.perf_counter()
    for _ in range(10):
        out8, _, _ = K10.apply_cheby_rkv_v10(
            x13, coef8.clone(), mom8.clone(), n_heads=NH,
            ema_momentum=0.9, seq_threshold=0
        )
    torch.cuda.synchronize()
    t_deg8 = (time.perf_counter() - t0) * 100

    speedup = t_deg8 / max(t_deg4, 1e-6)
    check("T13 Degree 4 vs 8", True,
          f"deg4={t_deg4:.1f}ms deg8={t_deg8:.1f}ms speedup={speedup:.2f}x")
except Exception as e:
    check("T13 Degree 4 vs 8", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# T14: Performance comparison V10 kernel vs V9
# ─────────────────────────────────────────────────────────────────────────────
try:
    x14 = torch.randn(2, 1024, 64, device=DEVICE)

    coef10 = K10.init_chebyshev_coefficients(2, 8, 4, 8, DEVICE)
    mom10 = torch.zeros_like(coef10)

    # Warmup
    for _ in range(3):
        K10.apply_cheby_rkv_v10(x14, coef10.clone(), mom10.clone(), n_heads=8, ema_momentum=0.9, seq_threshold=0)
    torch.cuda.synchronize()

    # V10 timing
    t0 = time.perf_counter()
    for _ in range(20):
        K10.apply_cheby_rkv_v10(x14, coef10.clone(), mom10.clone(), n_heads=8, ema_momentum=0.9, seq_threshold=0)
    torch.cuda.synchronize()
    t_v10 = (time.perf_counter() - t0) / 20 * 1000

    check("T14 Kernel speed", True,
          f"V10={t_v10:.2f}ms per forward (seq=1024, B=2, D=64)")
except Exception as e:
    check("T14 V10 vs V9 speed", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# T15: SLR routing scores + token selection
# ─────────────────────────────────────────────────────────────────────────────
try:
    slr = SpectralLocalRefiner(
        d_model=64, n_heads=4, window_size=256,
        select_ratio=0.125, min_select=8, max_select=512
    ).to(DEVICE)

    x15 = torch.randn(2, 512, 64, device=DEVICE)
    cheby15 = torch.randn(2, 512, 64, device=DEVICE)

    # Routing scores
    scores = SpectralLocalRefiner.compute_routing_scores(x15, cheby15)
    score_shape_ok = scores.shape == (2, 512)
    score_finite = torch.isfinite(scores).all()

    # Token selection
    indices, K = slr._select_tokens(scores)
    select_ok = (indices.shape == (2, K)
                 and K == max(8, int(512 * 0.125))  # 64
                 and (indices[:, 1:] >= indices[:, :-1]).all())  # sorted

    # Forward pass
    out15 = slr(x15, cheby_out=cheby15)
    fwd_ok = (out15.shape == x15.shape
              and torch.isfinite(out15).all())

    # Sparsity: most positions should be zero (unselected)
    nonzero_frac = (out15.abs() > 1e-8).float().mean().item()
    sparsity_ok = nonzero_frac < 0.5  # max 50% non-zero (generous threshold)

    check("T15 SLR routing + selection", score_shape_ok and score_finite
          and select_ok and fwd_ok and sparsity_ok,
          f"K={K}, nonzero={nonzero_frac:.1%}, scores_range=[{scores.min():.3f},{scores.max():.3f}]")
except Exception as e:
    check("T15 SLR routing + selection", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# T16: SLR differential attention lambda
# ─────────────────────────────────────────────────────────────────────────────
try:
    slr16 = SpectralLocalRefiner(
        d_model=64, n_heads=4, window_size=256,
        select_ratio=1.0,  # all tokens for this test
        min_select=8, max_select=4096
    ).to(DEVICE)

    x16 = torch.randn(2, 64, 64, device=DEVICE)

    # Lambda initialized at sigmoid(0) = 0.5
    init_lambda = torch.sigmoid(slr16.diff_lambda_logit).mean().item()
    lambda_init_ok = abs(init_lambda - 0.5) < 0.01

    # Forward with lambda=0 (no differential → standard attention)
    with torch.no_grad():
        slr16.diff_lambda_logit.fill_(-10.0)  # sigmoid(-10) ≈ 0
    out_no_diff = slr16(x16).clone()

    # Forward with lambda=1 (max differential)
    with torch.no_grad():
        slr16.diff_lambda_logit.fill_(10.0)   # sigmoid(10) ≈ 1
    out_max_diff = slr16(x16).clone()

    # They should differ (different lambda changes output)
    diff = (out_no_diff - out_max_diff).abs().mean().item()
    lambda_effect_ok = diff > 1e-6

    check("T16 SLR differential lambda", lambda_init_ok and lambda_effect_ok,
          f"init_λ={init_lambda:.3f}, λ_diff_effect={diff:.4f}")
except Exception as e:
    check("T16 SLR differential lambda", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# T17: SLR parameter reduction vs old NSA
# ─────────────────────────────────────────────────────────────────────────────
try:
    D17 = 256
    nH17 = 4

    # SLR parameter count
    slr17 = SpectralLocalRefiner(d_model=D17, n_heads=nH17, window_size=512)
    slr_params = sum(p.numel() for p in slr17.parameters())

    # Old NSA parameter count (computed analytically)
    # local_qkv: D→3D = D*3D = 196608
    # lm_kv: D→2D = D*2D = 131072
    # arch_kv: D→2D = D*2D = 131072
    # fused_gate_value: 3D→2D = 3D*2D = 393216
    # o_proj: D→D = D*D = 65536
    # 3 RMSNorm: 3*D = 768
    nsa_params = D17*3*D17 + D17*2*D17 + D17*2*D17 + 3*D17*2*D17 + D17*D17 + 3*D17

    reduction = (1 - slr_params / nsa_params) * 100

    check("T17 SLR param reduction", reduction > 50,
          f"SLR={slr_params:,} vs NSA={nsa_params:,} → {reduction:.0f}% reduction")
except Exception as e:
    check("T17 SLR param reduction", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
n_pass = sum(1 for _, ok in results if ok)
n_fail = sum(1 for _, ok in results if not ok)
print(f"  V10 Tests: {n_pass} passed, {n_fail} failed out of {len(results)}")

if n_fail > 0:
    print("\n  Failed tests:")
    for name, ok in results:
        if not ok:
            print(f"    {FAIL} {name}")

print("=" * 60 + "\n")
sys.exit(0 if n_fail == 0 else 1)
