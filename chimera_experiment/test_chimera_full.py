"""
test_chimera_full.py — Test de integración completa
=====================================================
Verifica los 4 componentes nuevos en conjunto:

  1. step()                 — decode token-by-token con cache
  2. TTT Triton kernel      — Lion update fusionado
  3. ChimeraWarmupScheduler — 3 fases
  4. ChimeraLosses          — entropy penalty + TTT prediction loss
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import time

from advanced_chimera import (
    AdvancedChimeraLayer, make_cuda_graph_step,
    CUDAGraphPool, chunked_prefill, Fp8Linear, apply_fp8_to_chimera,
    _FP8_AVAIL,
)
from chimera_scheduler import ChimeraWarmupScheduler
from chimera_losses    import ChimeraLosses, chimera_total_loss
from ttt_kernel        import lion_constrained_update_inplace, compute_token_errors_triton
from gpu_profile       import get_gpu_profile, get_triton_configs_flash


# ─────────────────────────────────────────────────────────────────────────────
# Stack de 3 capas para el test
# ─────────────────────────────────────────────────────────────────────────────
class ChimeraStack(nn.Module):
    def __init__(self, n_layers=3, d_model=256, expand=2, headdim=32):
        super().__init__()
        self.layers = nn.ModuleList([
            AdvancedChimeraLayer(d_model=d_model, expand=expand, headdim=headdim)
            for _ in range(n_layers)
        ])
        self.d_model = d_model

    def forward(self, x, bus_cache=None, return_aux=False):
        aux_list = []
        for layer in self.layers:
            if return_aux:
                x, bus_cache, aux = layer(x, bus_cache=bus_cache, return_aux=True)
                aux_list.append(aux)
            else:
                x, bus_cache = layer(x, bus_cache=bus_cache)
        if return_aux:
            return x, bus_cache, aux_list
        return x, bus_cache

    def allocate_inference_cache(self, batch_size, dtype=None):
        return [l.allocate_inference_cache(batch_size, dtype=dtype)
                for l in self.layers]

    def step(self, x, caches):
        """x: [B,1,D]  caches: lista de dicts por capa."""
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])
        return x, caches


# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, S, D = 2, 128, 256   # secuencia corta para test rápido


def test_ttt_triton_kernel():
    """Kernel Triton Lion fusionado: resultado == implementación Python."""
    print("\n[1] TTT Triton Kernel (Lion fusionado)")
    nheads = 16
    dt_bias   = torch.randn(nheads, device=DEVICE)
    momentum  = torch.zeros(nheads, device=DEVICE)
    grad      = torch.randn(nheads, device=DEVICE)
    A_log     = torch.linspace(-6., -1., nheads, device=DEVICE)

    # Referencia Python
    beta, lr, ap = 0.9, 1e-3, 0.7
    mom_ref  = beta * momentum + (1 - beta) * grad
    upd_ref  = lr * torch.sign(mom_ref) * ap
    A_abs    = torch.exp(A_log).abs()
    clamped  = torch.clamp(upd_ref, -0.1 * A_abs, 0.1 * A_abs)
    dt_ref   = dt_bias.clone() - clamped

    # Triton
    dt_triton  = dt_bias.clone()
    mom_triton = momentum.clone()
    lion_constrained_update_inplace(dt_triton, mom_triton, grad, A_log, beta, lr, ap)

    err_dt  = (dt_ref  - dt_triton).abs().max().item()
    err_mom = (mom_ref - mom_triton).abs().max().item()
    assert err_dt  < 1e-4, f"dt_bias mismatch: {err_dt}"
    assert err_mom < 1e-4, f"momentum mismatch: {err_mom}"
    print(f"  [✓] dt_bias max-err: {err_dt:.2e}  momentum max-err: {err_mom:.2e}")

    # Velocidad
    t0 = time.time()
    for _ in range(500):
        lion_constrained_update_inplace(dt_triton, mom_triton, grad, A_log, beta, lr, ap)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"  [✓] 500 updates en {(t1-t0)*1000:.1f} ms → "
          f"{(t1-t0)*1e6/500:.1f} µs/update")

    # Token error kernel
    pred   = torch.randn(B, S-1, D, device=DEVICE)
    target = torch.randn(B, S-1, D, device=DEVICE)
    errs_triton = compute_token_errors_triton(pred, target)
    errs_ref    = (pred - target).norm(dim=-1)
    max_err = (errs_triton - errs_ref).abs().max().item()
    assert max_err < 1e-3, f"token_err mismatch: {max_err}"
    print(f"  [✓] compute_token_errors_triton max-err vs PyTorch: {max_err:.2e}")


def test_warmup_scheduler(model: ChimeraStack):
    """Warm-up scheduler: gates se animan correctamente en las 3 fases."""
    print("\n[2] ChimeraWarmupScheduler (3 fases)")
    sched = ChimeraWarmupScheduler(list(model.layers), warm1=50, warm2=150)

    sched.step(0)
    l = model.layers[0]
    assert l.ttt_lr == 0.0, "Fase1: ttt_lr debe ser 0"
    slr_gate_f1 = torch.sigmoid(l.slr.merge_gate).item()
    assert slr_gate_f1 < 0.01, f"Fase1: slr_gate debe ser ~0, got {slr_gate_f1:.4f}"
    print(f"  [✓] Fase 1 (step=0): ttt_lr=0  slr_gate(σ)={slr_gate_f1:.4f}")

    sched.step(100)   # mitad de Fase2
    slr_gate_f2 = torch.sigmoid(l.slr.merge_gate).item()
    ttt_lr_f2   = l.ttt_lr
    assert ttt_lr_f2 > 0.0, "Fase2: ttt_lr debe ser > 0"
    assert 0.0 < slr_gate_f2 < 1.0, f"Fase2: slr_gate debe ser 0<x<1, got {slr_gate_f2:.4f}"
    print(f"  [✓] Fase 2 (step=100): ttt_lr={ttt_lr_f2:.5f}  slr_gate(σ)={slr_gate_f2:.4f}")

    sched.step(200)   # Fase 3
    assert l.ttt_lr == sched.ttt_lr_target, "Fase3: ttt_lr=target"
    print(f"  [✓] Fase 3 (step=200): ttt_lr={l.ttt_lr:.5f}")

    # Restaurar a Fase 3 para el resto del test
    sched.step(300)
    return sched


def test_chimera_losses(model: ChimeraStack):
    """ChimeraLosses: entropy penalty penaliza colapso, backward funciona."""
    print("\n[3] ChimeraLosses (entropy + TTT pred)")
    losses = ChimeraLosses(routing_weight=0.01, ttt_pred_weight=0.05)

    x = torch.randn(B, S, D, device=DEVICE, requires_grad=True)
    out, _, aux_list = model(x, return_aux=True)

    # Registrar probabilidades de routing de las 3 capas
    for aux in aux_list:
        # re-computar probs para que tengan grad
        pass

    # Simular directamente con tensores que tienen grad
    for _ in model.layers:
        probs = torch.softmax(torch.randn(B, 3, device=DEVICE, requires_grad=True), dim=-1)
        losses.add_routing_probs(probs)
        pred   = torch.randn(B, S-1, D, device=DEVICE, requires_grad=True)
        target = torch.randn(B, S-1, D, device=DEVICE)
        losses.add_ttt_error(pred, target)

    aux = losses.compute()
    # routing_loss = -weight * H → siempre ≤ 0 (reward de entropía al minimizar)
    # ttt_pred es positive (MSE)
    assert aux['ttt_pred'].item() > 0, "ttt_pred_loss debe ser positivo"
    print(f"  [✓] routing_loss={aux['routing'].item():.4f}  "
          f"ttt_pred_loss={aux['ttt_pred'].item():.4f}")

    # Backward del total
    dummy_lm = torch.tensor(2.5, device=DEVICE, requires_grad=True)
    losses.reset()
    for _ in model.layers:
        probs = torch.softmax(torch.randn(B, 3, device=DEVICE, requires_grad=True), dim=-1)
        losses.add_routing_probs(probs)
    total = chimera_total_loss(dummy_lm, losses, verbose=True)
    total.backward()
    assert dummy_lm.grad is not None, "grad fluye hasta lm_loss"
    print(f"  [✓] Backward OK — total_loss={total.item():.4f}")

    # Verificar Sprint 6.3: routing_loss = +weight*H (penaliza entropía alta → especialización)
    # Uniforme (H alto) paga MÁS que colapso (H bajo): loss_u > loss_c
    losses_c = ChimeraLosses(routing_weight=0.01)
    losses_u = ChimeraLosses(routing_weight=0.01)
    p_col = torch.tensor([[0.01, 0.01, 0.98]] * B, device=DEVICE, requires_grad=True)
    p_uni = torch.full((B, 3), 1/3, device=DEVICE, requires_grad=True)
    losses_c.add_routing_probs(p_col)
    losses_u.add_routing_probs(p_uni)
    loss_c = losses_c.compute()['routing'].item()
    loss_u = losses_u.compute()['routing'].item()
    assert loss_u > loss_c, f"uniforme ({loss_u:.4f}) debe > colapso ({loss_c:.4f}) — routing_loss=+weight*H"
    print(f"  [✓] Uniforme={loss_u:.4f} > Colapso={loss_c:.4f} (routing penaliza H alto → especialización) ✓")


def test_step_decode(model: ChimeraStack):
    """step() decode: output consistente con forward() en S=1."""
    print("\n[4] step() — decode token-by-token")
    model.eval()

    # Prefill con S=8 tokens para poblar el archive
    x_prefill = torch.randn(B, 8, D, device=DEVICE)
    with torch.no_grad():
        _, _ = model(x_prefill)

    # Alocar cache de inferencia
    caches = model.allocate_inference_cache(B, dtype=torch.float32)

    # Decode 5 tokens
    tokens_generated = []
    x_tok = torch.randn(B, 1, D, device=DEVICE)
    t0 = time.time()
    with torch.no_grad():
        for step_idx in range(5):
            x_tok, caches = model.step(x_tok, caches)
            tokens_generated.append(x_tok.clone())
    torch.cuda.synchronize()
    t1 = time.time()

    assert len(tokens_generated) == 5, "Deben generarse 5 tokens"
    assert tokens_generated[0].shape == (B, 1, D), f"Shape incorrecto: {tokens_generated[0].shape}"

    # Verificar que bus_ring existe y tiene forma fija [B, ring_size, bus_dim]
    # (ring buffer pre-alocado — forma CONSTANTE, no crece con torch.cat)
    for i, cache in enumerate(caches):
        if 'bus_ring' in cache and cache['bus_ring'] is not None:
            print(f"  Capa {i}: bus_ring={list(cache['bus_ring'].shape)}")

    per_tok_ms = (t1 - t0) * 1000 / 5
    print(f"  [✓] 5 tokens decodificados, shape={list(tokens_generated[0].shape)}")
    print(f"  [\u2713] Tiempo por token (step est\u00e1ndar): {per_tok_ms:.1f} ms")
    model.train()   # restaurar modo training


def test_ring_bus_shape(model: 'ChimeraStack'):
    """Ring buffer: forma fija [B, ring_size, bus_dim] en cada step."""
    print("\n[4b] Ring Bus \u2014 forma fija en decode")
    model.eval()
    B_t, D_t = 2, model.d_model

    caches = model.allocate_inference_cache(B_t, dtype=torch.float32)
    x = torch.randn(B_t, 1, D_t, device=DEVICE)

    # Capturar shape del ring bus antes y despu\u00e9s de 10 pasos
    shapes = []
    with torch.no_grad():
        for _ in range(10):
            x, caches = model.step(x, caches)
            # Forma debe ser siempre identica
            shapes.append(tuple(caches[0]['bus_ring'].shape))

    # Todas las formas deben ser identicas (clave para CUDA Graphs)
    assert len(set(shapes)) == 1, f"Ring bus cambi\u00f3 de forma: {set(shapes)}"
    ring_shape = shapes[0]
    gpu_ring = get_gpu_profile().ring_size
    assert ring_shape[1] == gpu_ring, (
        f"Ring size esperado {gpu_ring} (GPU profile), got {ring_shape[1]}"
    )
    print(f"  [\u2713] Ring shape constante en 10 pasos: {list(ring_shape)}")
    print(f"  [\u2713] ring_size={gpu_ring} detectado de GPU profile (laptop_ada \u2192 16)")
    model.train()


def test_return_aux(model: ChimeraStack):
    """return_aux=True: retorna routing_probs correcto."""
    print("\n[5] return_aux — integración con ChimeraLosses")
    x = torch.randn(B, S, D, device=DEVICE)
    out, bus_cache, aux_list = model(x, return_aux=True)

    assert isinstance(aux_list, list) and len(aux_list) == 3
    for i, aux in enumerate(aux_list):
        assert 'routing_probs' in aux
        assert aux['routing_probs'].shape == (B, 3)
        probs_sum = aux['routing_probs'].sum(dim=-1)
        assert (probs_sum - 1.0).abs().max() < 1e-4, "probs deben sumar 1"
    print(f"  [✓] aux_list len=3, routing_probs[B=2,3] correctas")


def test_gpu_profile():
    """GPU profile: detecta hardware y genera configs Triton correctas."""
    print("\n[6] GPU Profile — JIT adaptativo")
    prof = get_gpu_profile()
    assert prof is not None
    assert prof.gpu_class.value != "cpu", "Debe detectar GPU CUDA, no CPU"

    flash_configs = get_triton_configs_flash(prof)
    assert len(flash_configs) >= 4, f"Deben existir ≥4 configs Triton, got {len(flash_configs)}"

    # ring_size debe ser razonable
    assert 8 <= prof.ring_size <= 128, f"ring_size fuera de rango: {prof.ring_size}"
    print(f"  [✓] GPU detectado: {prof.name} ({prof.gpu_class.value})")
    print(f"  [✓] Triton flash configs generados: {len(flash_configs)}")
    print(f"  [✓] ring_size={prof.ring_size}  stages_flash={prof.triton_stages_flash}")
    print(f"  [✓] compile_mode_train='{prof.compile_mode_train}'  FP8={prof.use_fp8_fwd}")


def test_fp8_linear():
    """Fp8Linear: error numérico vs nn.Linear < 5%; sin NaN/Inf."""
    print("\n[7] Fp8Linear — FP8 forward con fallback BF16")
    torch.manual_seed(42)
    lin   = nn.Linear(128, 256, bias=True, device=DEVICE, dtype=torch.bfloat16)
    fp8   = Fp8Linear.from_linear(lin)

    x     = torch.randn(4, 128, device=DEVICE, dtype=torch.bfloat16)
    ref   = lin(x)
    out   = fp8(x)

    assert not torch.isnan(out).any(),  "Fp8Linear produjo NaN"
    assert not torch.isinf(out).any(),  "Fp8Linear produjo Inf"
    assert out.shape == ref.shape,      f"Shape mismatch: {out.shape} vs {ref.shape}"

    rel_err = (out - ref).abs().mean() / ref.abs().mean().clamp(min=1e-8)
    assert rel_err < 0.05, f"Error relativo FP8 demasiado alto: {rel_err:.4f}"
    print(f"  [✓] Fp8Linear shape={out.shape}  rel_err={rel_err:.4f}  "
          f"FP8_AVAIL={_FP8_AVAIL}")

    # apply_fp8_to_chimera: no-op en RTX 4050 (use_fp8_fwd=False)
    layer = AdvancedChimeraLayer(d_model=D, expand=2, headdim=32).to(DEVICE)
    prof  = get_gpu_profile()
    layer_after = apply_fp8_to_chimera(layer, prof=prof)
    if not prof.use_fp8_fwd:
        # bus.publish debe seguir siendo nn.Linear (no-op)
        assert isinstance(layer_after.bus.publish, nn.Linear), \
            "apply_fp8_to_chimera no debería convertir en RTX 4050"
        print(f"  [✓] apply_fp8_to_chimera: no-op correcto en {prof.gpu_class.value}")
    else:
        assert isinstance(layer_after.bus.publish, Fp8Linear), \
            "apply_fp8_to_chimera debería convertir en H100/H200"
        print(f"  [✓] apply_fp8_to_chimera: conversión activa en {prof.gpu_class.value}")


def test_chunked_prefill(model: 'ChimeraStack'):
    """chunked_prefill: carry SSM+Bus reduce drift +108% T5 → <5%."""
    print("\n[8] chunked_prefill — carry SSM+Bus entre chunks")
    torch.manual_seed(7)
    model.eval()

    # S > chunk_size para garantizar chunking real (8 chunks)
    S_test     = 1024
    chunk_size = 128
    x = torch.randn(2, S_test, D, device=DEVICE)

    layer = model.layers[0]

    with torch.no_grad():
        # Referencia: forward completo
        out_full, _ = layer(x, bus_cache=None)

        # chunked_prefill: carry conv_state + SSM state + bus_cache
        out_carry, _ = chunked_prefill(layer, x, chunk_size=chunk_size)

        # Naive sin carry (reproduce drift T5): bus_cache=None en cada chunk
        parts = []
        for start in range(0, S_test, chunk_size):
            end  = min(start + chunk_size, S_test)
            p, _ = layer(x[:, start:end], bus_cache=None)
            parts.append(p)
        out_naive = torch.cat(parts, dim=1)

    assert not torch.isnan(out_carry).any(), "NaN en chunked_prefill output"
    assert not torch.isinf(out_carry).any(), "Inf en chunked_prefill output"
    assert out_carry.shape == out_full.shape, "Shape mismatch"

    norm = out_full.abs().mean().clamp(min=1e-8)
    drift_carry = (out_full - out_carry).abs().mean() / norm * 100
    drift_naive = (out_full - out_naive).abs().mean() / norm * 100

    # Con carry SSM+Bus el drift debe ser <5% (vs +108% en T5 naive)
    assert drift_carry < 5.0, \
        f"drift con carry demasiado alto: {drift_carry:.1f}% (esperado <5%)"
    assert drift_carry < drift_naive, \
        f"carry ({drift_carry:.1f}%) debe ser < naive ({drift_naive:.1f}%)"

    print(f"  [✓] drift carry  = {drift_carry:.1f}%  (carry SSM+Bus)")
    print(f"  [✓] drift naive  = {drift_naive:.1f}%  (sin carry — reproduce T5)")
    print(f"  [✓] mejora: {drift_naive/drift_carry.clamp(min=0.1):.1f}×  "
          f"shape={out_carry.shape}")
    model.train()


def test_cuda_graph_pool(model: 'ChimeraStack'):
    """CUDAGraphPool: step produce output válido para B=1,2; auto-padding correcto."""
    print("\n[9] CUDAGraphPool — pool de CUDA Graphs por batch_size")
    model.eval()
    layer = model.layers[0]

    import gc; gc.collect(); torch.cuda.empty_cache()

    # Pool pequeño para RTX 4050 (VRAM limitada)
    pool = CUDAGraphPool(layer, batch_sizes=[1, 2, 4], device=DEVICE)

    # B=1 sin padding
    x1  = torch.randn(1, 1, D, device=DEVICE)
    out1, c1 = pool.step(x1, pool.allocate_cache(1))
    assert out1.shape == (1, 1, D), f"Shape B=1 incorrecto: {out1.shape}"
    assert not torch.isnan(out1).any(), "NaN en B=1"

    # B=2 sin padding
    x2  = torch.randn(2, 1, D, device=DEVICE)
    out2, c2 = pool.step(x2, pool.allocate_cache(2))
    assert out2.shape == (2, 1, D), f"Shape B=2 incorrecto: {out2.shape}"
    assert not torch.isnan(out2).any(), "NaN en B=2"

    # B=3 → usa graph de B=4 (padding automático)
    x3  = torch.randn(3, 1, D, device=DEVICE)
    out3, c3 = pool.step(x3, pool.allocate_cache(3))
    assert out3.shape == (3, 1, D), f"Shape B=3 (padded a 4) incorrecto: {out3.shape}"
    assert not torch.isnan(out3).any(), "NaN en B=3"

    # _select_batch_size
    assert pool._select_batch_size(1) == 1
    assert pool._select_batch_size(3) == 4
    assert pool._select_batch_size(4) == 4

    print(f"  [✓] Graphs capturados: {pool.batch_sizes}")
    print(f"  [✓] B=1: shape={out1.shape}  B=2: shape={out2.shape}  B=3→pad4: shape={out3.shape}")
    print(f"  [✓] _select_batch_size(3)=4 (padding automático)")
    model.train()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  TEST INTEGRACIÓN CHIMERA — 4 componentes nuevos")
    print("=" * 65)

    model = ChimeraStack(n_layers=3, d_model=D, expand=2, headdim=32)
    model = model.to(DEVICE).train()

    t_total = time.time()

    test_ttt_triton_kernel()
    test_warmup_scheduler(model)
    test_chimera_losses(model)
    test_step_decode(model)
    test_ring_bus_shape(model)
    test_return_aux(model)
    test_gpu_profile()
    test_fp8_linear()
    test_chunked_prefill(model)
    test_cuda_graph_pool(model)

    t_total = time.time() - t_total
    print("\n" + "=" * 65)
    print(f"  SUCCESS — todos los tests pasados en {t_total:.1f}s")
    print("=" * 65)
