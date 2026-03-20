"""
eval_arch_300.py — Evaluación Completa de Arquitectura CHIMERA + SpectralVSA v2
================================================================================
300 steps de mini-entrenamiento con instrumentación exhaustiva:
  • Estabilidad numérica: NaN/Inf, overflow BF16, condition number, gradient norms
  • Velocidad y throughput: tokens/s por componente, forward/backward split
  • Uso de memoria: VRAM peak, activaciones, fragmentation ratio
  • Routing: distribución de tiers, entropía, balance, colapso
  • SpectralVSA: K_active evolution, Lanczos power, corrección de errores,
                 noise floor, interferencia VSA, β espectral
  • Identificación de cuellos de botella: timing por componente (mamba2, slr,
                                          bus, archive, router, ttt)
  • Reporte estructurado + recomendaciones de optimización

Ejecutar:
    source /home/OrthoSSM/venv/bin/activate
    cd /home/OrthoSSM/chimera_h200
    python3 eval_arch_300.py
"""
import sys, os, time, json, math, collections
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.set_float32_matmul_precision("high")

# ─── Imports del stack ────────────────────────────────────────────────────────
from chimera_config import ChimeraConfig
from chimera_lm     import ChimeraLM
from advanced_chimera import ChimeraAnnealer

# ─── Configuración del experimento ───────────────────────────────────────────
B, S, VOCAB = 2, 512, 4096
N_STEPS     = 500
AUX_WEIGHT  = 0.01
LOG_INTERVAL     = 10
ARCHIVE_INTERVAL = 20
DETAIL_INTERVAL  = 50   # timing componente detallado

cfg = ChimeraConfig.small_125M()
cfg.use_spectral_vsa           = True
cfg.max_seq_len                = S
cfg.dtype                      = "bfloat16"
cfg.lr                         = 6e-4
cfg.warmup_steps               = 50
cfg.spectral_K                 = 32
cfg.spectral_K_min             = 4
cfg.spectral_energy_threshold  = 0.95
cfg.spectral_lanczos_power_max = 3.0   # reducido de 4.0 para evitar over-damping
cfg.spectral_disc_gamma        = 3.0
cfg.spectral_error_refresh     = 0.5
cfg.grad_clip                  = 0.5   # más agresivo: contiene explosiones Mamba2
cfg.arch_threshold             = 0.45  # baja de 0.50 → SLR recibe más gradiente

print("=" * 68)
print("  CHIMERA + SpectralVSA v2 — Evaluación Arquitectural 300 Steps")
print("=" * 68)
print(f"  GPU : {torch.cuda.get_device_name(0)}")
print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB total")
vram_est = cfg.vram_estimate()
print(f"  Est.: {vram_est['total_gb']:.3f} GB  "
      f"(weights={vram_est['weights_mb']:.0f} MB, "
      f"spectral={vram_est['landmarks_mb']:.1f} MB)")
print(f"  B={B}, S={S}, vocab={VOCAB}, steps={N_STEPS}")
print()

# ─── Construcción del modelo ──────────────────────────────────────────────────
model = ChimeraLM(cfg, vocab_size=VOCAB, ckpt_interval=999).cuda().bfloat16()
print(f"  Parámetros: {model.num_parameters()/1e6:.1f}M")
print(f"  SpectralVSA: {'ON (ChebyHolo)' if cfg.use_spectral_vsa else 'OFF (NativeLandmark)'}")
n_layers = cfg.n_layers
print()

# ─── Optimizer con warmup coseno ─────────────────────────────────────────────
# Liberar fragmentación VRAM acumulada antes de construir buffers del optimizer
torch.cuda.empty_cache()

optimizer = torch.optim.AdamW(
    model.parameters(), lr=cfg.lr, weight_decay=0.1, betas=(0.9, 0.95))
# BF16 no necesita GradScaler — el rango de exponente es idéntico a FP32 (8 bits).
# GradScaler es exclusivo de FP16 (5-bit exponent → underflow frecuente).
# Usar scaler con BF16 lanza RuntimeError en PyTorch ≥ 2.1 en _unscale_grads_.

def warmup_cosine_lr(step, warmup=50, total=500, lr_max=6e-4, lr_min=6e-5):
    if step < warmup:
        return lr_max * (step + 1) / warmup
    t = (step - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))

# ─── Annealer de umbrales del router (cold-start) ────────────────────────────
# Fuerza FAST durante los primeros pasos → Mamba2 aprende estructura base.
# Cosine decay de umbrales altos (0.90/0.85) a targets (0.30/0.50).
annealer = ChimeraAnnealer(
    model, warmup_steps=N_STEPS,
    slr_start=0.90, slr_target=0.30,
    arch_start=0.85, arch_target=0.50,
)

# ─── Acceso centralizado al archive ──────────────────────────────────────────
def get_archive():
    return model.stack.layers[0].archive

# ─── Instrumentación: hooks de timing por componente ─────────────────────────
# Mide el tiempo de subcomponentes de la primera capa (representativa)
_comp_timers = collections.defaultdict(list)  # name → list of ms
_hook_events = {}   # name → (start_event, end_event)

def _make_hooks(name):
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    _hook_events[name] = (ev_s, ev_e)

    def pre_hook(module, inp):
        ev_s.record()

    def post_hook(module, inp, out):
        ev_e.record()

    return pre_hook, post_hook

layer0 = model.stack.layers[0]
_hook_handles = []
for _name, _sub in [
    ("mamba2",  layer0.mamba2),
    ("slr",     layer0.slr),
    ("bus",     layer0.bus),
    ("router",  layer0.router),
    ("archive", layer0.archive),
]:
    pre_h, post_h = _make_hooks(_name)
    _hook_handles.append(_sub.register_forward_pre_hook(pre_h))
    _hook_handles.append(_sub.register_forward_hook(post_h))

# ─── Acumuladores de métricas ──────────────────────────────────────────────────
metrics = {
    "losses": [],
    "grad_norms": [],
    "routing_fast": [], "routing_hybrid": [], "routing_full": [],
    "routing_entropy": [],
    "k_active": [],
    "lanczos_power": [],
    "disc_count": [],
    "noise_floor": [],
    "condition_number": [],
    "vram_mb": [],
    "step_ms": [],
    "tokens_per_sec": [],
    "nan_steps": [],
    "overflow_steps": [],
    "injection_gate": [],
    "blend_gate_val": [],
    "current_residuals": [],  # correction_ratio
}
comp_timing_samples = collections.defaultdict(list)  # step → {comp: ms}
grad_per_group = []   # list of dicts

# ─── Detección de overflow BF16 ────────────────────────────────────────────────
def check_bf16_overflow(loss_val):
    """BF16 max ≈ 65504. overflow → loss = 65504 or inf"""
    return math.isinf(loss_val) or math.isnan(loss_val) or loss_val > 50

# ─── Función de gradient norm por grupo ───────────────────────────────────────
def get_grad_norms():
    norms = {}
    # Parámetros del archive (SpectralVSA)
    arch_norm = 0.0
    arch_params = list(get_archive().parameters())
    for p in arch_params:
        if p.grad is not None:
            arch_norm += p.grad.float().norm().item() ** 2
    norms["archive_spectral"] = math.sqrt(arch_norm)

    # Mamba2 SSM
    m2_norm = 0.0
    for p in layer0.mamba2.parameters():
        if p.grad is not None:
            m2_norm += p.grad.float().norm().item() ** 2
    norms["mamba2"] = math.sqrt(m2_norm)

    # Router
    rtr_norm = 0.0
    for p in layer0.router.parameters():
        if p.grad is not None:
            rtr_norm += p.grad.float().norm().item() ** 2
    norms["router"] = math.sqrt(rtr_norm)

    # SLR
    slr_norm = 0.0
    for p in layer0.slr.parameters():
        if p.grad is not None:
            slr_norm += p.grad.float().norm().item() ** 2
    norms["slr"] = math.sqrt(slr_norm)

    # TTT params (ttt_U, ttt_V, ttt_full_scale)
    ttt_norm = 0.0
    for name, p in layer0.named_parameters():
        if "ttt" in name and p.grad is not None:
            ttt_norm += p.grad.float().norm().item() ** 2
    norms["ttt"] = math.sqrt(ttt_norm)

    # Total (global)
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.float().norm().item() ** 2
    norms["total"] = math.sqrt(total)

    return norms

# ─── Datos sintéticos (pre-generados para reproducibilidad) ──────────────────
torch.manual_seed(42)
data = [torch.randint(0, VOCAB, (B, S), device="cuda") for _ in range(N_STEPS + 5)]

# ─── Warmup: 5 pasos silenciosos ─────────────────────────────────────────────
print("  [Warmup] 5 pasos silenciosos...")
model.train()
torch.cuda.reset_peak_memory_stats()
for i in range(5):
    ids = data[i]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, loss, _ = model(ids, labels=ids, aux_weight=AUX_WEIGHT)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
print("  [Warmup] Completado\n")

# ─── Activar timing de componentes ───────────────────────────────────────────
_record_comp = False
_comp_step_data = {}

# ─── Loop principal 300 steps ─────────────────────────────────────────────────
print("─" * 78)
print(f"  {'Step':>4} | {'Loss':>7} | {'GNorm':>7} | {'Tok/s':>7} | "
      f"{'pF':>4} {'pH':>4} {'pA':>4} | Kact | {'ΔErr':>6}")
print("─" * 78)

ev_start = torch.cuda.Event(enable_timing=True)
ev_end   = torch.cuda.Event(enable_timing=True)

for step in range(N_STEPS):
    # Scheduler LR
    lr = warmup_cosine_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # Anneal router thresholds (cosine cold-start)
    annealer.step(step)

    ids = data[step + 5]
    labels = ids.clone()

    # Activar grabación de componentes en steps detallados
    _record_comp = (step % DETAIL_INTERVAL == 0)

    ev_start.record()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, loss, loss_dict = model(ids, labels=labels, aux_weight=AUX_WEIGHT)

    # Detect NaN/Inf antes del backward
    loss_val = loss.item()
    is_bad = check_bf16_overflow(loss_val) or torch.isnan(loss).item()
    if is_bad:
        metrics["nan_steps"].append(step)
        print(f"  [WARN] NaN/Overflow at step {step}: loss={loss_val}")
        optimizer.zero_grad(set_to_none=True)
        continue

    loss.backward()

    # Gradient norm (antes del clip)
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.float().norm().item() ** 2
    grad_norm = math.sqrt(total_norm_sq)

    # Per-group grad norms — DEBE ser antes de zero_grad
    if step % 50 == 0 and step > 0:
        gnorms = get_grad_norms()
        gnorms["step"] = step
        grad_per_group.append(gnorms)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    ev_end.record()
    torch.cuda.synchronize()

    step_ms = ev_start.elapsed_time(ev_end)
    tokens_sec = (B * S * 1000) / step_ms

    # Timing de componentes (sincronizar los eventos de hooks)
    if _record_comp:
        comp_data = {}
        for cname, (es, ee) in _hook_events.items():
            try:
                ms = es.elapsed_time(ee)
            except Exception:
                ms = 0.0
            comp_data[cname] = round(ms, 3)
            comp_timing_samples[cname].append(ms)
        _comp_step_data[step] = comp_data

    # Archive info
    arch_info = get_archive().get_archive_info()

    # Routing (from loss_dict — ChimeraRoutingLoss info dict)
    routing_val = loss_dict.get("routing", loss_dict.get("routing_loss", 0.0))
    if hasattr(routing_val, "item"):
        routing_val = routing_val.item()

    # Tier distribution telemetry: extract from loss_dict (populated by ChimeraRoutingLoss)
    p_fast    = loss_dict.get("routing/p_fast",   0.333)
    p_hybrid  = loss_dict.get("routing/p_hybrid", 0.333)
    p_full    = loss_dict.get("routing/p_full",   0.333)
    h_bits    = loss_dict.get("routing/H_bits",   0.0)

    # fast_prob_ema de la capa como proxy adicional
    fast_ema = layer0.fast_prob_ema.item()

    # VRAM
    vram_mb = torch.cuda.memory_allocated() / 1024**2

    # Registrar
    metrics["losses"].append(loss_val)
    metrics["grad_norms"].append(min(grad_norm, 999.0))
    metrics["step_ms"].append(step_ms)
    metrics["tokens_per_sec"].append(tokens_sec)
    metrics["vram_mb"].append(vram_mb)
    metrics["k_active"].append(arch_info["K_active"])
    metrics["lanczos_power"].append(arch_info["lanczos_power"])
    metrics["disc_count"].append(arch_info["disc_count"])
    metrics["noise_floor"].append(arch_info["noise_floor"])
    metrics["condition_number"].append(arch_info["condition_number"])
    metrics["injection_gate"].append(arch_info["inject_gate"])
    metrics["blend_gate_val"].append(arch_info["blend_gate"])
    metrics["current_residuals"].append(arch_info.get("error_correction_norm", 0.0))
    # Tier distribution telemetry
    metrics["routing_fast"].append(p_fast)
    metrics["routing_hybrid"].append(p_hybrid)
    metrics["routing_full"].append(p_full)
    metrics["routing_entropy"].append(h_bits)

    if step % LOG_INTERVAL == 0:
        comp_str = ""
        if _record_comp and step in _comp_step_data:
            cd = _comp_step_data[step]
            comp_str = (f" [m2={cd.get('mamba2',0):.1f} "
                        f"slr={cd.get('slr',0):.1f} "
                        f"arc={cd.get('archive',0):.1f} "
                        f"bus={cd.get('bus',0):.1f}ms]")

        print(f"  {step:>4} | {loss_val:>7.4f} | {grad_norm:>7.3f} | "
              f"{tokens_sec/1000:>6.1f}k | "
              f"F={p_fast:.2f} H={p_hybrid:.2f} A={p_full:.2f} | "
              f"{arch_info['K_active']:>4} | "
              f"{arch_info.get('error_correction_norm',0):>6.4f}"
              f"{comp_str}")

print("─" * 78)

# ─── Phase 2: Diagnósticos post-entrenamiento ─────────────────────────────────
print("\n" + "=" * 68)
print("  DIAGNÓSTICOS POST-ENTRENAMIENTO")
print("=" * 68)

# ── Tier Distribution Summary ─────────────────────────────────────────────────
if metrics["routing_fast"]:
    n = len(metrics["routing_fast"])
    last_50 = max(0, n - 50)
    rf_avg = sum(metrics["routing_fast"][last_50:])  / max(1, n - last_50)
    rh_avg = sum(metrics["routing_hybrid"][last_50:]) / max(1, n - last_50)
    ra_avg = sum(metrics["routing_full"][last_50:])   / max(1, n - last_50)
    he_avg = sum(metrics["routing_entropy"][last_50:]) / max(1, n - last_50)

    print(f"\n  Tier Distribution (last 50 steps):")
    print(f"    FAST (Mamba2):   {rf_avg*100:>5.1f}%  {'✓ OK' if 0.40 <= rf_avg <= 0.85 else '⚠ ALERT'}")
    print(f"    HYBRID (SLR):    {rh_avg*100:>5.1f}%  {'✓ OK' if 0.10 <= rh_avg <= 0.40 else '⚠ ALERT'}")
    print(f"    FULL (Archive):  {ra_avg*100:>5.1f}%  {'✓ OK' if 0.03 <= ra_avg <= 0.25 else '⚠ ALERT'}")
    print(f"    Entropy (bits):  {he_avg:>5.2f}")
    if rf_avg < 0.40:
        print(f"    🔴 ALERT: p_fast < 40% — SLR dominando. Subir balance_weight o slr_threshold.")
    if rf_avg > 0.85:
        print(f"    🔴 ALERT: p_fast > 85% — SLR/Archive infrautilizados. Bajar umbrales.")

    # Annealer state
    ann_state = annealer.get_state()
    if ann_state:
        print(f"    Annealer final:  slr_thr={ann_state['annealer/slr_threshold']:.3f}  "
              f"arch_thr={ann_state['annealer/arch_threshold']:.3f}")

model.eval()
archive = get_archive()

with torch.no_grad():
    h_test = torch.randn(1, 1024, cfg.d_model, device="cuda", dtype=torch.float32)

    # 1. Decaimiento espectral
    decay = archive.measure_spectral_decay(h_test)
    print(f"\n[ESPECTRAL] β={decay['beta_estimate']:.3f}  "
          f"E@16={decay['energy_at_16']*100:.1f}%  "
          f"E@32={decay['energy_at_32']*100:.1f}%  "
          f"K_rec={decay['K_recommended']}")
    beta_ok = 0.5 < decay['beta_estimate'] < 4.0
    print(f"  SSST válido: {'✓' if beta_ok else '✗ (señal random, esperado < 0.5 en datos sin estructura)'}")

    # 2. Interferencia VSA
    h_vsa = torch.randn(512, cfg.d_model, device="cuda", dtype=torch.float32)
    interf = archive.measure_vsa_interference(h_vsa)
    print(f"\n[VSA INTERF] raw={interf['mean_rel_error_raw']:.6f}  "
          f"corrected={interf['mean_rel_error_corrected']:.6f}  "
          f"reducción={interf['error_reduction_factor']:.1f}×")

    # 3. Lanczos / Gibbs
    gibbs = archive.measure_lanczos_effect(h_vsa)
    print(f"\n[GIBBS]  raw={gibbs['gibbs_amplitude_raw']:.6f}  "
          f"lanczos={gibbs['gibbs_amplitude_lanczos']:.6f}  "
          f"supresión={gibbs['gibbs_suppression']:.2f}×")
    print(f"  RMSE smooth: raw={gibbs['rmse_smooth_raw']:.6f}  "
          f"lanczos={gibbs['rmse_smooth_lanczos']:.6f}  "
          f"overhead={((gibbs['rmse_smooth_lanczos']/max(gibbs['rmse_smooth_raw'],1e-10))-1)*100:.1f}%")

    # 4. Error correction
    err_q = archive.measure_error_correction_quality()
    print(f"\n[ERROR CORR]  L2={err_q['correction_l2']:.6f}  "
          f"ratio={err_q['correction_ratio']:.6f}  "
          f"Kahan={err_q['kahan_comp_V_real_norm']:.8f}")
    kahan_active = err_q['kahan_comp_V_real_norm'] > 1e-10
    print(f"  Kahan compensación: {'ACTIVA ✓' if kahan_active else 'INACTIVA — sin drift acumulado'}")

# ─── Análisis estadístico de métricas ─────────────────────────────────────────
def stats(lst, label):
    if not lst:
        return f"{label}: N/A"
    t = torch.tensor(lst, dtype=torch.float32)
    return (f"{label}: "
            f"μ={t.mean().item():.4f}  "
            f"σ={t.std().item():.4f}  "
            f"min={t.min().item():.4f}  "
            f"max={t.max().item():.4f}")

first_50_loss  = metrics["losses"][:50]
last_50_loss   = metrics["losses"][-50:]
first_mean = sum(first_50_loss) / len(first_50_loss) if first_50_loss else 0
last_mean  = sum(last_50_loss)  / len(last_50_loss)  if last_50_loss  else 0
loss_trend = last_mean - first_mean

print("\n" + "=" * 68)
print("  ANÁLISIS ESTADÍSTICO")
print("=" * 68)
print(f"\n  Loss convergencia:  first_50={first_mean:.4f}  →  last_50={last_mean:.4f}  "
      f"Δ={loss_trend:+.4f}  {'↓ convergiendo' if loss_trend < -0.05 else '→ estable' if abs(loss_trend) < 0.1 else '↑ DIVERGIENDO'}")
print(f"  {stats(metrics['losses'], 'Loss')}")
print(f"  {stats(metrics['grad_norms'], 'Grad norm')}")
avg_step_ms = sum(metrics["step_ms"]) / len(metrics["step_ms"])
avg_toks    = sum(metrics["tokens_per_sec"]) / len(metrics["tokens_per_sec"])
print(f"\n  Throughput: {avg_toks/1000:.2f}k tok/s  ({avg_step_ms:.1f} ms/step)")
print(f"  VRAM peak:  {torch.cuda.max_memory_allocated()/1024**2:.1f} MB  "
      f"(reserved={torch.cuda.memory_reserved()/1024**2:.1f} MB)")
frag = 1.0 - torch.cuda.memory_allocated() / max(torch.cuda.memory_reserved(), 1)
print(f"  Fragmentación VRAM: {frag*100:.1f}%  "
      f"({'OK' if frag < 0.3 else 'ALTA — considerar memory_efficient ops'})")

# K_active stats
k_vals = metrics["k_active"]
k_max_frac = sum(1 for k in k_vals if k == cfg.spectral_K) / len(k_vals)
print(f"\n  K_active:  μ={sum(k_vals)/len(k_vals):.1f}  "
      f"max_frac={k_max_frac*100:.1f}% en K_max={cfg.spectral_K}")
print(f"  Lanczos p: {stats(metrics['lanczos_power'], 'p')}")
print(f"  Disc count final: {metrics['disc_count'][-1]}")
print(f"  Noise floor final: {metrics['noise_floor'][-1]:.8f}")
print(f"  Condition #: {stats(metrics['condition_number'], 'κ')}")
print(f"\n  Overflow events: {len(metrics['overflow_steps'])}  "
      f"{'✓ clean' if not metrics['overflow_steps'] else '⚠ steps: ' + str(metrics['overflow_steps'][:10])}")
print(f"  NaN events:      {len(metrics['nan_steps'])}  "
      f"{'✓ clean' if not metrics['nan_steps'] else '⚠ steps: ' + str(metrics['nan_steps'][:10])}")

# ─── Bottleneck analysis ───────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  ANÁLISIS DE CUELLOS DE BOTELLA")
print("=" * 68)
total_components = 0
comp_means = {}
for cname, vals in comp_timing_samples.items():
    if vals:
        m = sum(vals) / len(vals)
        comp_means[cname] = m
        total_components += m

if comp_means:
    sorted_comps = sorted(comp_means.items(), key=lambda x: -x[1])
    print(f"\n  Timing por componente (promedio, capa 0):")
    print(f"  {'Componente':12s} | {'ms avg':>8} | {'% total':>8} | Bottleneck")
    print(f"  {'─'*12} | {'─'*8} | {'─'*8} | {'─'*20}")
    for cname, ms in sorted_comps:
        frac = ms / max(total_components, 1e-9) * 100
        flag = "⚑ PRINCIPAL" if frac > 40 else ("▲ alto" if frac > 25 else "")
        print(f"  {cname:12s} | {ms:>8.3f} | {frac:>8.1f}% | {flag}")
    print(f"\n  Suma componentes: {total_components:.3f} ms")
    print(f"  Step total:       {avg_step_ms:.2f} ms  (incluye embedding, norm, lm_head)")
    overhead_ms = avg_step_ms - total_components
    print(f"  Overhead otros:   {overhead_ms:.2f} ms  "
          f"(embedding+norm+lm_head+loss+optimizer ~{overhead_ms:.0f} ms)")
else:
    print("  [INFO] Sin datos de timing de componentes — revisar hooks CUDA")

# Gradient por grupo final
if grad_per_group:
    last_gnorms = grad_per_group[-1]
    print(f"\n  Gradient norms por componente (step {last_gnorms['step']}):")
    for k, v in last_gnorms.items():
        if k != "step":
            bar = "█" * max(1, int(min(v/0.1, 20)))
            flag = ""
            if v < 1e-6:
                flag = "  ← ⚠ MUERTO (vanishing gradient)"
            elif v > 10.0:
                flag = "  ← ⚠ EXPLODIENDO"
            print(f"    {k:20s}: {v:8.5f} {bar}{flag}")

# ─── Diagnóstico SpectralVSA profundo ───────────────────────────────────────────
print("\n" + "=" * 68)
print("  DIAGNÓSTICO SpectralVSA v2")
print("=" * 68)
arch_final = archive.get_archive_info()
print(f"\n  Tipo:            {arch_final['type']}")
print(f"  K_max/K_active:  {arch_final['K_max']} / {arch_final['K_active']}")
print(f"  has_memory:      {arch_final['has_memory']}")
print(f"  steps archivados:{arch_final['step_count']}")
print(f"  buf_fill:        {arch_final['buf_fill']}/{cfg.spectral_window}")
print(f"  I_total (Δ prom):{arch_final['I_total']}")
print(f"  delta_ema:       {arch_final['delta_ema']}")
print(f"  delta_low_freq:  {arch_final['delta_low_freq']}")
print(f"  delta_high_freq: {arch_final['delta_high_freq']}")
print(f"  lanczos_power:   {arch_final['lanczos_power']}")
print(f"  condition_num:   {arch_final['condition_number']}")
print(f"  noise_floor:     {arch_final['noise_floor']}")
print(f"  disc_count total:{arch_final['disc_count']}")
print(f"  err_corr norm:   {arch_final['error_correction_norm']}")
print(f"  inject_gate σ:   {arch_final['inject_gate']}")
print(f"  blend_gate σ:    {arch_final['blend_gate']}")
print(f"  memoria/capa:    {arch_final['memory_bytes']} bytes = {arch_final['memory_bytes']/1024:.1f} KB (solo V_mem)")

# K_active dynamics analysis
k_arr = metrics["k_active"]
k_changes = sum(1 for i in range(1, len(k_arr)) if k_arr[i] != k_arr[i-1])
print(f"\n  Dynamic K: {k_changes} cambios en {N_STEPS} steps")
print(f"  K_active range: [{min(k_arr)}, {max(k_arr)}]  μ={sum(k_arr)/len(k_arr):.1f}")
if k_changes == 0:
    if max(k_arr) == cfg.spectral_K:
        print(f"  ⚠ K siempre en K_max={cfg.spectral_K} → datos random sin estructura espectral clara")
        print(f"    (esperado en datos sintéticos; en texto real K caería a 6-12)")
    else:
        print(f"  ⚠ K siempre en {k_arr[0]} sin cambios → señal muy uniforme")

# ─── Recomendaciones ──────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  RECOMENDACIONES DE OPTIMIZACIÓN")
print("=" * 68)

recommendations = []

# 1. Loss convergencia
if loss_trend > 0.1:
    recommendations.append(("CRITICO", "Divergencia detectada",
        "Reducir lr de 6e-4 a 3e-4, aumentar warmup de 50 a 100 steps"))
elif loss_trend > -0.05:
    recommendations.append(("INFO", "Loss estancado",
        "Datos sintéticos random → sin convergencia real esperada. OK para diagnóstico"))

# 2. Overflow BF16
if len(metrics["overflow_steps"]) > 5:
    recommendations.append(("WARN", f"Múltiples overflow BF16 ({len(metrics['overflow_steps'])} eventos)",
        "Reducir lr, añadir loss scaling más conservador, o audit NaN propagation"))

# 3. K_active siempre en max
if k_max_frac > 0.9:
    recommendations.append(("PERF", "K_active pegado en K_max",
        f"En datos reales K debería ≤ 12. Función dinámica correcta. "
        f"Si en producción K persiste alto → bajar energy_threshold de 0.95 a 0.90"))

# 4. Bottleneck de componentes
if comp_means:
    top_comp, top_ms = sorted_comps[0]
    frac = top_ms / max(total_components, 1e-9)
    if frac > 0.5:
        recommendations.append(("PERF", f"Bottleneck: {top_comp} ({frac*100:.0f}% del tiempo)",
            {"mamba2": "Normal — SSD scan es O(S·D·d_state). Considerar FlashSSM con chunk_size=256",
             "slr": "SLR Triton caro. Verificar que el kernel usa num_warps=8 en Ada Lovelace",
             "archive": "SpectralVSA lento → K_active.item() causa CPU-GPU sync en hot-path",
             "bus": "AsyncLightBus costoso → ring_size demasiado grande o d_bus muy alto",
             "router": "Router costoso (inesperado) → revisar d_hidden del MLP"
             }.get(top_comp, "Revisar implementación")))

# 5. Gradient vanishing
if grad_per_group:
    last_gnorms = grad_per_group[-1]
    for k, v in last_gnorms.items():
        if k != "step" and v < 1e-5 and k != "ttt":
            recommendations.append(("WARN", f"Gradient muerto en {k} (norm={v:.2e})",
                "Revisar init de pesos, agregar gradient passthrough, o reducir arch_threshold"))

# 6. Condition number alto
cond_vals = metrics["condition_number"]
if cond_vals and max(cond_vals) > 1000:
    recommendations.append(("WARN", f"Condition number alto (max={max(cond_vals):.0f})",
        "Coeficientes mal condicionados → activar rank-aware regularization o reducir K_max"))

# 7. Noise floor
nf_vals = metrics["noise_floor"]
if nf_vals and max(nf_vals) > 0.01:
    recommendations.append(("WARN", f"Noise floor alto ({max(nf_vals):.5f})",
        "Ruido acumulado en buffer → considerar reducir window_size o agregar decaimiento al buffer"))

# 8. Fragmentación VRAM
if frag > 0.3:
    recommendations.append(("PERF", f"Fragmentación VRAM {frag*100:.0f}%",
        "Usar torch.cuda.memory._snapshot() para analizar. Posible fix: torch.cuda.empty_cache() periódico"))

# 9. Kahan activo
if not kahan_active:
    recommendations.append(("INFO", "Kahan compensation inactiva",
        "Normal si son pocos steps. Se acumula con secuencias largas (>1000 tokens históricos)"))

for severity, title, detail in recommendations:
    icons = {"CRITICO": "🔴", "WARN": "🟡", "PERF": "🔵", "INFO": "⚪"}
    icon = icons.get(severity, "⚪")
    print(f"\n  {icon} [{severity}] {title}")
    if isinstance(detail, str):
        print(f"    → {detail}")
    else:
        for k, v in detail.items():
            print(f"    [{k}] {v}")

if not recommendations:
    print("\n  ✓ Sin recomendaciones críticas — arquitectura en estado saludable")

# ─── Guardar reporte JSON ────────────────────────────────────────────────────
report = {
    "summary": {
        "n_steps": N_STEPS,
        "first_50_loss_mean": round(first_mean, 5),
        "last_50_loss_mean": round(last_mean, 5),
        "loss_trend": round(loss_trend, 5),
        "avg_tokens_per_sec": round(avg_toks, 1),
        "avg_step_ms": round(avg_step_ms, 2),
        "vram_peak_mb": round(torch.cuda.max_memory_allocated()/1024**2, 1),
        "nan_events": len(metrics["nan_steps"]),
        "overflow_events": len(metrics["overflow_steps"]),
        "k_active_mean": round(sum(k_vals)/len(k_vals), 2),
        "k_changes": k_changes,
        "disc_count_final": metrics["disc_count"][-1] if metrics["disc_count"] else 0,
        "beta_spectral": decay['beta_estimate'],
        "vsa_error_reduction": interf['error_reduction_factor'],
        "gibbs_suppression": gibbs['gibbs_suppression'],
        "kahan_active": kahan_active,
    },
    "tier_distribution": {
        "p_fast_last50":    round(rf_avg, 4) if metrics["routing_fast"] else None,
        "p_hybrid_last50":  round(rh_avg, 4) if metrics["routing_hybrid"] else None,
        "p_full_last50":    round(ra_avg, 4) if metrics["routing_full"] else None,
        "entropy_last50":   round(he_avg, 4) if metrics["routing_entropy"] else None,
        "annealer_final":   ann_state if ann_state else {},
    },
    "component_timing_ms": {k: round(v, 3) for k, v in comp_means.items()},
    "recommendations": [{"severity": s, "title": t} for s, t, _ in recommendations],
}

out_path = os.path.join(os.path.dirname(__file__), "eval_arch_300_results.json")
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n  Reporte guardado: eval_arch_300_results.json")
print("\n" + "=" * 68)
print("  EVALUACIÓN COMPLETADA")
print("=" * 68)
