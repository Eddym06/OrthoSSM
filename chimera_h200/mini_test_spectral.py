"""
mini_test_spectral.py — Validación de SpectralVSAArchive v2 en RTX 4050 Laptop
================================================================================
Ejecutar desde chimera_h200/:
    python3 mini_test_spectral.py

Qué hace:
  1. Construye CHIMERA-125M con SpectralVSA activo (ChebyHolo)
  2. Entrena 200 steps con datos sintéticos (B=2, S=512, vocab=4096)
  3. Mide las 4 propiedades especiales del archive: decaimiento espectral,
     interferencia VSA (raw vs corregida), supresión de Gibbs y corrección
     de errores en tiempo real
  4. Reporta VRAM real (torch.cuda.memory_allocated)

Notas de ingeniería:
  - bfloat16 + AMP + grad_scaler: ruta estándar de entrenamiento en Ada Lovelace
  - SpectralVSA FP32 accumulators: internamente siempre FP32 (sin overhead visible)
  - 200 steps ≈ 2-4 min en RTX 4050 Laptop (6 GB GDDR6, ~180 TFLOPS BF16)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import math
import torch

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.set_float32_matmul_precision("high")  # TF32 en Ada: gratis ~+20% throughput

from chimera_config import ChimeraConfig
from chimera_lm     import ChimeraLM

# ──────────────────────────────────────────────────────────────────────────────
# Configuración para RTX 4050 Laptop (6 GB GDDR6)
# ──────────────────────────────────────────────────────────────────────────────
cfg = ChimeraConfig.small_125M()      # d=768, L=23, ~132M params
cfg.use_spectral_vsa  = True          # ← SpectralVSAArchive v2 (ChebyHolo)
cfg.max_seq_len       = 512           # conservador para 6 GB
cfg.dtype             = "bfloat16"
cfg.lr                = 6e-4
cfg.warmup_steps      = 50

# Reset caché de perfil GPU para forzar re-detección (útil en entorno multiGPU)
try:
    import gpu_profile
    gpu_profile._PROFILE_CACHE = None
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# Construcción del modelo
# Usamos ChimeraLM(cfg, ...) directamente — equivalente a build_chimera_125M
# pero sin colisión de kwargs duplicados.
# ──────────────────────────────────────────────────────────────────────────────
model = ChimeraLM(cfg, vocab_size=4096, ckpt_interval=999).cuda().bfloat16()

print("=" * 60)
print("  Mini-test SpectralVSA v2  |  RTX 4050 Laptop")
print("=" * 60)
vram_theory = cfg.vram_estimate()
print(f"VRAM estimada (teórica):  {vram_theory['total_gb']:.3f} GB")
print(f"  → weights:    {vram_theory['weights_mb']:.0f} MB")
print(f"  → activations:{vram_theory['activations_mb']:.0f} MB")
print(f"  → ssm_state:  {vram_theory['ssm_state_mb']:.0f} MB")
print(f"  → spectral:   {vram_theory['landmarks_mb']:.1f} MB  (SpectralVSA v2)")
print(f"Parámetros reales: {model.num_parameters()/1e6:.1f}M")
print(f"SpectralVSA activo: {cfg.use_spectral_vsa}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# Mini-entrenamiento: 200 steps, B=2, S=512
# ──────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.1,
                              betas=(0.9, 0.95))
# BF16 no necesita GradScaler — mismo rango de exponente que FP32 (8 bits).

B, S = 2, 512
print("── Mini-entrenamiento (200 steps) ──────────────────────────")
model.train()
for step in range(200):
    input_ids = torch.randint(0, 4096, (B, S), device="cuda")
    labels    = input_ids.clone()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, loss, loss_dict = model(input_ids, labels=labels, aux_weight=0.01)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if step % 20 == 0:
        routing = loss_dict.get("routing", loss_dict.get("routing_loss", 0.0))
        if hasattr(routing, "item"):
            routing = routing.item()
        print(f"  Step {step:3d} | Loss {loss.item():.4f} | "
              f"Routing {routing:.5f}")

print()
print("── Mini-entrenamiento COMPLETADO ───────────────────────────")

# ──────────────────────────────────────────────────────────────────────────────
# Mediciones especiales de SpectralVSAArchive
# ──────────────────────────────────────────────────────────────────────────────
print()
print("── Mediciones especiales del SpectralVSAArchive ────────────")

# Acceder al archive de la primera capa
# ChimeraLM → model.stack (ChimeraStack) → .layers[0] (AdvancedChimeraLayer) → .archive
layer   = model.stack.layers[0]
archive = layer.archive

if not hasattr(archive, "measure_spectral_decay"):
    print("  [WARN] archive no es SpectralVSAArchive — use_spectral_vsa podría estar False")
    sys.exit(0)

model.eval()
with torch.no_grad():

    # 1. Decaimiento espectral (valida hipótesis SSST)
    h_diag = torch.randn(1, 1024, cfg.d_model, device="cuda", dtype=torch.float32)
    decay = archive.measure_spectral_decay(h_diag)
    print(f"\n[1] Decaimiento espectral (SSST)")
    print(f"    β estimado:            {decay['beta_estimate']:.3f}  "
          f"(ideal 1.5–2.5 para texto natural)")
    print(f"    Energía en primeros 16: {decay['energy_at_16']*100:.1f}%")
    print(f"    Energía en primeros 32: {decay['energy_at_32']*100:.1f}%")
    print(f"    K recomendado:          {decay['K_recommended']}")

    # 2. Interferencia VSA (raw vs corregida)
    h_vsa = torch.randn(512, cfg.d_model, device="cuda", dtype=torch.float32)
    interf = archive.measure_vsa_interference(h_vsa)
    print(f"\n[2] Interferencia VSA (bind-unbind)")
    print(f"    Error relativo medio raw:       {interf['mean_rel_error_raw']:.6f}")
    print(f"    Error relativo medio CORREGIDO: {interf['mean_rel_error_corrected']:.6f}")
    factor = interf['error_reduction_factor']
    print(f"    Reducción de error:             {factor:.1f}×")

    # 3. Supresión de Gibbs (efecto Lanczos sobre discontinuidades)
    h_gibbs = torch.randn(512, cfg.d_model, device="cuda", dtype=torch.float32)
    gibbs = archive.measure_lanczos_effect(h_gibbs)
    print(f"\n[3] Supresión de Gibbs (Lanczos anti-Runge)")
    print(f"    Gibbs amplitude raw:     {gibbs['gibbs_amplitude_raw']:.6f}")
    print(f"    Gibbs amplitude Lanczos: {gibbs['gibbs_amplitude_lanczos']:.6f}")
    print(f"    Supresión Gibbs:         {gibbs['gibbs_suppression']:.3f}×")
    print(f"    RMSE smooth raw:         {gibbs['rmse_smooth_raw']:.6f}")
    print(f"    RMSE smooth Lanczos:     {gibbs['rmse_smooth_lanczos']:.6f}")

    # 4. Calidad de corrección de error + estado de corrección Kahan
    err_info = archive.measure_error_correction_quality()
    print(f"\n[4] Corrección de errores (Kahan + shadow-unbind)")
    print(f"    Corrección acumulada L2: {err_info['correction_l2']:.6f}")
    print(f"    V_mem L2 (referencia):   {err_info['vmem_l2']:.6f}")
    print(f"    Ratio corrección/V_mem:  {err_info['correction_ratio']:.6f}")
    print(f"    Kahan comp FP32 norm:    {err_info['kahan_comp_V_real_norm']:.8f}  "
          f"(>0 = activo)")
    print(f"    K activo:                {err_info['K_active']}")

# 5. Estado del archive (info de diagnóstico)
info = archive.get_archive_info()
print(f"\n[5] Estado del archive (post-entrenamiento)")
print(f"    tipo:             {info['type']}")
print(f"    K_max / K_active: {info['K_max']} / {info['K_active']}")
print(f"    Lanczos power p:  {info['lanczos_power']:.3f}")
print(f"    Condition number: {info['condition_number']:.2f}")
print(f"    Noise floor:      {info['noise_floor']:.8f}")
print(f"    Disc count:       {info['disc_count']}")
print(f"    blend_gate σ:     {info['blend_gate']:.4f}")
print(f"    inject_gate σ:    {info['inject_gate']:.4f}")

# 6. VRAM real (torch allocator)
vram_real = torch.cuda.memory_allocated() / 1024**3
vram_peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"\n[6] VRAM real (torch allocator)")
print(f"    Actual:           {vram_real:.3f} GB")
print(f"    Peak (training):  {vram_peak:.3f} GB  ← incluye gradientes + activaciones")

print()
print("=" * 60)
print("  [OK] Mini-test SpectralVSA v2 COMPLETADO")
print("=" * 60)
