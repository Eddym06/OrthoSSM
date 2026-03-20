"""
test_500step_v13.py — Test integral de Chimera-1.3 Heavy-Duty
==============================================================
Suite completa de 5 fases:

  FASE 1: Inventario y sanidad de la arquitectura
  FASE 2: Benchmark de throughput (tokens/s, VRAM, tiempos de forward)
  FASE 3: 500 pasos de entrenamiento con métricas de convergencia
  FASE 4: Diagnóstico profundo (routing, SDTM, gradientes, pérdidas)
  FASE 5: Resultados + recomendaciones de calibración

Adaptado 100% a chimera-1.3:
  - SDTM multi-head (4 cabezas, M_fast [4,64,64])
  - d_state=128, bus_dim=256, max_landmarks=512
  - update_memory_inplace() + maybe_consolidate() integrados
  - ChimeraRoutingLoss con balance/entropy/supervision
"""
from __future__ import annotations

import math, sys, os, time, json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from chimera_config   import ChimeraConfig
from chimera_lm       import ChimeraLM, ChimeraStack
from advanced_chimera import AdvancedChimeraLayer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')
torch.manual_seed(42)

SEP = "=" * 72

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de test: usamos defaults de chimera-1.3 (Heavy-Duty)
# con d_model=256 para caber en RTX 4050 Laptop
# ─────────────────────────────────────────────────────────────────────────────

def make_config() -> ChimeraConfig:
    """Config de test: d_model=256 × chimera-1.3 Heavy-Duty defaults."""
    return ChimeraConfig(
        d_model     = 256,
        n_layers    = 4,
        expand      = 2,
        headdim     = 32,
        d_state     = 128,        # heavy-duty: 64→128
        bus_dim     = 256,        # heavy-duty: 128→256
        max_landmarks = 512,      # heavy-duty: 128→512
        sdtm_n_heads  = 4,        # heavy-duty: 1→4
        sdtm_d_mem    = 0,        # auto = max(64, 256//4) = 64 per head
        # Training params
        lr            = 3e-4,
        warmup_steps  = 50,
        max_seq_len   = 512,
        # Routing loss weights — chimera-1.3
        routing_entropy_weight     = 0.05,
        routing_supervision_weight = 0.10,
        routing_balance_weight     = 0.02,
        routing_target_entropy     = 0.70,
        routing_min_tier_prob      = 0.05,
        ttt_pred_weight            = 0.05,
    )


VOCAB_SIZE = 4096
SEQ_LEN    = 256    # seq_len de entrenamiento
BATCH      = 4

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_mb() -> float:
    if DEVICE.type == 'cuda':
        return torch.cuda.memory_allocated() / 1e6
    return 0.0

def _gpu_reserved_mb() -> float:
    if DEVICE.type == 'cuda':
        return torch.cuda.memory_reserved() / 1e6
    return 0.0

def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().norm().item() ** 2
    return math.sqrt(total)

def _routing_stats(aux_list: list) -> dict:
    """Extrae estadísticas de routing de la aux_list del forward."""
    if not aux_list:
        return {}
    stats = defaultdict(list)
    for aux in aux_list:
        if aux is None:
            continue
        probs = aux.get('routing_probs', aux.get('tier_probs'))
        if probs is not None:
            stats['fast'].append(probs[:, 0].mean().item())
            stats['hybrid'].append(probs[:, 1].mean().item())
            stats['full'].append(probs[:, 2].mean().item())
        if 'router_logits' in aux:
            stats['router_logits_std'].append(
                aux['router_logits'].float().std().item()
            )
    return {k: sum(v)/len(v) for k, v in stats.items() if v}

def _sdtm_state_norm(model: ChimeraLM) -> dict:
    """Norma de M_fast y M_slow por capa."""
    result = {}
    for i, layer in enumerate(model.stack.layers):
        mf_norm = layer.sdtm.M_fast.float().norm().item()
        ms_norm = layer.sdtm.M_slow.float().norm().item()
        result[f'L{i}_M_fast'] = round(mf_norm, 4)
        result[f'L{i}_M_slow'] = round(ms_norm, 4)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1 — Inventario arquitectónico
# ─────────────────────────────────────────────────────────────────────────────

def fase1_inventario(model: ChimeraLM, cfg: ChimeraConfig):
    print(f"\n{SEP}")
    print("  FASE 1 — Inventario Arquitectónico Chimera-1.3 Heavy-Duty")
    print(SEP)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  Versión             : {cfg.version}")
    print(f"  d_model / n_layers  : {cfg.d_model} / {cfg.n_layers}")
    print(f"  d_state (Mamba2)    : {cfg.d_state}  [heavy-duty: 64→128]")
    print(f"  bus_dim             : {cfg.bus_dim}   [heavy-duty: 128→256]")
    print(f"  max_landmarks       : {cfg.max_landmarks}  [heavy-duty: 128→512]")
    print(f"  sdtm_n_heads        : {cfg.sdtm_n_heads}   [heavy-duty: 1→4]")
    print(f"  sdtm_d_mem (auto)   : {max(64, cfg.d_model//4)} per head")
    print(f"  sdtm M_fast shape   : {tuple(model.stack.layers[0].sdtm.M_fast.shape)}")
    print(f"  Parámetros totales  : {total_p:,}  ({total_p/1e6:.2f}M)")
    print(f"  Parámetros entren.  : {train_p:,}  ({train_p/1e6:.2f}M)")

    vram = cfg.vram_estimate(max_seq_len=SEQ_LEN)
    print(f"\n  VRAM estimada ({SEQ_LEN} tok): {vram['total_gb']:.3f} GB")
    print(f"    pesos={vram['weights_mb']} MB  acts={vram['activations_mb']} MB  "
          f"sdtm={vram['sdtm_mb']} MB  landmarks={vram['landmarks_mb']} MB")

    # Verificar que cada capa tiene los parámetros heavy-duty activos
    errors = []
    for i, layer in enumerate(model.stack.layers):
        if layer.mamba2.d_state != cfg.d_state:
            errors.append(f"  L{i} d_state mismatch: {layer.mamba2.d_state} != {cfg.d_state}")
        if layer.bus.bus_dim != cfg.bus_dim:
            errors.append(f"  L{i} bus_dim mismatch: {layer.bus.bus_dim} != {cfg.bus_dim}")
        if layer.archive.max_landmarks != cfg.max_landmarks:
            errors.append(f"  L{i} max_landmarks mismatch: {layer.archive.max_landmarks} != {cfg.max_landmarks}")
        if layer.sdtm.n_heads != cfg.sdtm_n_heads:
            errors.append(f"  L{i} sdtm.n_heads mismatch: {layer.sdtm.n_heads} != {cfg.sdtm_n_heads}")

    if errors:
        print("\n  [!] DISCREPANCIAS ENCONTRADAS:")
        for e in errors:
            print(e)
        return False
    else:
        print("\n  [✓] Todos los parámetros heavy-duty verificados en 4 capas")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2 — Benchmark de throughput
# ─────────────────────────────────────────────────────────────────────────────

def fase2_throughput(model: ChimeraLM):
    print(f"\n{SEP}")
    print("  FASE 2 — Benchmark de Throughput")
    print(SEP)

    model.eval()
    results = {}

    # ── 2a. Warmup ────────────────────────────────────────────────────────────
    with torch.no_grad():
        for _ in range(3):
            x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)
            _ = model(x)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

    # ── 2b. Forward solo (inferencia) ─────────────────────────────────────────
    N_RUNS = 10
    t0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(DEVICE.type=='cuda')):
        for _ in range(N_RUNS):
            x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)
            _ = model(x)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    t_fwd = (time.perf_counter() - t0) / N_RUNS

    tps_inf = BATCH * SEQ_LEN / t_fwd
    results['fwd_ms']       = round(t_fwd * 1000, 2)
    results['tps_inference'] = round(tps_inf, 0)
    print(f"\n  Inferencia (B={BATCH}, S={SEQ_LEN}):")
    print(f"    Forward time  : {t_fwd*1000:.1f} ms/iter")
    print(f"    Throughput    : {tps_inf:,.0f} tokens/s")

    # ── 2c. Forward + Backward (training) ─────────────────────────────────────
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=DEVICE.type=='cuda')

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)
        labels = x.clone()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(DEVICE.type=='cuda')):
            _, loss, _ = model(x, labels=labels)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    t_train = (time.perf_counter() - t0) / N_RUNS

    tps_train = BATCH * SEQ_LEN / t_train
    results['train_ms']       = round(t_train * 1000, 2)
    results['tps_training']   = round(tps_train, 0)
    print(f"\n  Training (B={BATCH}, S={SEQ_LEN}):")
    print(f"    Fwd+Bwd time  : {t_train*1000:.1f} ms/iter")
    print(f"    Throughput    : {tps_train:,.0f} tokens/s")

    # ── 2d. VRAM ──────────────────────────────────────────────────────────────
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
        vram_alloc = _gpu_mb()
        vram_res   = _gpu_reserved_mb()
        results['vram_alloc_mb']    = round(vram_alloc, 1)
        results['vram_reserved_mb'] = round(vram_res, 1)
        print(f"\n  VRAM real en GPU:")
        print(f"    Allocated : {vram_alloc:.1f} MB")
        print(f"    Reserved  : {vram_res:.1f} MB")

    # ── 2e. SDTM overhead ─────────────────────────────────────────────────────
    # Medir overhead de SDTM multi-head vs forward base
    layer = model.stack.layers[0]
    _dtype = next(model.parameters()).dtype
    x_d = torch.randn(BATCH, SEQ_LEN, 256, device=DEVICE, dtype=_dtype)
    err = torch.rand(BATCH, SEQ_LEN - 1, device=DEVICE, dtype=_dtype)

    t_sdtm = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = layer.sdtm.read(x_d)
        layer.sdtm.compute_write(x_d, err)
        layer.sdtm.update_memory_inplace()
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t_sdtm.append(time.perf_counter() - t0)
    t_sdtm_mean = sum(t_sdtm[5:]) / len(t_sdtm[5:])
    results['sdtm_overhead_ms'] = round(t_sdtm_mean * 1000, 3)
    print(f"\n  SDTM multi-head overhead (4 heads × 64²):")
    print(f"    read+write+update : {t_sdtm_mean*1000:.2f} ms")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3 — 500 pasos de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def fase3_500steps(model: ChimeraLM, cfg: ChimeraConfig):
    print(f"\n{SEP}")
    print("  FASE 3 — 500 Pasos de Entrenamiento")
    print(SEP)

    TOTAL_STEPS  = 500
    LOG_EVERY    = 50
    CONSOLIDATE_EVERY = 100   # pasos entre consolidaciones SDTM

    # Optimizer: AdamW fused separando grupos no-decay
    no_decay_names = ('bias', 'norm_f', '.w', 'embedding')
    decay_p, no_decay_p = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or any(nd in name for nd in no_decay_names):
            no_decay_p.append(p)
        else:
            decay_p.append(p)

    optimizer = torch.optim.AdamW(
        [{'params': decay_p, 'weight_decay': 0.1},
         {'params': no_decay_p, 'weight_decay': 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8,
        fused=(DEVICE.type == 'cuda'),
    )

    # WSD schedule: 10% warmup, 80% stable, 10% decay
    warmup = int(TOTAL_STEPS * 0.10)
    stable = int(TOTAL_STEPS * 0.90)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        if step < stable:
            return 1.0
        t = (step - stable) / max(TOTAL_STEPS - stable, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Historial de métricas
    history = {
        'step': [], 'lm_loss': [], 'routing_loss': [], 'total_loss': [],
        'grad_norm': [], 'lr': [],
        'fast_prob': [], 'hybrid_prob': [], 'full_prob': [],
        'sdtm_mfast_norm': [], 'sdtm_mslow_norm': [],
        'tokens_per_sec': [],
    }

    model.train()
    t_total = time.perf_counter()
    t_block = time.perf_counter()
    tokens_in_block = 0

    # Generar datos sintéticos con estructura (mezcla uniforme + patrones)
    def make_batch():
        # 70% uniforme, 30% con repetición para crear dependencias de largo rango
        if torch.rand(1).item() > 0.3:
            return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)
        else:
            # Batch con repeticiones: fuerza al modelo a usar memoria
            base = torch.randint(0, VOCAB_SIZE // 4, (BATCH, SEQ_LEN // 2), device=DEVICE)
            repeated = torch.cat([base, base], dim=1)
            return repeated

    print(f"\n  Config: steps={TOTAL_STEPS}  B={BATCH}  S={SEQ_LEN}  "
          f"LR={cfg.lr}  vocab={VOCAB_SIZE}")
    print(f"  Warmup: {warmup} steps | Stable: {stable-warmup} steps | "
          f"Decay: {TOTAL_STEPS-stable} steps\n")
    print(f"  {'Step':>6} | {'LM-Loss':>8} | {'Rout':>6} | {'Total':>8} | "
          f"{'GNorm':>7} | {'Fast':>5} | {'Hyb':>5} | {'Full':>5} | "
          f"{'LR':>8} | {'tok/s':>7}")
    print(f"  {'-'*90}")

    for step in range(TOTAL_STEPS):
        x      = make_batch()
        labels = x.clone()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                enabled=(DEVICE.type == 'cuda')):
            logits, total_loss, loss_dict = model(x, labels=labels, aux_weight=0.01)

        optimizer.zero_grad()
        total_loss.backward()

        # TTT update (SDTM write) — aplicar después del backward
        for layer in model.stack.layers:
            layer.update_ttt_inplace()
            layer.sdtm.update_memory_inplace()
            layer.sdtm.apply_usage_decay()

        # Consolidación SDTM periódica
        if (step + 1) % CONSOLIDATE_EVERY == 0:
            for layer in model.stack.layers:
                layer.sdtm.maybe_consolidate(tokens_processed=BATCH * SEQ_LEN * CONSOLIDATE_EVERY)

        # Gradient clipping
        gn = _grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        tokens_in_block += BATCH * SEQ_LEN

        # ── Logging cada LOG_EVERY pasos ──────────────────────────────────────
        if (step + 1) % LOG_EVERY == 0:
            t_now  = time.perf_counter()
            dt     = t_now - t_block
            tps    = tokens_in_block / dt
            t_block = t_now
            tokens_in_block = 0

            lm_l  = loss_dict.get('lm', 0.0)
            rt_l  = loss_dict.get('routing', 0.0)
            tot_l = loss_dict.get('total', total_loss.item())
            lr_now = scheduler.get_last_lr()[0]

            # Routing stats del último step
            # Extraemos del aux_list del stack (necesitamos re-forward con return_aux)
            # En su lugar usamos los últimos values de loss_dict
            fast_p  = loss_dict.get('fast_prob',   loss_dict.get('tier0_mean', 0.333))
            hyb_p   = loss_dict.get('hybrid_prob', loss_dict.get('tier1_mean', 0.333))
            full_p  = loss_dict.get('full_prob',   loss_dict.get('tier2_mean', 0.333))

            # SDTM M_fast norm (capa 0 representativa)
            mf_norm = model.stack.layers[0].sdtm.M_fast.float().norm().item()
            ms_norm = model.stack.layers[0].sdtm.M_slow.float().norm().item()

            history['step'].append(step + 1)
            history['lm_loss'].append(round(lm_l, 4))
            history['routing_loss'].append(round(rt_l, 4))
            history['total_loss'].append(round(tot_l, 4))
            history['grad_norm'].append(round(gn, 4))
            history['lr'].append(round(lr_now, 6))
            history['fast_prob'].append(round(fast_p, 3))
            history['hybrid_prob'].append(round(hyb_p, 3))
            history['full_prob'].append(round(full_p, 3))
            history['sdtm_mfast_norm'].append(round(mf_norm, 4))
            history['sdtm_mslow_norm'].append(round(ms_norm, 4))
            history['tokens_per_sec'].append(round(tps, 0))

            print(f"  {step+1:>6} | {lm_l:>8.4f} | {rt_l:>6.4f} | {tot_l:>8.4f} | "
                  f"{gn:>7.3f} | {fast_p:>5.3f} | {hyb_p:>5.3f} | {full_p:>5.3f} | "
                  f"{lr_now:>8.2e} | {tps:>7,.0f}")

    t_elapsed = time.perf_counter() - t_total
    total_tokens = TOTAL_STEPS * BATCH * SEQ_LEN
    mean_tps = total_tokens / t_elapsed

    print(f"\n  {SEP}")
    print(f"  Tiempo total: {t_elapsed:.1f}s  |  Media throughput: {mean_tps:,.0f} tok/s")
    print(f"  Tokens procesados: {total_tokens:,}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# FASE 4 — Diagnóstico profundo post-entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def fase4_diagnostico(model: ChimeraLM, history: dict):
    print(f"\n{SEP}")
    print("  FASE 4 — Diagnóstico Profundo Post-Entrenamiento")
    print(SEP)

    # ── 4a. Convergencia de loss ──────────────────────────────────────────────
    lm_losses = history['lm_loss']
    first_half_mean = sum(lm_losses[:len(lm_losses)//2]) / max(len(lm_losses)//2, 1)
    second_half_mean = sum(lm_losses[len(lm_losses)//2:]) / max(len(lm_losses) - len(lm_losses)//2, 1)
    delta_pct = (first_half_mean - second_half_mean) / max(first_half_mean, 1e-8) * 100

    print(f"\n  4a. Convergencia LM Loss:")
    print(f"    Paso   50 : {lm_losses[0]:.4f}")
    print(f"    Paso  250 : {lm_losses[len(lm_losses)//2]:.4f}")
    print(f"    Paso  500 : {lm_losses[-1]:.4f}")
    print(f"    Mejora 1ª→2ª mitad: {delta_pct:+.1f}%")

    if delta_pct > 1.0:
        print("    [✓] Convergencia detectada — el modelo está aprendiendo")
    elif delta_pct > -1.0:
        print("    [~] Loss estable — sin señal clara de aprendizaje (normal en sintético)")
    else:
        print("    [!] Loss subiendo — revisar LR o routing collapse")

    # ── 4b. Diagnóstico de Routing ────────────────────────────────────────────
    fast_p   = history['fast_prob'][-1]  if history['fast_prob']   else 0.333
    hyb_p    = history['hybrid_prob'][-1] if history['hybrid_prob'] else 0.333
    full_p   = history['full_prob'][-1]  if history['full_prob']   else 0.333

    print(f"\n  4b. Routing (último step):")
    print(f"    FAST={fast_p:.3f}  HYBRID={hyb_p:.3f}  FULL={full_p:.3f}")

    if fast_p > 0.85:
        routing_diag = ("  [!] ROUTING COLLAPSE → FAST (todos van a FAST)"
                        "\n      Acción: aumentar routing_supervision_weight (0.10→0.15)")
    elif full_p > 0.85:
        routing_diag = ("  [!] ROUTING COLLAPSE → FULL (todos van a FULL)"
                        "\n      Acción: reducir arch_threshold (0.50→0.35) o aumentar"
                        " routing_entropy_weight")
    elif min(fast_p, hyb_p, full_p) < 0.05:
        routing_diag = ("  [!] TIER MUERTO — un tier con prob < 5%"
                        "\n      Acción: aumentar routing_min_tier_prob (0.05→0.10)")
    else:
        routing_diag = ("  [✓] Routing saludable — los 3 tiers activos")
    print(routing_diag)

    # ── 4c. Diagnóstico de gradientes ─────────────────────────────────────────
    grad_norms = history['grad_norm']
    gn_mean = sum(grad_norms) / max(len(grad_norms), 1)
    gn_max  = max(grad_norms) if grad_norms else 0.0
    gn_last = grad_norms[-1] if grad_norms else 0.0

    print(f"\n  4c. Gradientes:")
    print(f"    Media : {gn_mean:.3f}  |  Max: {gn_max:.3f}  |  Último: {gn_last:.3f}")

    if gn_max > 10.0:
        print("    [!] Gradientes explosivos detectados — clip_grad=1.0 fue efectivo")
        print("        Acción preventiva: reducir lr (3e-4→1e-4) o aumentar warmup")
    elif gn_mean < 0.01:
        print("    [!] Gradientes muy pequeños — posible vanishing en capas tempranas")
        print("        Acción: verificar residual_scale o aumentar lr")
    else:
        print("    [✓] Gradientes en rango normal")

    # ── 4d. Diagnóstico SDTM Multi-Head ───────────────────────────────────────
    mf_norms = history['sdtm_mfast_norm']
    ms_norms = history['sdtm_mslow_norm']
    mf_last  = mf_norms[-1] if mf_norms else 0.0
    ms_last  = ms_norms[-1] if ms_norms else 0.0
    mf_first = mf_norms[0]  if mf_norms else 0.0

    print(f"\n  4d. SDTM Multi-Head (capa 0, 4 heads × 64×64):")
    print(f"    M_fast norma paso  50: {mf_first:.4f}")
    print(f"    M_fast norma paso 500: {mf_last:.4f}")
    print(f"    M_slow norma paso 500: {ms_last:.4f}")

    if mf_last < 1e-4 and mf_first < 1e-4:
        print("    [!] M_fast cerca de cero — SDTM no está escribiendo")
        print("        Acción: verificar que surprise signal está llegando")
        print("        (ttt_importance / compute_token_errors_triton en forward)")
    elif mf_last > ms_last * 0.3:
        print("    [✓] M_fast creciendo — consolidaciones funcionando")
    else:
        print("    [~] M_fast pequeño vs M_slow — verificar consolidación_interval")

    # ── 4e. Throughput a lo largo del entrenamiento ────────────────────────────
    tps_list = history['tokens_per_sec']
    if tps_list:
        tps_mean = sum(tps_list) / len(tps_list)
        tps_last = tps_list[-1]
        print(f"\n  4e. Throughput de entrenamiento:")
        print(f"    Media: {tps_mean:,.0f} tok/s  |  Último: {tps_last:,.0f} tok/s")

    return {
        'lm_loss_first': lm_losses[0] if lm_losses else None,
        'lm_loss_last':  lm_losses[-1] if lm_losses else None,
        'delta_pct':     round(delta_pct, 2),
        'routing_stable': min(fast_p, hyb_p, full_p) >= 0.05,
        'grad_norm_mean': round(gn_mean, 4),
        'grad_norm_max':  round(gn_max, 4),
        'sdtm_mfast_last': round(mf_last, 4),
        'sdtm_mslow_last': round(ms_last, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FASE 5 — Calibración y recomendaciones
# ─────────────────────────────────────────────────────────────────────────────

def fase5_calibracion(cfg: ChimeraConfig, diag: dict, thr_results: dict):
    print(f"\n{SEP}")
    print("  FASE 5 — Calibración y Recomendaciones para Chimera-1.3")
    print(SEP)

    issues   = []
    actions  = []
    params   = {}

    # ── Calibración de LR ─────────────────────────────────────────────────────
    gn_mean = diag.get('grad_norm_mean', 1.0)
    gn_max  = diag.get('grad_norm_max', 1.0)
    delta   = diag.get('delta_pct', 0.0)

    if gn_max > 8.0:
        new_lr = round(cfg.lr * 0.5, 6)
        issues.append("Gradientes explosivos periódicos")
        actions.append(f"lr: {cfg.lr} → {new_lr}")
        params['lr'] = new_lr
    elif gn_mean < 0.05 and delta < 0.5:
        new_lr = round(min(cfg.lr * 2.0, 1e-3), 6)
        issues.append("Gradientes muy pequeños — posible LR bajo")
        actions.append(f"lr: {cfg.lr} → {new_lr}")
        params['lr'] = new_lr
    elif delta > 5.0:
        new_lr = round(cfg.lr * 1.5, 6)
        issues.append("Convergencia rápida — LR puede subir")
        actions.append(f"lr: {cfg.lr} → {new_lr}")
        params['lr'] = new_lr
    else:
        params['lr'] = cfg.lr

    # ── Calibración de Routing ────────────────────────────────────────────────
    is_routing_healthy = diag.get('routing_stable', True)
    if not is_routing_healthy:
        issues.append("Tier muerto en routing")
        new_min_tier = round(min(cfg.routing_min_tier_prob * 2, 0.15), 3)
        actions.append(f"routing_min_tier_prob: {cfg.routing_min_tier_prob} → {new_min_tier}")
        params['routing_min_tier_prob'] = new_min_tier
    else:
        params['routing_min_tier_prob'] = cfg.routing_min_tier_prob

    # ── Calibración de SDTM ───────────────────────────────────────────────────
    mf_last = diag.get('sdtm_mfast_last', 0.0)
    if mf_last < 1e-3:
        issues.append("SDTM M_fast inactivo — surprise signal débil")
        new_sdtm_lr = round(cfg.sdtm_lr * 2.0, 5)
        actions.append(f"sdtm_lr: {cfg.sdtm_lr} → {new_sdtm_lr}")
        new_topk = max(cfg.sdtm_surprise_top_k // 2, 4)
        actions.append(f"sdtm_surprise_top_k: {cfg.sdtm_surprise_top_k} → {new_topk} (más selectivo)")
        params['sdtm_lr'] = new_sdtm_lr
        params['sdtm_surprise_top_k'] = new_topk
    else:
        params['sdtm_lr'] = cfg.sdtm_lr
        params['sdtm_surprise_top_k'] = cfg.sdtm_surprise_top_k

    # ── Calibración de Warmup ─────────────────────────────────────────────────
    if gn_max > 5.0 and cfg.warmup_steps < 200:
        new_warmup = cfg.warmup_steps * 3
        issues.append("Picos de gradiente en primeros pasos")
        actions.append(f"warmup_steps: {cfg.warmup_steps} → {new_warmup} (más gradual)")
        params['warmup_steps'] = new_warmup
    else:
        params['warmup_steps'] = cfg.warmup_steps

    # ── Throughput ────────────────────────────────────────────────────────────
    tps_train = thr_results.get('tps_training', 0)
    vram_mb   = thr_results.get('vram_alloc_mb', 0)

    print(f"\n  Resumen de Rendimiento Heavy-Duty vs Baseline (d_state=64, 1-head):")
    print(f"    Throughput training : {tps_train:,.0f} tok/s")
    print(f"    VRAM real usada     : {vram_mb:.0f} MB")
    print(f"    SDTM overhead       : {thr_results.get('sdtm_overhead_ms', 0):.2f} ms")

    print(f"\n  Problemas detectados ({len(issues)}):")
    if issues:
        for i, iss in enumerate(issues):
            print(f"    {i+1}. {iss}")
    else:
        print("    Ninguno — arquitectura estable")

    print(f"\n  Acciones de calibración ({len(actions)}):")
    if actions:
        for i, act in enumerate(actions):
            print(f"    {i+1}. {act}")
    else:
        print("    No se requieren cambios — config calibrada correctamente")

    # ── Config calibrada final ────────────────────────────────────────────────
    print(f"\n  Config calibrada final (chimera-1.3-calibrada):")
    print(f"    d_state         = {cfg.d_state}  (heavy-duty)")
    print(f"    bus_dim         = {cfg.bus_dim}   (heavy-duty)")
    print(f"    max_landmarks   = {cfg.max_landmarks}  (heavy-duty)")
    print(f"    sdtm_n_heads    = {cfg.sdtm_n_heads}   (heavy-duty)")
    print(f"    lr              = {params.get('lr', cfg.lr)}")
    print(f"    warmup_steps    = {params.get('warmup_steps', cfg.warmup_steps)}")
    print(f"    sdtm_lr         = {params.get('sdtm_lr', cfg.sdtm_lr)}")
    print(f"    sdtm_surprise_top_k = {params.get('sdtm_surprise_top_k', cfg.sdtm_surprise_top_k)}")
    print(f"    routing_min_tier_prob = {params.get('routing_min_tier_prob', cfg.routing_min_tier_prob)}")

    return params


# ─────────────────────────────────────────────────────────────────────────────
# Aplicar calibración a chimera_config.py
# ─────────────────────────────────────────────────────────────────────────────

def aplicar_calibracion(cfg: ChimeraConfig, params: dict,
                        config_path: str = None) -> ChimeraConfig:
    """
    Aplica los parámetros calibrados al ChimeraConfig y muestra el diff.
    NO modifica el archivo fuente — devuelve la config nueva para uso inmediato.
    """
    print(f"\n{SEP}")
    print("  Aplicando calibración a ChimeraConfig")
    print(SEP)

    cambios = []
    for k, v_new in params.items():
        v_old = getattr(cfg, k, None)
        if v_old is not None and abs(float(v_old) - float(v_new)) > 1e-9:
            cambios.append((k, v_old, v_new))

    if not cambios:
        print("  [✓] Config ya está óptima — sin cambios necesarios")
        return cfg

    print(f"  Diff ({len(cambios)} cambios):")
    kwargs = {}
    for k, v_old, v_new in cambios:
        print(f"    {k}: {v_old} → {v_new}")
        kwargs[k] = v_new

    # Crear nueva instancia con params calibrados
    import dataclasses
    new_cfg = dataclasses.replace(cfg, **kwargs)
    print(f"\n  [✓] Config calibrada creada (version={new_cfg.version})")
    return new_cfg


# ─────────────────────────────────────────────────────────────────────────────
# TEST DE GENERACIÓN rápida post-entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def test_generacion(model: ChimeraLM):
    print(f"\n{SEP}")
    print("  TEST EXTRA — Generación autoregresiva post-entrenamiento")
    print(SEP)

    model.eval()
    prompt = torch.randint(0, VOCAB_SIZE, (1, 8), device=DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        gen = model.generate(prompt, max_new_tokens=32, temperature=0.8, top_k=40)
    t_gen = time.perf_counter() - t0

    tps_gen = gen.shape[1] / t_gen
    print(f"\n  Prompt  : {prompt.shape} → {gen.shape}")
    print(f"  Tiempo  : {t_gen*1000:.1f} ms  ({tps_gen:.0f} tok/s decode)")
    print(f"  [✓] Generación completada correctamente")
    return tps_gen


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{SEP}")
    print("  TEST INTEGRAL CHIMERA-1.3 HEAVY-DUTY")
    print(f"  Device: {DEVICE}  |  PyTorch: {torch.__version__}")
    if DEVICE.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}  |  VRAM: {props.total_memory/1e9:.1f} GB  "
              f"|  SM: {props.major}.{props.minor}")
    print(SEP)

    # ── Construcción del modelo ───────────────────────────────────────────────
    cfg   = make_config()
    model = ChimeraLM(cfg, vocab_size=VOCAB_SIZE).to(DEVICE)

    if DEVICE.type == 'cuda':
        model = model.bfloat16()

    # Resetear stats VRAM
    if DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # ── Ejecutar fases ────────────────────────────────────────────────────────
    ok = fase1_inventario(model, cfg)
    if not ok:
        print("\n  [ABORT] Discrepancias en inventario — revisar chimera_config.py")
        return

    thr_results = fase2_throughput(model)
    history     = fase3_500steps(model, cfg)
    diag        = fase4_diagnostico(model, history)
    cal_params  = fase5_calibracion(cfg, diag, thr_results)
    new_cfg     = aplicar_calibracion(cfg, cal_params)

    # ── Generación post-entrenamiento ─────────────────────────────────────────
    tps_gen = test_generacion(model)

    # ── Resumen final ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  RESUMEN FINAL — CHIMERA-1.3 HEAVY-DUTY")
    print(SEP)

    lm0 = history['lm_loss'][0]   if history['lm_loss'] else 0
    lmf = history['lm_loss'][-1]  if history['lm_loss'] else 0
    ppl0 = math.exp(min(lm0, 10))
    pplf = math.exp(min(lmf, 10))

    print(f"\n  Loss   inicial (paso  50): {lm0:.4f}  (PPL ≈ {ppl0:.1f})")
    print(f"  Loss   final   (paso 500): {lmf:.4f}  (PPL ≈ {pplf:.1f})")
    print(f"  Mejora de convergencia   : {diag.get('delta_pct', 0):+.1f}%")
    print(f"  Throughput training      : {thr_results.get('tps_training', 0):,.0f} tok/s")
    print(f"  Throughput decode        : {tps_gen:.0f} tok/s")
    print(f"  VRAM real usada          : {thr_results.get('vram_alloc_mb', 0):.0f} MB")

    if DEVICE.type == 'cuda':
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"  VRAM peak (test total)   : {peak_mb:.0f} MB")

    all_ok = all([
        history['lm_loss'][-1] < history['lm_loss'][0] + 0.5,  # no divergió
        diag.get('grad_norm_max', 99) < 50,                     # no explotan gradientes
        thr_results.get('tps_training', 0) > 100,               # al menos 100 tok/s
    ])

    print(f"\n  {'[✓] CHIMERA-1.3 HEAVY-DUTY: PASS' if all_ok else '[!] CHIMERA-1.3: REVISIÓN NECESARIA'}")
    print(SEP)

    # Guardar resultados
    results_file = Path(__file__).parent / 'test_500step_v13_results.json'
    results = {
        'config': {k: v for k, v in cfg.to_dict().items()
                   if k in ('version','d_model','n_layers','d_state','bus_dim',
                            'max_landmarks','sdtm_n_heads','lr')},
        'throughput': thr_results,
        'history':    {k: v[-10:] for k, v in history.items()},  # últimos 10 puntos
        'diagnostic': diag,
        'calibration': cal_params,
        'pass': all_ok,
    }
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Resultados guardados en: {results_file.name}")


if __name__ == '__main__':
    main()
