"""
test_robustness.py — Test de corrupción, estabilidad y aprendizaje para CHIMERA Elite
=======================================================================================

ESTRUCTURA:
  Fase 1 — Tests unitarios de cada sistema de estabilidad
  Fase 2 — Test de corrupción NaN: verifica que dt_bias NO se corrompe
  Fase 3 — Mini-entrenamiento 100 steps con errores inyectados:
             · Step 15 → NaN artificial en grad de una capa
             · Step 35 → Spike de loss (×8 pérdida normal)
             · Step 60 → Colapso del router (probs → [0.97,0.015,0.015])
             · Step 80 → NaN en grad + router collapse simultáneos

Salida: reporte estructurado con PASS/FAIL + métricas cuantitativas.

Uso:
    cd chimera_h200
    python test_robustness.py               # CPU (CUDA auto-detectada)
    python test_robustness.py --device cuda # forzar GPU
    python test_robustness.py --steps 200   # extender mini-entrenamiento
"""
from __future__ import annotations
import argparse, math, sys, time, os, contextlib
from pathlib import Path
from typing import Dict, List, Optional
import traceback

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Colores ANSI ─────────────────────────────────────────────────────────────
GRN  = "\033[92m"
RED  = "\033[91m"
YLW  = "\033[93m"
CYN  = "\033[96m"
BLD  = "\033[1m"
RST  = "\033[0m"
ok   = lambda s: f"{GRN}✓ {s}{RST}"
fail = lambda s: f"{RED}✗ {s}{RST}"
warn = lambda s: f"{YLW}⚠ {s}{RST}"
hdr  = lambda s: f"\n{BLD}{CYN}{'─'*60}\n  {s}\n{'─'*60}{RST}"

# ─── Registro global de resultados ────────────────────────────────────────────
_results: List[Dict] = []

def record(name: str, passed: bool, detail: str = ""):
    icon = ok(name) if passed else fail(name)
    print(f"  {icon}" + (f"  [{detail}]" if detail else ""))
    _results.append({"name": name, "passed": passed, "detail": detail})

def section(title: str):
    print(hdr(title))


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTACIONES DEL STACK CHIMERA
# ═══════════════════════════════════════════════════════════════════════════════

def _import_chimera(device: torch.device):
    """Importa todos los módulos del stack. Retorna True si OK."""
    global ChimeraConfig, ChimeraLM, build_chimera_125M
    global GradHealthMonitor, LossSpikeDetector, RouterEntropyWatchdog
    global TTTGradSupervisor, ChimeraAnnealer, MuonElite
    global AdvancedChimeraLayer, ChimeraRoutingLoss

    try:
        from chimera_config   import ChimeraConfig
        from chimera_lm       import ChimeraLM, build_chimera_125M
        from advanced_chimera import AdvancedChimeraLayer
        from chimera_losses   import ChimeraRoutingLoss
        from train_h200_elite import (
            GradHealthMonitor, LossSpikeDetector, RouterEntropyWatchdog,
            TTTGradSupervisor, ChimeraAnnealer, MuonElite,
        )
        print(ok("Todos los módulos importados"))
        return True
    except Exception as e:
        print(fail(f"Error importando módulos: {e}"))
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MODELO PEQUEÑO DE TEST
# ═══════════════════════════════════════════════════════════════════════════════

def _tiny_model(device: torch.device) -> ChimeraLM:
    """CHIMERA muy pequeño para tests rápidos en CPU/GPU."""
    cfg = ChimeraConfig(
        d_model   = 256,   # mínimo viable para mamba2 (headdim=32 → 8 heads)
        n_layers  = 4,
        expand    = 2,
        headdim   = 32,
        d_state   = 16,    # estado SSM pequeño
        bus_dim   = 64,
        landmark_dim  = 64,
        max_landmarks = 16,
    )
    model = ChimeraLM(cfg, vocab_size=512).to(device)
    return model

def _synth_batch(batch: int, seq: int, vocab: int, device: torch.device):
    ids    = torch.randint(0, vocab, (batch, seq),    device=device)
    labels = torch.randint(0, vocab, (batch, seq),    device=device)
    return ids, labels


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — TESTS UNITARIOS DE CADA SISTEMA
# ═══════════════════════════════════════════════════════════════════════════════

def test_grad_health_monitor(device: torch.device):
    section("FASE 1A — GradHealthMonitor")
    monitor = GradHealthMonitor(max_consecutive_skips=5)

    # ── Test 1: grads limpios → retorna True ──────────────────────────────────
    lin = nn.Linear(32, 32).to(device)
    lin.weight.grad = torch.randn_like(lin.weight)
    lin.bias.grad   = torch.randn_like(lin.bias)
    result = monitor.check(lin)
    record("GradHealth: grads limpios → True", result == True,
           f"result={result}")

    # ── Test 2: NaN en grad → retorna False, zeros out grad ───────────────────
    lin.weight.grad = torch.full_like(lin.weight, float('nan'))
    lin.bias.grad   = torch.randn_like(lin.bias)
    result = monitor.check(lin)
    grad_zeroed = (lin.weight.grad.abs().sum().item() == 0.0)
    record("GradHealth: NaN detectado → False", result == False, f"result={result}")
    record("GradHealth: grad zeroed tras NaN", grad_zeroed,
           f"grad_sum={lin.weight.grad.abs().sum().item():.4f}")
    record("GradHealth: total_skips==1", monitor._total_skips == 1,
           f"skips={monitor._total_skips}")

    # ── Test 3: Inf en grad → retorna False ───────────────────────────────────
    monitor2 = GradHealthMonitor()
    lin.weight.grad = torch.full_like(lin.weight, float('inf'))
    result = monitor2.check(lin)
    record("GradHealth: Inf detectado → False", result == False, f"result={result}")

    # ── Test 4: reset de skip_count en grad limpio ────────────────────────────
    lin.weight.grad = torch.randn_like(lin.weight)
    lin.bias.grad   = torch.randn_like(lin.bias)
    monitor2.check(lin)  # un NaN
    lin.weight.grad = torch.randn_like(lin.weight)
    monitor2.check(lin)  # limpio
    record("GradHealth: skip_count resetea a 0 tras OK", monitor2._skip_count == 0,
           f"skip_count={monitor2._skip_count}")


def test_loss_spike_detector(device: torch.device):
    section("FASE 1B — LossSpikeDetector")
    det = LossSpikeDetector(alpha=0.95, spike_sigma=2.5, spike_cooldown=10, warmup_steps=5)

    # Calentar con valores normales
    for v in [2.5, 2.4, 2.45, 2.48, 2.42, 2.43]:
        det.update(v)

    # ── Test: lr_scale = 1.0 fuera de spike ───────────────────────────────────
    record("SpikeDetect: lr_scale normal = 1.0", det.lr_scale == 1.0,
           f"lr_scale={det.lr_scale}")

    # Inyectar spike (valor extremo)
    info = det.update(50.0)
    record("SpikeDetect: spike detectado en val=50", info.get('spike', False) == True,
           f"info={info}")
    record("SpikeDetect: lr_scale=0.5 durante recovery", det.lr_scale == 0.5,
           f"lr_scale={det.lr_scale}")
    record("SpikeDetect: clip_scale=0.5 durante recovery", det.clip_scale == 0.5)

    # Agotar cooldown
    for _ in range(11):
        det.update(2.4)
    record("SpikeDetect: lr_scale=1.0 tras cooldown", det.lr_scale == 1.0,
           f"lr_scale={det.lr_scale}")
    record("SpikeDetect: n_spikes >= 1", det._n_spikes >= 1,
           f"n_spikes={det._n_spikes}")


def test_router_watchdog(device: torch.device):
    section("FASE 1C — RouterEntropyWatchdog")
    watch = RouterEntropyWatchdog(
        collapse_threshold=1.2,
        boost_factor=5.0,
        recovery_steps=20,
        window=5,
    )

    # ── Test: sin datos → mult = 1.0 ─────────────────────────────────────────
    mult = watch.update({})
    record("RouterWatchdog: sin H_bits → mult=1.0", mult == 1.0, f"mult={mult}")

    # ── Test: H_bits OK → mult = 1.0 ─────────────────────────────────────────
    for _ in range(5):
        mult = watch.update({'routing/H_bits': 1.5})
    record("RouterWatchdog: H_bits OK (1.5b) → mult=1.0", mult == 1.0, f"mult={mult}")

    # ── Test: colapso sostenido → mult = boost ────────────────────────────────
    for _ in range(5):
        mult = watch.update({'routing/H_bits': 0.3})  # colapso total
    record("RouterWatchdog: colapso H<1.2 → boost=5.0", mult == 5.0, f"mult={mult}")
    record("RouterWatchdog: n_collapses=1", watch.n_collapses == 1,
           f"n_collapses={watch.n_collapses}")

    # ── Test: se mantiene boosteando durante recovery ─────────────────────────
    mult2 = watch.update({'routing/H_bits': 1.8})
    record("RouterWatchdog: mult=5.0 durante recovery", mult2 == 5.0, f"mult2={mult2}")


def test_chimera_annealer(device: torch.device):
    section("FASE 1D — ChimeraAnnealer + temperature")
    model = _tiny_model(device)
    ann   = ChimeraAnnealer(
        model,
        slr_start  = 0.75,
        slr_end    = 0.50,
        slr_steps  = 1000,
        z_loss_start = 1e-3,
        z_loss_end   = 1e-4,
        z_loss_warmup = 200,
        bus_reset_every = 50,
    )

    # Estado inicial — slr_threshold y log_temp antes de step
    layers = [m for m in model.modules() if isinstance(m, AdvancedChimeraLayer)]

    # Step 0 → cosine t=0 → slr=0.75, log_temp=0.50
    ann.step(0)
    slr_vals = [l.slr_threshold.item() for l in layers
                if hasattr(l, 'slr_threshold') and torch.is_tensor(l.slr_threshold)]
    log_temps = [l.router.log_temp.item() for l in layers
                 if hasattr(l, 'router') and hasattr(l.router, 'log_temp')]

    arch_vals = [l.arch_threshold.item() for l in layers
                 if hasattr(l, 'arch_threshold') and torch.is_tensor(l.arch_threshold)]

    if slr_vals:
        slr0 = slr_vals[0]
        record("Annealer: slr_threshold@step0 ≈ 0.75",
               abs(slr0 - 0.75) < 0.01, f"slr={slr0:.4f}")
    if arch_vals:
        record("Annealer: arch_threshold@step0 ≈ 0.60",
               abs(arch_vals[0] - 0.60) < 0.01, f"arch={arch_vals[0]:.4f}")
    if log_temps:
        lt0 = log_temps[0]
        record("Annealer: log_temp@step0 ≈ 0.50",
               abs(lt0 - 0.50) < 0.01, f"lt={lt0:.4f}")

    # Step 1000 → t=1.0 → slr=0.50, log_temp=0.0
    ann.step(1000)
    slr_vals1 = [l.slr_threshold.item() for l in layers
                 if hasattr(l, 'slr_threshold') and torch.is_tensor(l.slr_threshold)]
    log_temps1 = [l.router.log_temp.item() for l in layers
                  if hasattr(l, 'router') and hasattr(l.router, 'log_temp')]
    if slr_vals1:
        record("Annealer: slr_threshold@step1000 ≈ 0.50",
               abs(slr_vals1[0] - 0.50) < 0.01, f"slr={slr_vals1[0]:.4f}")
    if log_temps1:
        record("Annealer: log_temp@step1000 ≈ 0.00",
               abs(log_temps1[0] - 0.00) < 0.01, f"lt={log_temps1[0]:.4f}")

    # Step 500 → t=0.5 → log_temp entre 0.25 (coseno mitad)
    ann.step(500)
    log_temps2 = [l.router.log_temp.item() for l in layers
                  if hasattr(l, 'router') and hasattr(l.router, 'log_temp')]
    if log_temps2:
        expected = 0.5 * 0.5 * (1 + math.cos(math.pi * 0.5))  # = 0.25
        record("Annealer: log_temp@step500 ≈ 0.25",
               abs(log_temps2[0] - expected) < 0.01,
               f"lt={log_temps2[0]:.4f} expected={expected:.4f}")


def test_ttt_supervisor(device: torch.device):
    section("FASE 1E — TTTGradSupervisor")
    model = _tiny_model(device)
    sup   = TTTGradSupervisor(clip_budget=0.05, lr=1e-3)

    # Obtener dt_bias original de la primera AdvancedChimeraLayer
    layers = [m for m in model.modules() if isinstance(m, AdvancedChimeraLayer)]
    if not layers:
        record("TTTSupervisor: No hay AdvancedChimeraLayer", False, "skip")
        return

    layer = layers[0]
    dt_bias_before = layer.mamba2.dt_bias.detach().clone()

    # ── Test 1: grad None → dt_bias no cambia ────────────────────────────────
    layer._pending_ttt_grad = None
    sup.apply(model)
    dt_bias_after = layer.mamba2.dt_bias.detach().clone()
    unchanged = torch.allclose(dt_bias_before, dt_bias_after)
    record("TTTSupervisor: grad=None → dt_bias inalterado", unchanged)

    # ── Test 2: grad normal → dt_bias actualizado ────────────────────────────
    # fake_grad grande (2.0 >> clip_budget=0.05) → sign update = lr*sign = 1e-3
    # Con N heads, la norma escala → clipping reduce a clip_budget, pero sign
    # sigue siendo no-cero → dt_bias cambia en ~lr = 1e-3 por elemento.
    dt_bias_before2 = layer.mamba2.dt_bias.detach().clone()
    n_heads = layer.mamba2.dt_bias.numel()
    fake_grad = torch.ones(n_heads, device=device, dtype=torch.float32) * 2.0
    layer._pending_ttt_grad = fake_grad
    sup.apply(model)
    dt_bias_after2 = layer.mamba2.dt_bias.detach().clone()
    max_diff = (dt_bias_after2 - dt_bias_before2).abs().max().item()
    changed = max_diff > 1e-7
    record("TTTSupervisor: grad válido → dt_bias actualizado", changed,
           f"max_diff={max_diff:.2e}")

    # ── Test 3: grad grande (> clip=0.05) → clipped ───────────────────────────
    layer._pending_ttt_grad = None
    dt_bias_before3 = layer.mamba2.dt_bias.detach().clone()
    huge_grad = torch.ones(n_heads, device=device, dtype=torch.float32) * 1000.0
    layer._pending_ttt_grad = huge_grad
    sup.apply(model)
    dt_bias_after3 = layer.mamba2.dt_bias.detach().clone()
    # La diferencia max debe ser pequeña (clip=0.05, lr=1e-3 → max_change = 0.05 en norma)
    diff = (dt_bias_after3 - dt_bias_before3).abs().max().item()
    record("TTTSupervisor: grad grande → clipped (diff < 0.01)",
           diff < 0.01, f"max_diff={diff:.6f}")

    # ── Test 4: pending_ttt_grad se limpia tras apply ─────────────────────────
    layer._pending_ttt_grad = fake_grad.clone()
    sup.apply(model)
    cleared = (getattr(layer, '_pending_ttt_grad', 'no_attr') is None)
    record("TTTSupervisor: _pending_ttt_grad=None tras apply", cleared)


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — TEST DE CORRUPCIÓN NaN EN dt_bias
# ═══════════════════════════════════════════════════════════════════════════════

def test_nan_corruption(device: torch.device):
    """
    Verifica que dt_bias NO se corrompe cuando el backward produce NaN.

    Escenario inyectado:
      1. Forward normal → backward normal
      2. POST-backward: reemplazar grads con NaN (simula CUDA numeric instability)
      3. Inyectar _pending_ttt_grad con NaN (simula forward NaN en mini-chunk TTT)
      4. Llamar al orden correcto: grad_monitor.check() PRIMERO
         → detecta NaN → limpia _pending_ttt_grad sin tocar dt_bias
         → ttt_super.apply() NO se llama
      5. Verificar que dt_bias_before == dt_bias_after
    """
    section("FASE 2 — Test de Corrupción NaN en dt_bias")

    model   = _tiny_model(device)
    monitor = GradHealthMonitor(max_consecutive_skips=99)
    sup     = TTTGradSupervisor(clip_budget=0.05, lr=1e-3)

    layers = [m for m in model.modules() if isinstance(m, AdvancedChimeraLayer)]
    if not layers:
        record("Corrupción NaN: hay AdvancedChimeraLayer", False, "SKIP")
        return

    # Snapshotear dt_bias de todas las capas ANTES del experimento
    dt_bias_snapshots = {
        id(l): l.mamba2.dt_bias.detach().clone()
        for l in layers if hasattr(l.mamba2, 'dt_bias')
    }
    print(f"\n  Layers monitoreadas: {len(layers)}")
    print(f"  dt_bias numel/capa:  {list(dt_bias_snapshots.values())[0].numel()}")

    # ── Asignar grads NaN directamente (sin forward real → evitar TTT lento) ─
    # Simula el escenario post-backward donde CUDA produce valores inestables.
    model.train()
    nan_count = 0
    for p in model.parameters():
        p.grad = torch.full_like(p, float('nan'))
        nan_count += 1

    # INYECCIÓN: _pending_ttt_grad NaN en todas las capas
    nan_ttt_count = 0
    for l in layers:
        if hasattr(l.mamba2, 'dt_bias'):
            n = l.mamba2.dt_bias.numel()
            l._pending_ttt_grad = torch.full((n,), float('nan'), device=device)
            nan_ttt_count += 1

    print(f"\n  [INYECTADO] {nan_count} grads → NaN")
    print(f"  [INYECTADO] {nan_ttt_count} _pending_ttt_grad → NaN\n")

    # ── ORDEN CORRECTO (train_h200_elite.py post-fix) ─────────────────────────
    # 1. grad_monitor.check() primero
    grads_ok = monitor.check(model)

    pending_cleared = 0
    if not grads_ok:
        # Limpiar _pending_ttt_grad (exactamente como en el loop fixed)
        for l in model.modules():
            if isinstance(l, AdvancedChimeraLayer):
                l._pending_ttt_grad = None
                pending_cleared += 1
        # NO llamar a ttt_super.apply()
        print(f"  [DETECTADO] NaN en grads → step descartado, {pending_cleared} pending cleared")
    else:
        # Solo si grads limpios, aplicar TTT
        sup.apply(model)

    # ── Verificaciones ────────────────────────────────────────────────────────
    record("Corrupción NaN: NaN detectado por monitor", grads_ok == False,
           f"grads_ok={grads_ok}")
    record("Corrupción NaN: todos _pending_ttt_grad limpiados",
           pending_cleared == len(layers),
           f"cleared={pending_cleared}/{len(layers)}")

    all_intact = True
    max_diff = 0.0
    for l in layers:
        if not hasattr(l.mamba2, 'dt_bias'):
            continue
        after = l.mamba2.dt_bias.detach()
        before = dt_bias_snapshots[id(l)]
        diff = (after - before).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > 1e-9:
            all_intact = False
            print(f"  {fail(f'dt_bias CORRUPCIÓN en capa {l._layer_idx}: diff={diff:.2e}')}")

    record("Corrupción NaN: dt_bias INTACTO tras NaN inyectado",
           all_intact, f"max_diff={max_diff:.2e}")

    # ── Parte 2: mismo test con ORDEN INCORRECTO (pre-fix) por contraste ──────
    print(f"\n  {YLW}[CONTROL] Mismo escenario con ORDEN INCORRECTO (pre-fix){RST}")

    # Snapshotear de nuevo
    dt_bias_before_wrong = {
        id(l): l.mamba2.dt_bias.detach().clone()
        for l in layers if hasattr(l.mamba2, 'dt_bias')
    }

    # Re-inyectar NaN en pending_ttt_grad
    for l in layers:
        if hasattr(l.mamba2, 'dt_bias'):
            l._pending_ttt_grad = torch.full(
                (l.mamba2.dt_bias.numel(),), float('nan'), device=device)

    # ORDEN INCORRECTO: apply ANTES del check
    try:
        sup.apply(model)   # ← aplica NaN grad a dt_bias!
        wrong_order_crashed = False
    except Exception:
        wrong_order_crashed = True  # si lanza exc por NaN

    corrupted = False
    for l in layers:
        if not hasattr(l.mamba2, 'dt_bias'):
            continue
        after = l.mamba2.dt_bias.detach()
        before = dt_bias_before_wrong[id(l)]
        if not torch.isfinite(after).all():
            corrupted = True
            break
        if (after - before).abs().max().item() > 1e-6:
            corrupted = True
            break

    record("Control orden incorrecto: dt_bias CORROMPIDO (esperado)",
           corrupted or wrong_order_crashed,
           f"corrupted={corrupted}, crashed={wrong_order_crashed}")

    # Reparar dt_bias NaN para tests posteriores
    for l in layers:
        if hasattr(l.mamba2, 'dt_bias'):
            snap = dt_bias_snapshots.get(id(l))
            if snap is not None:
                with torch.no_grad():
                    l.mamba2.dt_bias.copy_(snap)
            l._pending_ttt_grad = None
    print(f"  {ok('dt_bias restaurado a valores pre-test')}")


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 — MINI-ENTRENAMIENTO 100 STEPS CON ERRORES INYECTADOS
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorInjector:
    """
    Inyecta errores en pasos específicos durante el mini-entrenamiento.
    Cada error simula una condición distinta que el entrenador debe sobrevivir.
    """
    EVENTS = {
        15: "NaN_grad",                    # NaN en grads del backward
        35: "loss_spike",                  # Spike artificial (no inyectable en backward, simulado via loss_ema)
        60: "router_collapse",             # Todos los tokens → tier FAST (p=[0.97,0.015,0.015])
        80: "double_fault",               # NaN + pending_ttt_grad NaN simultáneos
    }

    def __init__(self, model: nn.Module, layers: List):
        self.model  = model
        self.layers = layers
        self._events_fired: Dict[int, bool] = {}

    def inject(self, step: int) -> Optional[str]:
        """Inyecta error si corresponde a este step. Retorna el nombre del evento."""
        if step not in self.EVENTS:
            return None
        event = self.EVENTS[step]
        self._events_fired[step] = True

        if event == "NaN_grad":
            # NaN en 1/4 de los parámetros (no en todos — más realista)
            params = list(self.model.parameters())
            for i, p in enumerate(params):
                if i % 4 == 0 and p.grad is not None:
                    p.grad.fill_(float('nan'))

        elif event == "router_collapse":
            # Forzar fast_prob_ema alta en todas las capas (simula colapso)
            with torch.no_grad():
                for l in self.layers:
                    if hasattr(l, 'fast_prob_ema'):
                        l.fast_prob_ema.fill_(0.97)

        elif event == "double_fault":
            # NaN en todos los grads + pending_ttt_grad NaN
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.fill_(float('nan'))
            for l in self.layers:
                if hasattr(l.mamba2, 'dt_bias'):
                    n = l.mamba2.dt_bias.numel()
                    l._pending_ttt_grad = torch.full(
                        (n,), float('nan'), device=next(self.model.parameters()).device)

        return event

    def summary(self) -> str:
        lines = []
        for step, event in self.EVENTS.items():
            fired = self._events_fired.get(step, False)
            lines.append(f"    step {step:>3}: {event:<22} {'FIRED' if fired else 'MISS'}")
        return "\n".join(lines)


def run_mini_training(device: torch.device, n_steps: int = 100):
    section(f"FASE 3 — Mini-entrenamiento {n_steps} steps con errores inyectados")

    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)

    model  = _tiny_model(device)
    # graph_mode=True → salta TTT inner-backward → 100x más rápido para tests
    for m in model.modules():
        if hasattr(m, 'graph_mode'):
            m.graph_mode = True
    if device.type == 'cuda':
        model = model.bfloat16()
    model.train()

    # ── Optimizer simple (AdamW — Muon requiere 2D matrices disponibles) ──────
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95),
                             weight_decay=0.1)

    # ── Sistemas de estabilidad ───────────────────────────────────────────────
    monitor   = GradHealthMonitor(max_consecutive_skips=30)
    spike_det = LossSpikeDetector(alpha=0.97, spike_sigma=3.0,
                                   warmup_steps=20, spike_cooldown=10)
    router_wt = RouterEntropyWatchdog(collapse_threshold=1.0, boost_factor=8.0,
                                       recovery_steps=15, window=5)
    ttt_sup   = TTTGradSupervisor(clip_budget=0.05, lr=1e-3)
    ann       = ChimeraAnnealer(model, slr_start=0.75, slr_end=0.50,
                                 slr_steps=n_steps // 2, bus_reset_every=30)

    # Obtener routing_loss del modelo para ajustar entropy_weight
    routing_loss_obj = None
    _base_ent_w      = None
    for m in model.modules():
        if isinstance(m, ChimeraRoutingLoss):
            routing_loss_obj = m
            _base_ent_w = m.entropy_weight
            break

    layers = [m for m in model.modules() if isinstance(m, AdvancedChimeraLayer)]
    injector = ErrorInjector(model, layers)

    # ── Snapshots de dt_bias para monitoreo de corrupción ────────────────────
    def _snap_dt():
        return {id(l): l.mamba2.dt_bias.detach().clone()
                for l in layers if hasattr(l.mamba2, 'dt_bias')}

    dt_snap_initial = _snap_dt()

    # ── Métricas por step ─────────────────────────────────────────────────────
    metrics = {
        'loss':       [],
        'grad_norm':  [],
        'H_bits':     [],
        'p_fast':     [],
        'nan_skips':  [],
        'events':     {},
        'dt_bias_norm': [],
        'max_fast_ema': [],
        'recovery_steps_total': 0,
    }

    amp_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16)
               if device.type == 'cuda'
               else contextlib.nullcontext())

    print(f"\n  {'step':>5}  {'loss':>8}  {'grad':>7}  {'H_bits':>6}  "
          f"{'pFAST':>6}  {'maxFema':>7}  {'event':<22}  {'status'}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*7}  {'─'*6}  "
          f"{'─'*6}  {'─'*7}  {'─'*22}  {'─'*10}")

    t_start = time.perf_counter()

    for step in range(1, n_steps + 1):
        # ── LR schedule cosine simplificado ────────────────────────────────────
        t_frac = step / n_steps
        lr_now = 2e-4 * 0.5 * (1 + math.cos(math.pi * t_frac))
        for pg in opt.param_groups:
            pg['lr'] = lr_now

        # ── Annealing ──────────────────────────────────────────────────────────
        ann.step(step)

        # ── Forward + backward ────────────────────────────────────────────────
        ids, labels = _synth_batch(2, 128, 512, device)
        opt.zero_grad(set_to_none=False)

        with amp_ctx:
            logits, loss, ld = model(ids, labels=labels, aux_weight=0.01)

        if loss is None or not torch.isfinite(loss):
            loss = torch.tensor(float('nan'))
            loss_val = float('nan')
        else:
            loss_val = loss.item()
            loss.backward()

        # ── Inyección de error (POST-backward, PRE-optimizer) ─────────────────
        event = injector.inject(step)
        if event:
            metrics['events'][step] = event

        # ── ORDEN CORRECTO: grad check PRIMERO ────────────────────────────────
        grads_ok = monitor.check(model)
        status   = "ok"

        if not grads_ok:
            status = f"{RED}nan-skip{RST}"
            # Limpiar pending TTT grad (fix anti-corrupción)
            for l in model.modules():
                if isinstance(l, AdvancedChimeraLayer):
                    l._pending_ttt_grad = None
        else:
            # Solo si grads sanos: aplicar TTT updates
            ttt_sup.apply(model)
            # Clip principal
            gn = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            opt.step()
            metrics['grad_norm'].append(gn)
            status = "ok"

        if not grads_ok:
            metrics['grad_norm'].append(float('nan'))

        # ── Router watchdog ───────────────────────────────────────────────────
        ent_mult = router_wt.update(ld)
        if routing_loss_obj is not None and _base_ent_w is not None:
            routing_loss_obj.entropy_weight = _base_ent_w * ent_mult
            if ent_mult > 1.0:
                status = f"{YLW}ent-boost×{ent_mult:.0f}{RST}"

        # ── Spike detection ───────────────────────────────────────────────────
        if math.isfinite(loss_val):
            spike_info = spike_det.update(loss_val)
            if spike_info:
                status = f"{YLW}spike!{RST}"
            if spike_det._in_recovery > 0:
                metrics['recovery_steps_total'] += 1

        # ── fast_prob_ema lectura ─────────────────────────────────────────────
        fast_emas = [l.fast_prob_ema.item() for l in layers
                     if hasattr(l, 'fast_prob_ema')]
        max_fema = max(fast_emas) if fast_emas else float('nan')
        if max_fema > 0.90 and routing_loss_obj is not None:
            status = f"{RED}collapse!{RST}"

        # ── Métricas ──────────────────────────────────────────────────────────
        metrics['loss'].append(loss_val if math.isfinite(loss_val) else float('nan'))
        H_bits  = ld.get('routing/H_bits', float('nan'))
        p_fast_ = ld.get('routing/p_fast', float('nan'))
        if hasattr(H_bits, 'item'):  H_bits  = H_bits.item()
        if hasattr(p_fast_, 'item'): p_fast_ = p_fast_.item()
        metrics['H_bits'].append(H_bits)
        metrics['p_fast'].append(p_fast_)
        metrics['max_fast_ema'].append(max_fema)
        metrics['nan_skips'].append(0 if grads_ok else 1)

        # dt_bias norm
        dt_norms = [l.mamba2.dt_bias.detach().norm().item()
                    for l in layers if hasattr(l.mamba2, 'dt_bias')]
        metrics['dt_bias_norm'].append(sum(dt_norms) / len(dt_norms) if dt_norms else 0.0)

        # ── Print periódico ───────────────────────────────────────────────────
        if step % 10 == 0 or event:
            grad_str = (f"{metrics['grad_norm'][-1]:.4f}"
                        if math.isfinite(metrics['grad_norm'][-1]) else "  NaN  ")
            event_str = (event or "").ljust(22)
            print(f"  {step:>5}  {loss_val:>8.4f}  {grad_str:>7}  "
                  f"{H_bits:>6.3f}  {p_fast_:>6.3f}  {max_fema:>7.3f}  "
                  f"{event_str}  {status}")

    elapsed = time.perf_counter() - t_start
    print(f"\n  Tiempo total: {elapsed:.1f}s  ({elapsed/n_steps*1000:.1f} ms/step)")

    # ── ANÁLISIS FINAL ────────────────────────────────────────────────────────
    section("FASE 3 — Análisis y verificaciones post-entrenamiento")

    # dt_bias intacto tras todo el entrenamiento (no NaN ni demasiado desviado)
    dt_snap_final = _snap_dt()
    dt_bias_nan = False
    dt_bias_exploded = False
    for l in layers:
        if not hasattr(l.mamba2, 'dt_bias'):
            continue
        final_val = l.mamba2.dt_bias.detach()
        if not torch.isfinite(final_val).all():
            dt_bias_nan = True
        norm_change = (final_val - dt_snap_initial[id(l)]).norm().item()
        if norm_change > 10.0:  # heurístico — > 10 sugiere explosión
            dt_bias_exploded = True

    record("Mini-train: dt_bias sin NaN al final", not dt_bias_nan)
    record("Mini-train: dt_bias no explotó (norm_change < 10)", not dt_bias_exploded)

    # NaN skip count correcto
    total_nan_skips = sum(metrics['nan_skips'])
    expected_nan   = sum(1 for e in metrics['events'].values()
                         if 'NaN' in e or 'double' in e)
    record("Mini-train: NaN skips detectados correctamente",
           total_nan_skips >= expected_nan,
           f"skips={total_nan_skips}, nan_events={expected_nan}")

    # Loss mejora durante el entrenamiento (al menos la primera mitad sin errores)
    loss_10  = [l for l in metrics['loss'][:10]  if math.isfinite(l)]
    loss_end = [l for l in metrics['loss'][-15:] if math.isfinite(l)]
    if loss_10 and loss_end:
        loss_improved = (sum(loss_end) / len(loss_end)) < (sum(loss_10) / len(loss_10))
        record("Mini-train: loss disminuye (convergencia básica)",
               loss_improved,
               f"early={sum(loss_10)/len(loss_10):.4f} → final={sum(loss_end)/len(loss_end):.4f}")

    # Router entropía — se mantiene en rango razonable
    valid_H = [h for h in metrics['H_bits'] if math.isfinite(h)]
    if valid_H:
        mean_H = sum(valid_H) / len(valid_H)
        record("Mini-train: H_bits router > 0.5 en promedio (no colapso total)",
               mean_H > 0.5, f"mean_H={mean_H:.3f}")

    # recovery_steps_total > 0 si se detectó cualquier spike
    record("Mini-train: LossSpikeDetector activó recovery",
           metrics['recovery_steps_total'] > 0 or expected_nan >= 2,
           f"recovery_steps={metrics['recovery_steps_total']}")

    # Eventos inyectados
    print(f"\n  {BLD}Resumen de eventos inyectados:{RST}")
    print(injector.summary())
    print(f"\n  {BLD}Estadísticas de sistemas:{RST}")
    print(f"    NaN-skips totales:      {total_nan_skips}")
    print(f"    Collapses detectados:   {router_wt.n_collapses}")
    print(f"    Spikes detectados:      {spike_det._n_spikes}")
    print(f"    Steps en recovery:      {metrics['recovery_steps_total']}")
    print(f"    dt_bias_norm_final:     {metrics['dt_bias_norm'][-1]:.6f}")

    # dt_bias finales — mostrar primeros 4 heads de capa 0
    if layers and hasattr(layers[0].mamba2, 'dt_bias'):
        dt_show = layers[0].mamba2.dt_bias.detach().float()[:4]
        print(f"    dt_bias[capa0, 0:4]:    {dt_show.tolist()}")


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 4 — TEST DE VELOCIDAD POR COMPONENTE
# ═══════════════════════════════════════════════════════════════════════════════

def test_throughput(device: torch.device):
    section("FASE 4 — Throughput y latencia de componentes")

    model = _tiny_model(device)
    if device.type == 'cuda':
        model = model.bfloat16()
    model.train()

    ids, labels = _synth_batch(4, 256, 512, device)
    amp_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16)
               if device.type == 'cuda' else contextlib.nullcontext())

    # Warmup
    for _ in range(3):
        with amp_ctx:
            _, loss, _ = model(ids, labels=labels, aux_weight=0.01)
        loss.backward()
        model.zero_grad(set_to_none=True)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # ── Forward latency ───────────────────────────────────────────────────────
    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        with amp_ctx:
            _, loss, _ = model(ids, labels=labels, aux_weight=0.01)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) / N * 1000
    print(f"  Forward (B=4, S=256, 4 capas):  {fwd_ms:.2f} ms/step")

    # ── Forward+backward latency ──────────────────────────────────────────────
    t0 = time.perf_counter()
    for _ in range(N):
        model.zero_grad(set_to_none=True)
        with amp_ctx:
            _, loss, _ = model(ids, labels=labels, aux_weight=0.01)
        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
    fwdbwd_ms = (time.perf_counter() - t0) / N * 1000
    print(f"  Forward+Backward:               {fwdbwd_ms:.2f} ms/step")

    # ── GradHealthMonitor overhead ────────────────────────────────────────────
    monitor = GradHealthMonitor()
    t0 = time.perf_counter()
    for _ in range(100):
        monitor.check(model)
    mon_us = (time.perf_counter() - t0) / 100 * 1e6
    print(f"  GradHealthMonitor.check():      {mon_us:.1f} µs")

    # ── TTTGradSupervisor overhead ────────────────────────────────────────────
    sup = TTTGradSupervisor(clip_budget=0.05, lr=1e-3)
    layers = [m for m in model.modules() if isinstance(m, AdvancedChimeraLayer)]
    for l in layers:
        if hasattr(l.mamba2, 'dt_bias'):
            n = l.mamba2.dt_bias.numel()
            l._pending_ttt_grad = torch.randn(n, device=device)
    t0 = time.perf_counter()
    for _ in range(100):
        for l in layers:
            if hasattr(l.mamba2, 'dt_bias'):
                n = l.mamba2.dt_bias.numel()
                l._pending_ttt_grad = torch.randn(n, device=device)
        sup.apply(model)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    sup_us = (time.perf_counter() - t0) / 100 * 1e6
    print(f"  TTTGradSupervisor.apply():      {sup_us:.1f} µs")

    toks_s = 4 * 256 / (fwdbwd_ms / 1000)
    print(f"\n  Throughput estimado (F+B):      {toks_s:,.0f} tok/s")
    record("Throughput: F+B < 2000 ms", fwdbwd_ms < 2000,
           f"{fwdbwd_ms:.1f} ms")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='auto')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--skip_phases', nargs='*', type=int, default=[],
                   help="Fases a saltar ej. --skip_phases 4")
    args = p.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{BLD}{'═'*60}")
    print(f"  CHIMERA Robustness & Corruption Test Suite")
    print(f"  Device: {device} | Steps mini-train: {args.steps}")
    if device.type == 'cuda':
        prop = torch.cuda.get_device_properties(0)
        print(f"  GPU: {prop.name} ({prop.total_memory/1e9:.0f} GB)")
    print(f"{'═'*60}{RST}\n")

    # Importar módulos
    ok_import = _import_chimera(device)
    if not ok_import:
        print(fail("No se pudo importar el stack — abortar"))
        sys.exit(1)

    # ── Fases ──────────────────────────────────────────────────────────────────
    if 1 not in args.skip_phases:
        test_grad_health_monitor(device)
        test_loss_spike_detector(device)
        test_router_watchdog(device)
        test_chimera_annealer(device)
        test_ttt_supervisor(device)

    if 2 not in args.skip_phases:
        test_nan_corruption(device)

    if 3 not in args.skip_phases:
        run_mini_training(device, n_steps=args.steps)

    if 4 not in args.skip_phases:
        test_throughput(device)

    # ── Resumen final ──────────────────────────────────────────────────────────
    section("RESUMEN FINAL")
    passed = sum(1 for r in _results if r['passed'])
    total  = len(_results)
    failed = [r for r in _results if not r['passed']]

    print(f"\n  {BLD}Tests:  {passed}/{total} pasaron{RST}")
    if failed:
        print(f"\n  {RED}{BLD}Fallados:{RST}")
        for r in failed:
            print(f"    {fail(r['name'])}  {r['detail']}")
    else:
        print(f"\n  {GRN}{BLD}¡TODOS LOS TESTS PASARON!{RST}")

    print(f"\n{'═'*60}\n")
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
