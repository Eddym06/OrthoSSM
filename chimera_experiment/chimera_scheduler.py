"""
ChimeraWarmupScheduler — Problema 9 / Sprint 6.1
==================================================
Warm-up escalonado en 3 fases para estabilizar training del stack CHIMERA.

Fase 1 (steps 0–warm1):     Solo Mamba2 SSD. ttt_lr=0, slr_gate→0, bus_gate→0.
Fase 2 (steps warm1–warm2): TTT-Lite + SLR con ramp lineal.
Fase 3 (steps warm2+):      CHIMERA completo (TTT-Full, archive, bus).

Uso:
    scheduler = ChimeraWarmupScheduler(model_stack, warm1=1000, warm2=3000)
    # En cada step de training:
    scheduler.step(global_step)

El scheduler modifica directamente los parámetros de gate (slr_gate,
bus.gate, ttt_lr, ttt_full_scale) de cada capa — no crea un optimizer
separado, trabaja sobre los `nn.Parameter` directamente para compatibilidad
con cualquier optimizer externo (AdamW, etc.).
"""
import math
import torch
import torch.nn as nn
from typing import List


class ChimeraWarmupScheduler:
    """
    Controla el warm-up escalonado de un stack de AdvancedChimeraLayers.

    Args:
        layers:  lista de AdvancedChimeraLayer (no ChimeraStack, acceso directo)
        warm1:   step donde termina Fase 1 (default 1000)
        warm2:   step donde termina Fase 2 (default 3000)
        ttt_lr_target:   lr máximo para TTT-Lite al final de Fase 2 (default 1e-3)
        slr_gate_target: gate máximo SLR al final del warm-up (no clampeado)
        bus_gate_target: gate máximo bus   al final del warm-up
    """

    def __init__(
        self,
        layers: List[nn.Module],
        warm1: int = 1000,
        warm2: int = 3000,
        ttt_lr_target: float = 1e-3,
    ):
        self.layers        = layers
        self.warm1         = warm1
        self.warm2         = warm2
        self.ttt_lr_target = ttt_lr_target
        self._global_step  = 0

        # Guardar valores objetivo de los gates (el nn.Parameter init_value)
        # para restaurarlos en Fase 3 (no los tocamos en Fase 3+)
        self._slr_gate_init   = {}   # layer_idx → tensor inicial del param
        self._bus_gate_init   = {}
        self._tff_scale_init  = {}   # ttt_full_scale init

        for i, layer in enumerate(layers):
            self._slr_gate_init[i]  = layer.slr.merge_gate.data.clone()
            self._bus_gate_init[i]  = layer.bus.gate.data.clone()
            self._tff_scale_init[i] = layer.ttt_full_scale.data.clone()

    # ─────────────────────────────────────────────────────────────────────────

    def _frac_phase2(self, step: int) -> float:
        """Fracción de avance dentro de Fase 2 (0.0 → 1.0)."""
        if step <= self.warm1:
            return 0.0
        if step >= self.warm2:
            return 1.0
        return (step - self.warm1) / max(1, self.warm2 - self.warm1)

    def _phase(self, step: int) -> int:
        if step < self.warm1:
            return 1
        if step < self.warm2:
            return 2
        return 3

    def step(self, global_step: int):
        """
        Llamar ANTES del forward de cada mini-batch.
        Ajusta los parámetros de gate y ttt_lr de todas las capas.
        """
        self._global_step = global_step
        phase = self._phase(global_step)
        frac2 = self._frac_phase2(global_step)

        for i, layer in enumerate(self.layers):
            if phase == 1:
                # ── Fase 1: solo Mamba2, todo desactivado ─────────────────
                # SLR gate → escalar a mínimo (sigmoid(-10) ≈ 0)
                with torch.no_grad():
                    layer.slr.merge_gate.fill_(-10.0)
                    layer.bus.gate.fill_(0.0)          # sigmoid(0)=0.5 → bus off
                    layer.ttt_full_scale.fill_(-10.0)  # TTT-Full off
                layer.ttt_lr = 0.0                     # sin Lion update

            elif phase == 2:
                # ── Fase 2: ramp lineal 0→target ─────────────────────────
                # SLR gate: -10 → init_value  (ramp en logit space)
                init_slr = self._slr_gate_init[i].item()
                slr_logit = -10.0 + frac2 * (init_slr - (-10.0))
                with torch.no_grad():
                    layer.slr.merge_gate.fill_(slr_logit)

                # Bus gate: 0 → init_value
                init_bus = self._bus_gate_init[i].item()
                bus_logit = -10.0 * (1.0 - frac2) + init_bus * frac2
                with torch.no_grad():
                    layer.bus.gate.fill_(bus_logit)

                # TTT-Full: off durante toda Fase 2 (se activa en Fase 3)
                with torch.no_grad():
                    layer.ttt_full_scale.fill_(-10.0)

                # TTT-Lite lr: 0 → ttt_lr_target
                layer.ttt_lr = frac2 * self.ttt_lr_target

            else:
                # ── Fase 3: CHIMERA completo ──────────────────────────────
                # Restaurar gates a sus valores entrenados (no tocar)
                # Solo asegurar ttt_lr en target y TTT-Full scale liberado
                layer.ttt_lr = self.ttt_lr_target
                # TTT-Full: restaurar al valor init (permite aprendizaje)
                # Solo si aún está en -10 (primera vez que llegamos a Fase 3)
                if layer.ttt_full_scale.data.item() < -9.0:
                    with torch.no_grad():
                        layer.ttt_full_scale.copy_(self._tff_scale_init[i])

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def current_phase(self) -> int:
        return self._phase(self._global_step)

    def state_dict(self):
        return {
            'global_step': self._global_step,
            'warm1': self.warm1,
            'warm2': self.warm2,
            'ttt_lr_target': self.ttt_lr_target,
        }

    def load_state_dict(self, d):
        self._global_step = d['global_step']
        self.warm1        = d['warm1']
        self.warm2        = d['warm2']
        self.ttt_lr_target = d['ttt_lr_target']

    def __repr__(self):
        return (f"ChimeraWarmupScheduler("
                f"step={self._global_step}, "
                f"phase={self.current_phase}, "
                f"warm1={self.warm1}, warm2={self.warm2})")


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from advanced_chimera import AdvancedChimeraLayer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [AdvancedChimeraLayer(d_model=256).to(device) for _ in range(3)]
    sched  = ChimeraWarmupScheduler(layers, warm1=100, warm2=300)

    print("=== ChimeraWarmupScheduler test ===")
    checkpoints = [0, 50, 100, 150, 200, 250, 300, 400]
    header = f"{'step':>6} | {'phase':>5} | {'ttt_lr':>8} | {'slr_gate(σ)':>11} | {'bus_gate(σ)':>11} | {'tff_scale(σ)':>12}"
    print(header)
    print("─" * len(header))

    for step in checkpoints:
        sched.step(step)
        l = layers[0]
        slr_sig = torch.sigmoid(l.slr.merge_gate).item()
        bus_sig = torch.sigmoid(l.bus.gate).item()
        tff_sig = torch.sigmoid(l.ttt_full_scale).item()
        print(f"  {step:>4}  |  {sched.current_phase:>3}  |  {l.ttt_lr:.5f}  "
              f"|    {slr_sig:.4f}   |    {bus_sig:.4f}   |    {tff_sig:.4f}")

    print()
    # Verificar fases
    sched.step(0);   assert sched.current_phase == 1, "Fase 1 OK"
    sched.step(150); assert sched.current_phase == 2, "Fase 2 OK"
    sched.step(400); assert sched.current_phase == 3, "Fase 3 OK"
    print("[✓] Fases correctas")

    # En Fase 1, ttt_lr debe ser 0
    sched.step(0)
    assert layers[0].ttt_lr == 0.0, "Fase1 ttt_lr=0"
    print("[✓] ttt_lr=0 en Fase 1")

    # En Fase 3, ttt_lr = target
    sched.step(500)
    assert layers[0].ttt_lr == sched.ttt_lr_target, "Fase3 ttt_lr=target"
    print("[✓] ttt_lr=target en Fase 3")

    print("\n[SUCCESS] ChimeraWarmupScheduler OK")
