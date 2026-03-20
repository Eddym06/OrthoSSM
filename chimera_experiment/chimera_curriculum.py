"""
chimera_curriculum.py — Curriculum Learning Adaptativo para CHIMERA
===================================================================
CHIMERA tiene una señal interna que los Transformers NO tienen: el router
de complejidad indica en tiempo real qué fracción del batch es procesada
por cada tier (FAST/HYBRID/FULL). Esto es un proxy directo de la dificultad
percibida por el modelo.

Principio: NO usar fases fijas por conteo de tokens (como propone Gemini),
sino un curriculum ADAPTATIVO que reacciona a las estadísticas del router.

Señales del router:
  - p_fast alta (>0.75):  el modelo está en zona de confort → subir dificultad
  - p_full alta (>0.25):  el modelo está luchando → mantener o bajar
  - p_hybrid alta (>0.40): el modelo está aprendiendo activamente → buen balance

Ejes de dificultad:
  1. Complejidad lingüística: Wiki → código → papers/legales
  2. Longitud de secuencia:   256 → 2K → 8K → 32K+
  3. Diversidad de dominio:   mismo dominio → multi-dominio

Uso:
    from chimera_curriculum import AdaptiveCurriculum

    curriculum = AdaptiveCurriculum(
        phases=[
            CurriculumPhase("foundation", complexity=0.3, max_seq_len=512),
            CurriculumPhase("intermediate", complexity=0.6, max_seq_len=2048),
            CurriculumPhase("advanced", complexity=0.9, max_seq_len=8192),
        ]
    )

    for step in range(total_steps):
        routing_stats = get_routing_stats(model)
        mixture = curriculum.get_mixture(step, routing_stats)
        batch = dataloader.sample(mixture)
        ...
        curriculum.update(routing_stats, loss)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum


@dataclass
class RoutingStats:
    """Estadísticas del router para una ventana de entrenamiento.

    Se extraen del modelo durante el training loop y se pasan al
    curriculum scheduler para que adapte la mezcla de datos.
    """
    p_fast: float = 0.333        # probabilidad promedio del tier FAST
    p_hybrid: float = 0.333      # probabilidad promedio del tier HYBRID
    p_full: float = 0.333        # probabilidad promedio del tier FULL
    loss: float = 0.0            # loss LM actual
    loss_delta: float = 0.0      # cambio de loss vs paso anterior
    step: int = 0                # paso de entrenamiento
    gradient_norm: float = 0.0   # norma del gradiente (proxy de estabilidad)


@dataclass
class DataMixture:
    """Descripción de la mezcla de datos para un paso de entrenamiento.

    Los pesos son relativos (se normalizan a sum=1 internamente).
    max_seq_len controla la longitud máxima de secuencias muestreadas.
    """
    # Pesos por categoría de dificultad
    easy_weight: float = 0.5       # Wikipedia, libros simples
    medium_weight: float = 0.3     # Código, artículos técnicos
    hard_weight: float = 0.2       # Papers largos, documentos legales

    # Longitud de secuencia
    max_seq_len: int = 2048        # Max tokens por secuencia

    # Nivel de complejidad objetivo (0-1)
    target_complexity: float = 0.5

    # Metadatos
    phase_name: str = "default"
    confidence: float = 1.0        # confianza del scheduler en esta mezcla

    @property
    def weights(self) -> dict:
        """Pesos normalizados."""
        total = self.easy_weight + self.medium_weight + self.hard_weight
        if total <= 0:
            return {'easy': 0.333, 'medium': 0.333, 'hard': 0.334}
        return {
            'easy': self.easy_weight / total,
            'medium': self.medium_weight / total,
            'hard': self.hard_weight / total,
        }


@dataclass
class CurriculumPhase:
    """Definición de una fase de curriculum.

    Cada fase tiene un rango de complejidad objetivo y criterios de
    transición basados en las estadísticas del router.
    """
    name: str
    complexity: float = 0.5        # complejidad objetivo (0=fácil, 1=difícil)
    max_seq_len: int = 2048        # longitud máxima de secuencia
    min_steps: int = 0             # pasos mínimos antes de poder avanzar
    max_steps: int = 0             # pasos máximos (0=sin límite)

    # Criterios de transición
    advance_p_fast_threshold: float = 0.70   # si p_fast > esto → avanzar
    advance_loss_plateau: float = 0.01       # si |loss_delta| < esto → avanzar
    retreat_p_full_threshold: float = 0.35   # si p_full > esto → retroceder

    # Mezcla de datos para esta fase
    easy_weight: float = 0.5
    medium_weight: float = 0.3
    hard_weight: float = 0.2


class TransitionType(Enum):
    """Tipo de transición entre fases."""
    NONE = "none"
    ADVANCE = "advance"       # subir dificultad
    RETREAT = "retreat"       # bajar dificultad
    STAY = "stay"             # mantener (por min_steps o estabilidad)


class AdaptiveCurriculum:
    """
    Scheduler de curriculum learning adaptativo para CHIMERA.

    Usa las probabilidades del router como proxy de dificultad percibida
    para ajustar dinámicamente la mezcla de datos y la longitud de secuencia.

    A diferencia de un curriculum fijo, este scheduler:
      1. Puede retroceder si el modelo no converge (retreat)
      2. Transiciona suavemente (interpolación entre fases)
      3. Respeta min_steps para consolidación
      4. Monitorea estabilidad (gradient_norm, loss_delta)

    Protocolo:
      1. curriculum.get_mixture(step, stats) → DataMixture
      2. Dataloader muestrea según DataMixture
      3. curriculum.update(stats) registra progreso

    Log:
      curriculum.get_log() → historial de transiciones
    """

    def __init__(
        self,
        phases: List[CurriculumPhase] = None,
        transition_smoothness: float = 0.1,
        ema_alpha: float = 0.05,
    ):
        """
        Args:
            phases: lista de fases ordenadas por dificultad ascendente.
                    Si None, usa fases por defecto.
            transition_smoothness: factor de suavizado para transiciones
                                   (0=instantánea, 1=muy suave)
            ema_alpha: alpha para EMA de estadísticas del router
        """
        self.phases = phases or self._default_phases()
        self.transition_smoothness = transition_smoothness
        self.ema_alpha = ema_alpha

        # Estado interno
        self._current_phase_idx = 0
        self._steps_in_phase = 0
        self._total_steps = 0
        self._blend_factor = 0.0      # 0=100% fase actual, 1=100% siguiente

        # EMA de routing stats (suavizado para evitar transiciones ruidosas)
        self._ema_stats = RoutingStats()
        self._prev_loss = float('inf')
        self._loss_window: list = []       # ventana móvil para detectar plateau

        # Historial de transiciones
        self._log: List[dict] = []

    @staticmethod
    def _default_phases() -> List[CurriculumPhase]:
        """Fases por defecto para un curriculum de 3 etapas."""
        return [
            CurriculumPhase(
                name="foundation",
                complexity=0.3,
                max_seq_len=512,
                min_steps=1000,
                advance_p_fast_threshold=0.70,
                easy_weight=0.7,
                medium_weight=0.2,
                hard_weight=0.1,
            ),
            CurriculumPhase(
                name="intermediate",
                complexity=0.6,
                max_seq_len=2048,
                min_steps=2000,
                advance_p_fast_threshold=0.65,
                advance_loss_plateau=0.005,
                retreat_p_full_threshold=0.40,
                easy_weight=0.3,
                medium_weight=0.5,
                hard_weight=0.2,
            ),
            CurriculumPhase(
                name="advanced",
                complexity=0.9,
                max_seq_len=8192,
                min_steps=3000,
                advance_p_fast_threshold=0.60,
                retreat_p_full_threshold=0.35,
                easy_weight=0.1,
                medium_weight=0.3,
                hard_weight=0.6,
            ),
        ]

    @property
    def current_phase(self) -> CurriculumPhase:
        return self.phases[self._current_phase_idx]

    @property
    def next_phase(self) -> Optional[CurriculumPhase]:
        idx = self._current_phase_idx + 1
        if idx < len(self.phases):
            return self.phases[idx]
        return None

    @property
    def prev_phase(self) -> Optional[CurriculumPhase]:
        idx = self._current_phase_idx - 1
        if idx >= 0:
            return self.phases[idx]
        return None

    def get_mixture(
        self,
        step: int,
        routing_stats: RoutingStats = None,
    ) -> DataMixture:
        """
        Calcula la mezcla de datos para el paso actual.

        Si routing_stats se proporciona, actualiza el EMA y potencialmente
        transiciona entre fases.

        Args:
            step: paso de entrenamiento actual
            routing_stats: estadísticas del router (optional)

        Returns:
            DataMixture con pesos y max_seq_len
        """
        if routing_stats is not None:
            self._update_ema(routing_stats)

        phase = self.current_phase
        nxt = self.next_phase

        # Blend suave entre fases si estamos en transición
        if self._blend_factor > 0 and nxt is not None:
            bf = self._blend_factor
            mixture = DataMixture(
                easy_weight=phase.easy_weight * (1 - bf) + nxt.easy_weight * bf,
                medium_weight=phase.medium_weight * (1 - bf) + nxt.medium_weight * bf,
                hard_weight=phase.hard_weight * (1 - bf) + nxt.hard_weight * bf,
                max_seq_len=int(phase.max_seq_len * (1 - bf) + nxt.max_seq_len * bf),
                target_complexity=phase.complexity * (1 - bf) + nxt.complexity * bf,
                phase_name=f"{phase.name}→{nxt.name}",
                confidence=1.0 - bf * 0.3,
            )
        else:
            mixture = DataMixture(
                easy_weight=phase.easy_weight,
                medium_weight=phase.medium_weight,
                hard_weight=phase.hard_weight,
                max_seq_len=phase.max_seq_len,
                target_complexity=phase.complexity,
                phase_name=phase.name,
            )

        return mixture

    def update(self, routing_stats: RoutingStats):
        """
        Actualiza el estado del curriculum basado en estadísticas del router.

        Decide si transicionar, mantener o retroceder de fase.
        """
        self._total_steps += 1
        self._steps_in_phase += 1
        self._update_ema(routing_stats)

        # Detectar tipo de transición
        transition = self._evaluate_transition()

        if transition == TransitionType.ADVANCE:
            self._advance_phase()
        elif transition == TransitionType.RETREAT:
            self._retreat_phase()

        # Actualizar blend factor suavemente
        if self._blend_factor > 0:
            self._blend_factor = max(0, self._blend_factor - self.transition_smoothness)

    def _update_ema(self, stats: RoutingStats):
        """Actualiza las estadísticas EMA."""
        a = self.ema_alpha
        self._ema_stats.p_fast = (1 - a) * self._ema_stats.p_fast + a * stats.p_fast
        self._ema_stats.p_hybrid = (1 - a) * self._ema_stats.p_hybrid + a * stats.p_hybrid
        self._ema_stats.p_full = (1 - a) * self._ema_stats.p_full + a * stats.p_full
        self._ema_stats.loss = (1 - a) * self._ema_stats.loss + a * stats.loss
        self._ema_stats.gradient_norm = ((1 - a) * self._ema_stats.gradient_norm
                                          + a * stats.gradient_norm)

        # Loss delta
        self._loss_window.append(stats.loss)
        if len(self._loss_window) > 100:
            self._loss_window = self._loss_window[-100:]
        if len(self._loss_window) >= 10:
            recent = sum(self._loss_window[-10:]) / 10
            older = sum(self._loss_window[-20:-10]) / 10 if len(self._loss_window) >= 20 else recent
            self._ema_stats.loss_delta = recent - older

    def _evaluate_transition(self) -> TransitionType:
        """Evalúa si debe transicionar basado en el estado actual."""
        phase = self.current_phase

        # Respetar min_steps
        if self._steps_in_phase < phase.min_steps:
            return TransitionType.STAY

        # Forzar advance si max_steps alcanzado
        if phase.max_steps > 0 and self._steps_in_phase >= phase.max_steps:
            if self.next_phase is not None:
                return TransitionType.ADVANCE

        stats = self._ema_stats

        # Condición de RETREAT: modelo luchando demasiado
        if (self.prev_phase is not None
                and stats.p_full > phase.retreat_p_full_threshold):
            return TransitionType.RETREAT

        # Condición de ADVANCE: modelo en zona de confort
        if self.next_phase is not None:
            # Criterio 1: p_fast alta → datos son demasiado fáciles
            if stats.p_fast > phase.advance_p_fast_threshold:
                return TransitionType.ADVANCE

            # Criterio 2: loss plateau → no hay más que aprender
            if (phase.advance_loss_plateau > 0
                    and abs(stats.loss_delta) < phase.advance_loss_plateau
                    and len(self._loss_window) >= 20):
                return TransitionType.ADVANCE

        return TransitionType.NONE

    def _advance_phase(self):
        """Avanza a la siguiente fase."""
        if self._current_phase_idx >= len(self.phases) - 1:
            return

        old_phase = self.current_phase.name
        self._current_phase_idx += 1
        self._steps_in_phase = 0
        self._blend_factor = 1.0  # transición suave desde la fase anterior

        self._log.append({
            'step': self._total_steps,
            'transition': 'advance',
            'from': old_phase,
            'to': self.current_phase.name,
            'trigger': {
                'p_fast': self._ema_stats.p_fast,
                'p_full': self._ema_stats.p_full,
                'loss_delta': self._ema_stats.loss_delta,
            },
        })

    def _retreat_phase(self):
        """Retrocede a la fase anterior."""
        if self._current_phase_idx <= 0:
            return

        old_phase = self.current_phase.name
        self._current_phase_idx -= 1
        self._steps_in_phase = 0
        self._blend_factor = 0.0  # retroceso inmediato (hay que aliviar el modelo)

        self._log.append({
            'step': self._total_steps,
            'transition': 'retreat',
            'from': old_phase,
            'to': self.current_phase.name,
            'trigger': {
                'p_fast': self._ema_stats.p_fast,
                'p_full': self._ema_stats.p_full,
                'loss_delta': self._ema_stats.loss_delta,
            },
        })

    def get_log(self) -> list:
        """Historial de transiciones."""
        return self._log.copy()

    def get_status(self) -> dict:
        """Estado actual del curriculum."""
        return {
            'phase': self.current_phase.name,
            'phase_idx': self._current_phase_idx,
            'steps_in_phase': self._steps_in_phase,
            'total_steps': self._total_steps,
            'blend_factor': self._blend_factor,
            'ema_p_fast': self._ema_stats.p_fast,
            'ema_p_hybrid': self._ema_stats.p_hybrid,
            'ema_p_full': self._ema_stats.p_full,
            'ema_loss': self._ema_stats.loss,
            'ema_loss_delta': self._ema_stats.loss_delta,
            'transitions': len(self._log),
        }

    def save_state(self) -> dict:
        """Serializa el estado completo del curriculum para checkpoint."""
        return {
            'phase_idx': self._current_phase_idx,
            'steps_in_phase': self._steps_in_phase,
            'total_steps': self._total_steps,
            'blend_factor': self._blend_factor,
            'ema_stats': {
                'p_fast': self._ema_stats.p_fast,
                'p_hybrid': self._ema_stats.p_hybrid,
                'p_full': self._ema_stats.p_full,
                'loss': self._ema_stats.loss,
                'loss_delta': self._ema_stats.loss_delta,
                'gradient_norm': self._ema_stats.gradient_norm,
            },
            'loss_window': self._loss_window[-100:],
            'log': self._log,
        }

    def load_state(self, state: dict):
        """Restaura el estado del curriculum desde checkpoint."""
        self._current_phase_idx = state.get('phase_idx', 0)
        self._steps_in_phase = state.get('steps_in_phase', 0)
        self._total_steps = state.get('total_steps', 0)
        self._blend_factor = state.get('blend_factor', 0.0)
        if 'ema_stats' in state:
            for k, v in state['ema_stats'].items():
                setattr(self._ema_stats, k, v)
        self._loss_window = state.get('loss_window', [])
        self._log = state.get('log', [])


# ─────────────────────────────────────────────────────────────────────────────
# Helper: extraer routing stats del modelo
# ─────────────────────────────────────────────────────────────────────────────

def extract_routing_stats(
    model,
    aux_list: list = None,
    loss: float = 0.0,
    step: int = 0,
) -> RoutingStats:
    """
    Extrae RoutingStats del modelo y/o aux_list del forward.

    Args:
        model: ChimeraLM o stack con layers
        aux_list: lista de aux_dicts del forward (si disponible)
        loss: loss actual
        step: paso de entrenamiento

    Returns:
        RoutingStats
    """
    stats = RoutingStats(step=step, loss=loss)

    # Método 1: desde aux_list (más preciso — datos del batch actual)
    if aux_list:
        probs = []
        for aux in aux_list:
            if aux and 'routing_probs' in aux:
                probs.append(aux['routing_probs'])    # [B, 3]
        if probs:
            import torch
            avg_probs = torch.cat(probs, dim=0).mean(dim=0)   # [3]
            stats.p_fast = avg_probs[0].item()
            stats.p_hybrid = avg_probs[1].item()
            stats.p_full = avg_probs[2].item()
            return stats

    # Método 2: desde EMA del modelo (más estable — historia acumulada)
    stack = getattr(model, 'stack', None)
    if stack is not None:
        n = 0
        total_fast = 0.0
        for layer in stack.layers:
            if hasattr(layer, 'fast_prob_ema'):
                total_fast += layer.fast_prob_ema.item()
                n += 1
        if n > 0:
            stats.p_fast = total_fast / n
            # Estimar hybrid/full desde p_fast (heurística)
            remaining = 1.0 - stats.p_fast
            stats.p_hybrid = remaining * 0.65
            stats.p_full = remaining * 0.35

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== AdaptiveCurriculum smoke test ===\n")

    curriculum = AdaptiveCurriculum()

    # Simular progresión de entrenamiento
    print(f"  Phases: {[p.name for p in curriculum.phases]}")
    print(f"  Initial phase: {curriculum.current_phase.name}\n")

    # Fase 1: Foundation — datos fáciles, p_fast alta
    print("  --- Simulating Foundation phase ---")
    for step in range(1500):
        stats = RoutingStats(
            p_fast=0.6 + 0.002 * step / 1500,   # crece de 0.6 a 0.8
            p_hybrid=0.25,
            p_full=0.15 - 0.002 * step / 1500,
            loss=8.0 - 4.0 * step / 1500,
            step=step,
        )
        curriculum.update(stats)

        if step % 500 == 0:
            mix = curriculum.get_mixture(step, stats)
            status = curriculum.get_status()
            print(f"    Step {step}: phase={status['phase']}, "
                  f"p_fast={status['ema_p_fast']:.3f}, "
                  f"mix={mix.weights}")

    # Fase 2: Intermediate — datos medios
    print("\n  --- Simulating Intermediate phase ---")
    for step in range(1500, 4000):
        stats = RoutingStats(
            p_fast=0.55 + 0.001 * (step - 1500) / 2500,
            p_hybrid=0.30,
            p_full=0.15,
            loss=4.0 - 1.0 * (step - 1500) / 2500,
            step=step,
        )
        curriculum.update(stats)

        if step % 1000 == 0:
            mix = curriculum.get_mixture(step, stats)
            status = curriculum.get_status()
            print(f"    Step {step}: phase={status['phase']}, "
                  f"p_fast={status['ema_p_fast']:.3f}, "
                  f"mix={mix.weights}")

    # Mostrar transiciones
    print(f"\n  Transition log: {curriculum.get_log()}")

    # Test save/load
    saved = curriculum.save_state()
    new_curriculum = AdaptiveCurriculum()
    new_curriculum.load_state(saved)
    assert new_curriculum.current_phase.name == curriculum.current_phase.name
    print(f"\n  Save/load: OK (phase={new_curriculum.current_phase.name})")

    print("\n  [OK] AdaptiveCurriculum funcional.")
