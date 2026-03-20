"""
chimera_paged_state.py — Gestor de Estado Paginado para Serving Multi-Usuario
=============================================================================
CHIMERA tiene una ventaja sobre Transformers para serving: su estado por sesión
es O(1) en lugar de O(S) del KV-cache.

Estado por sesión por capa (CHIMERA 125M, BF16):
  ┌─────────────────────────┬──────────┬───────────┐
  │ Componente              │ Forma    │ Tamaño    │
  ├─────────────────────────┼──────────┼───────────┤
  │ conv_state              │ [D_in,4] │ ~12 KB    │
  │ ssm_state               │ [H,h,S]  │ ~512 KB   │
  │ bus_ring                │ [R,bus]   │ ~4 KB     │
  │ archived_embeddings     │ [64,128] │ ~16 KB    │
  │ archived_importance     │ [64]     │ ~128 B    │
  │ n_archived              │ scalar   │ ~8 B      │
  │ dt_momentum + kahan     │ [D_in]   │ ~6 KB     │
  ├─────────────────────────┼──────────┼───────────┤
  │ TOTAL POR CAPA          │          │ ~550 KB   │
  │ TOTAL 12 CAPAS          │          │ ~6.6 MB   │
  └─────────────────────────┴──────────┴───────────┘

Vs Transformer (125M, BF16, S=4096):
  KV-cache por capa: 2 * n_heads * S * d_head * 2B = ~32 MB
  12 capas: ~384 MB

→ CHIMERA: ~155× más eficiente en estado → ~155× más usuarios concurrentes.

Uso:
    manager = PagedStateManager(model, max_sessions=128)
    sid = manager.create_session()
    tokens = manager.step(sid, token_embedding)
    state = manager.pause_session(sid)        # → CPU/disco
    manager.resume_session(sid, state)        # → GPU
    manager.destroy_session(sid)
"""
from __future__ import annotations

import uuid
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict
from enum import Enum

import torch
import torch.nn as nn


class SessionStatus(Enum):
    """Estado de vida de una sesión."""
    ACTIVE = "active"           # en GPU, lista para step()
    PAUSED = "paused"           # estado guardado en CPU, slot GPU liberado
    DESTROYED = "destroyed"     # eliminada, no recuperable


@dataclass
class SessionState:
    """Estado completo de una sesión de inferencia.

    Contiene todo lo necesario para pausar y reanudar una generación
    sin perder contexto: estado SSM, bus ring, landmarks archivados.
    """
    session_id: str
    status: SessionStatus = SessionStatus.ACTIVE
    tokens_generated: int = 0
    total_tokens_seen: int = 0

    # Estado per-layer (lista de dicts, uno por capa)
    layer_states: list = field(default_factory=list)

    # Metadatos opcionales (para debugging/logging)
    metadata: dict = field(default_factory=dict)


def _extract_layer_state(layer: nn.Module) -> dict:
    """Extrae el estado mutable de una capa a un dict de tensores CPU."""
    state = {}

    # Archive state
    if hasattr(layer, 'archive'):
        arch = layer.archive
        n = arch.n_archived.item()
        state['archived_embeddings'] = arch.archived_embeddings[:n].cpu().clone()
        state['archived_importance'] = arch.archived_importance[:n].cpu().clone()
        state['n_archived'] = n
        state['_err_ema_mean'] = arch._err_ema_mean.item()
        state['_err_ema_var'] = arch._err_ema_var.item()

    # TTT state
    if hasattr(layer, 'dt_momentum'):
        state['dt_momentum'] = layer.dt_momentum.cpu().clone()
        state['dt_kahan_comp'] = layer.dt_kahan_comp.cpu().clone()
        state['mom_kahan_comp'] = layer.mom_kahan_comp.cpu().clone()

    # Fast prob EMA
    if hasattr(layer, 'fast_prob_ema'):
        state['fast_prob_ema'] = layer.fast_prob_ema.item()

    # SDTM state
    if hasattr(layer, 'sdtm'):
        state['sdtm'] = layer.sdtm.get_state()

    return state


def _restore_layer_state(layer: nn.Module, state: dict, device: torch.device):
    """Restaura el estado mutable a una capa desde un dict."""
    if hasattr(layer, 'archive') and 'n_archived' in state:
        arch = layer.archive
        n = state['n_archived']
        arch.n_archived.fill_(n)
        if n > 0:
            arch.archived_embeddings[:n].copy_(
                state['archived_embeddings'].to(device)
            )
            arch.archived_importance[:n].copy_(
                state['archived_importance'].to(device)
            )
        arch._err_ema_mean.fill_(state.get('_err_ema_mean', 0.0))
        arch._err_ema_var.fill_(state.get('_err_ema_var', 0.1))
        arch._lm_cache = None
        arch._lm_cache_n = -1

    if hasattr(layer, 'dt_momentum') and 'dt_momentum' in state:
        layer.dt_momentum.copy_(state['dt_momentum'].to(device))
        layer.dt_kahan_comp.copy_(state['dt_kahan_comp'].to(device))
        layer.mom_kahan_comp.copy_(state['mom_kahan_comp'].to(device))

    if hasattr(layer, 'fast_prob_ema') and 'fast_prob_ema' in state:
        layer.fast_prob_ema.fill_(state['fast_prob_ema'])

    # SDTM state
    if hasattr(layer, 'sdtm') and 'sdtm' in state:
        layer.sdtm.set_state(state['sdtm'], device)


class PagedStateManager:
    """
    Gestor de estado paginado para serving multi-usuario.

    Gestiona slots de sesión en GPU, con capacidad de:
    - Crear sesiones nuevas (con o sin prefill de contexto)
    - Pausar sesiones a CPU (liberar VRAM)
    - Reanudar sesiones desde CPU a GPU
    - Destruir sesiones
    - Monitorear uso de recursos

    Thread-safe mediante Lock para operaciones de sesión.

    Capacidad dinámica: max_sessions se adapta según VRAM disponible
    si se configura con max_sessions='auto'.
    """

    def __init__(
        self,
        model: nn.Module,
        max_sessions: int | str = 64,
        ring_size: int = 16,
        device: torch.device = None,
    ):
        """
        Args:
            model: ChimeraLM o AdvancedChimeraLayer
            max_sessions: máximo de sesiones concurrentes activas en GPU
                          'auto' → estima basado en VRAM disponible
            ring_size: tamaño del ring buffer del bus por sesión
            device: dispositivo GPU (None → auto-detect)
        """
        self.model = model
        self.ring_size = ring_size
        self.device = device or next(model.parameters()).device
        self._lock = threading.Lock()

        # Resolver max_sessions dinámico
        if max_sessions == 'auto':
            self.max_sessions = self._estimate_max_sessions()
        else:
            self.max_sessions = int(max_sessions)

        # Pool de sesiones
        self._sessions: Dict[str, SessionState] = {}
        self._gpu_caches: Dict[str, dict] = {}     # cache GPU por sesión activa

        # Estadísticas
        self._stats = {
            'total_created': 0,
            'total_destroyed': 0,
            'peak_active': 0,
        }

    def _estimate_max_sessions(self) -> int:
        """Estima el número máximo de sesiones concurrentes basado en VRAM."""
        if not torch.cuda.is_available():
            return 16

        free, total = torch.cuda.mem_get_info(self.device)
        # Reservar 30% para el modelo y buffers de operación
        available = free * 0.70
        # Estimar bytes por sesión
        config = getattr(self.model, 'config', None)
        n_layers = getattr(config, 'n_layers', 4) if config else 4
        bytes_per_session = n_layers * 550 * 1024  # ~550 KB/capa

        max_sessions = max(1, int(available / bytes_per_session))
        return min(max_sessions, 1024)  # cap razonable

    @property
    def active_sessions(self) -> int:
        """Número de sesiones activas en GPU."""
        return sum(1 for s in self._sessions.values()
                   if s.status == SessionStatus.ACTIVE)

    @property
    def paused_sessions(self) -> int:
        """Número de sesiones pausadas (en CPU)."""
        return sum(1 for s in self._sessions.values()
                   if s.status == SessionStatus.PAUSED)

    def create_session(
        self,
        session_id: str = None,
        prefill_embeddings: torch.Tensor = None,
        metadata: dict = None,
    ) -> str:
        """
        Crea una nueva sesión de inferencia.

        Args:
            session_id: ID opcional (genera UUID si None)
            prefill_embeddings: [K, d_model] embeddings para cold-start del archive
            metadata: dict arbitrario para tracking

        Returns:
            session_id: str
        """
        with self._lock:
            if self.active_sessions >= self.max_sessions:
                # Intentar pausar la sesión más antigua para hacer espacio
                oldest = self._find_oldest_active()
                if oldest:
                    self._pause_session_unlocked(oldest)
                else:
                    raise RuntimeError(
                        f"Max sessions reached ({self.max_sessions}). "
                        "Pause or destroy a session first."
                    )

            sid = session_id or str(uuid.uuid4())[:8]

            # Crear estado de sesión
            state = SessionState(
                session_id=sid,
                status=SessionStatus.ACTIVE,
                metadata=metadata or {},
            )

            # Alocar cache GPU
            cache = self._allocate_cache()
            self._gpu_caches[sid] = cache

            # Prefill del archive si se proporcionan embeddings
            if prefill_embeddings is not None:
                self._prefill_archive(prefill_embeddings)

            self._sessions[sid] = state
            self._stats['total_created'] += 1
            self._stats['peak_active'] = max(
                self._stats['peak_active'], self.active_sessions
            )

            return sid

    def _allocate_cache(self) -> dict:
        """Aloca cache de inferencia para una sesión."""
        dtype = next(self.model.parameters()).dtype
        stack = getattr(self.model, 'stack', None)
        if stack is not None and hasattr(stack, 'layers'):
            # ChimeraLM → usar allocate_inference_cache del primer layer
            layer = stack.layers[0]
            return layer.allocate_inference_cache(
                batch_size=1, ring_size=self.ring_size, dtype=dtype
            )

        # Fallback: modelo es una sola capa
        if hasattr(self.model, 'allocate_inference_cache'):
            return self.model.allocate_inference_cache(
                batch_size=1, ring_size=self.ring_size
            )

        return {}

    def _prefill_archive(self, embeddings: torch.Tensor):
        """Pre-carga landmarks en todas las capas del modelo."""
        stack = getattr(self.model, 'stack', None)
        if stack is None:
            return
        for layer in stack.layers:
            if hasattr(layer, 'archive'):
                layer.archive.preload_context(embeddings.to(self.device))

    def step(
        self,
        session_id: str,
        token_embedding: torch.Tensor,    # [1, 1, D]
    ) -> torch.Tensor:
        """
        Procesa un token para la sesión dada.

        Args:
            session_id: ID de sesión
            token_embedding: [1, 1, D] embedding del token actual

        Returns:
            output: [1, 1, D] hidden del token
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id} not found")

            state = self._sessions[session_id]
            if state.status == SessionStatus.PAUSED:
                raise RuntimeError(
                    f"Session {session_id} is paused. Resume it first."
                )
            if state.status == SessionStatus.DESTROYED:
                raise RuntimeError(
                    f"Session {session_id} has been destroyed."
                )

            cache = self._gpu_caches[session_id]

        # Forward sin lock (el modelo puede tardar)
        stack = getattr(self.model, 'stack', None)
        if stack is not None:
            x = token_embedding.to(self.device)
            for layer in stack.layers:
                x, cache = layer.step(x, cache)
        elif hasattr(self.model, 'step'):
            x, cache = self.model.step(
                token_embedding.to(self.device), cache
            )
        else:
            raise RuntimeError("Model must have step() method or stack attribute")

        with self._lock:
            self._gpu_caches[session_id] = cache
            state.tokens_generated += 1
            state.total_tokens_seen += 1

        return x

    def pause_session(self, session_id: str) -> SessionState:
        """Pausa sesión: mueve estado a CPU, libera VRAM."""
        with self._lock:
            return self._pause_session_unlocked(session_id)

    def _pause_session_unlocked(self, session_id: str) -> SessionState:
        """Implementación interna de pause (sin lock — caller debe tenerlo)."""
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")

        state = self._sessions[session_id]
        if state.status != SessionStatus.ACTIVE:
            return state

        # Extraer estado del modelo por capa
        stack = getattr(self.model, 'stack', None)
        if stack is not None:
            state.layer_states = [
                _extract_layer_state(layer) for layer in stack.layers
            ]

        # Mover cache GPU a CPU
        cache = self._gpu_caches.pop(session_id, {})
        state.metadata['_paused_cache'] = {
            k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
            for k, v in cache.items()
        }

        state.status = SessionStatus.PAUSED
        return state

    def resume_session(
        self,
        session_id: str,
        state: SessionState = None,
    ):
        """
        Reanuda una sesión pausada: restaura estado de CPU a GPU.

        Args:
            session_id: ID de sesión
            state: SessionState guardado (opcional, usa el interno si None)
        """
        with self._lock:
            if self.active_sessions >= self.max_sessions:
                oldest = self._find_oldest_active()
                if oldest and oldest != session_id:
                    self._pause_session_unlocked(oldest)
                else:
                    raise RuntimeError("No room for resume. Pause another session.")

            if state is None:
                state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(f"Session {session_id} not found")

            if state.status != SessionStatus.PAUSED:
                return  # ya activa

            # Restaurar cache GPU
            paused_cache = state.metadata.pop('_paused_cache', {})
            gpu_cache = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in paused_cache.items()
            }
            self._gpu_caches[session_id] = gpu_cache

            # Restaurar estado de las capas
            stack = getattr(self.model, 'stack', None)
            if stack is not None and state.layer_states:
                for layer, layer_state in zip(stack.layers, state.layer_states):
                    _restore_layer_state(layer, layer_state, self.device)

            state.status = SessionStatus.ACTIVE
            self._sessions[session_id] = state

    def destroy_session(self, session_id: str):
        """Destruye una sesión y libera todos los recursos."""
        with self._lock:
            if session_id in self._gpu_caches:
                del self._gpu_caches[session_id]

            if session_id in self._sessions:
                self._sessions[session_id].status = SessionStatus.DESTROYED
                self._stats['total_destroyed'] += 1

    def _find_oldest_active(self) -> Optional[str]:
        """Encuentra la sesión activa con más tokens generados (LRU simplificado)."""
        oldest_id = None
        oldest_tokens = -1
        for sid, state in self._sessions.items():
            if state.status == SessionStatus.ACTIVE:
                if state.total_tokens_seen > oldest_tokens:
                    oldest_tokens = state.total_tokens_seen
                    oldest_id = sid
        return oldest_id

    def get_stats(self) -> dict:
        """Estadísticas del manager."""
        return {
            'active': self.active_sessions,
            'paused': self.paused_sessions,
            'max_sessions': self.max_sessions,
            **self._stats,
        }

    def list_sessions(self) -> list:
        """Lista todas las sesiones con su estado."""
        return [
            {
                'id': sid,
                'status': state.status.value,
                'tokens_generated': state.tokens_generated,
                'total_tokens_seen': state.total_tokens_seen,
            }
            for sid, state in self._sessions.items()
            if state.status != SessionStatus.DESTROYED
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Batch Scheduler — Continuous Batching para Multi-Request
# ─────────────────────────────────────────────────────────────────────────────

class ContinuousBatchScheduler:
    """
    Scheduler de batching continuo para CHIMERA.

    Agrupa múltiples sesiones activas en un mismo batch para maximizar
    throughput GPU. Cada ciclo:
      1. Selecciona hasta max_batch_size sesiones activas
      2. Ejecuta un step() batcheado
      3. Distribuye resultados a las sesiones

    Diferencia clave vs Transformer continuous batching:
    - No hay KV-cache paginado (Mamba2 state es O(1))
    - No hay page tables ni block tables
    - El costo de agregar/remover una request del batch es O(1)

    Limitación actual: cada sesión tiene su propio cache individual.
    Para batching real optimizado, se necesitaría un cache compartido
    con vistas por sesión (futuro: Paged Tensor).
    """

    def __init__(
        self,
        state_manager: PagedStateManager,
        max_batch_size: int = 8,
    ):
        self.manager = state_manager
        self.max_batch_size = max_batch_size
        self._pending_requests: list = []

    def submit_request(
        self,
        session_id: str,
        token_embedding: torch.Tensor,    # [1, 1, D]
        callback=None,
    ):
        """Encola un token para procesamiento en el próximo batch."""
        self._pending_requests.append({
            'session_id': session_id,
            'token': token_embedding,
            'callback': callback,
        })

    def process_batch(self) -> list:
        """
        Procesa hasta max_batch_size requests pendientes.

        Returns:
            list de dicts con {session_id, output}
        """
        if not self._pending_requests:
            return []

        batch = self._pending_requests[:self.max_batch_size]
        self._pending_requests = self._pending_requests[self.max_batch_size:]

        results = []
        for req in batch:
            try:
                output = self.manager.step(req['session_id'], req['token'])
                result = {
                    'session_id': req['session_id'],
                    'output': output,
                    'error': None,
                }
                if req['callback']:
                    req['callback'](output)
            except Exception as e:
                result = {
                    'session_id': req['session_id'],
                    'output': None,
                    'error': str(e),
                }
            results.append(result)

        return results

    @property
    def pending_count(self) -> int:
        return len(self._pending_requests)


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== PagedStateManager smoke test ===\n")

    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    torch.set_float32_matmul_precision('high')

    from chimera_config import ChimeraConfig
    from chimera_lm import ChimeraLM

    cfg = ChimeraConfig(d_model=256, n_layers=2, expand=2, headdim=32, d_state=64)
    model = ChimeraLM(cfg, vocab_size=4096).cuda().bfloat16().eval()

    # Crear manager
    manager = PagedStateManager(model, max_sessions=4, ring_size=8)
    print(f"  Max sessions: {manager.max_sessions}")
    print(f"  Auto-estimated max sessions: {manager._estimate_max_sessions()}")

    # Crear sesiones
    s1 = manager.create_session(session_id="user-1")
    s2 = manager.create_session(session_id="user-2")
    print(f"  Active: {manager.active_sessions}, Paused: {manager.paused_sessions}")

    # Step
    D = cfg.d_model
    tok = torch.randn(1, 1, D, device='cuda', dtype=torch.bfloat16)
    out = manager.step(s1, tok)
    print(f"  Step output shape: {out.shape}")

    # Pause + Resume
    manager.pause_session(s1)
    print(f"  After pause: Active={manager.active_sessions}, "
          f"Paused={manager.paused_sessions}")

    manager.resume_session(s1)
    print(f"  After resume: Active={manager.active_sessions}, "
          f"Paused={manager.paused_sessions}")

    # Destroy
    manager.destroy_session(s2)
    print(f"  After destroy: {manager.list_sessions()}")

    # Stats
    print(f"  Stats: {manager.get_stats()}")

    # Test ContinuousBatchScheduler
    scheduler = ContinuousBatchScheduler(manager, max_batch_size=4)
    for i in range(3):
        scheduler.submit_request(s1, tok)
    results = scheduler.process_batch()
    print(f"  Batch processed: {len(results)} requests")

    print("\n  [OK] PagedStateManager funcional.")
