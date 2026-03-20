import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import contextlib
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# TF32 global: en Ampere (RTX 30xx/40xx) las operaciones FP32 pueden usar
# Tensor Float32 internamente, lo que da ~2-3× speedup en matmuls con
# pérdida de precisión insignificante (10 bits mantisa en vez de 23).
torch.set_float32_matmul_precision('high')


@contextlib.contextmanager
def _ttt_dt_bias_override(mamba2_module, temp_bias: torch.Tensor):
    """
    Context manager para TTT: sustituye temporalmente dt_bias por un tensor
    FP32 con requires_grad=True, ejecuta el forward, y restaura el original.

    Yields el nn.Parameter temporal para que autograd pueda trazarlo:

        with _ttt_dt_bias_override(m.mamba2, dt_bias_adapt) as dt_param:
            out  = m.mamba2(mini_chunk)
            loss = ...
            grad = torch.autograd.grad(loss, dt_param)[0]   # correcto

    Por qué yield (no return):
      nn.Parameter(temp_bias) crea un NUEVO objeto en el grafo autograd.
      Si se usa `torch.autograd.grad(loss, temp_bias)` en lugar de `dt_param`,
      PyTorch no encontrará `temp_bias` en el grafo (distinto objeto) y lanzará
      RuntimeError. El yield expone el objeto exacto que participa en el grafo.

    Hilo-seguro: `finally` garantiza restauración incluso ante excepciones.
    Compatible con DDP: el parámetro original nunca se elimina del state_dict.
    """
    original  = mamba2_module.dt_bias          # nn.Parameter original
    temp_param = nn.Parameter(temp_bias, requires_grad=temp_bias.requires_grad)
    try:
        mamba2_module.dt_bias = temp_param     # nn.Module.__setattr__ → registro oficial
        yield temp_param                       # caller usa ESTE objeto para autograd.grad
    finally:
        mamba2_module.dt_bias = original       # siempre restaura


from sgr_slr import SLRDifferentialModule
from landmark_native import NativeLandmarkArchive
from spectral_vsa_archive_v2 import SpectralVSAArchive
from ttt_kernel import lion_constrained_update_inplace, compute_token_errors_triton, spsa_mse_fused, spsa_factored_forward
from cas_swarm import ChimeraAutonomousSwarm
from gpu_profile import get_gpu_profile as _get_gpu_profile

_GPU_PROF = _get_gpu_profile()

class GatedComplexityPredictor(nn.Module):
    def __init__(self, d_model: int, n_tiers: int = 3,
                 min_prob_floor: float = 0.05):
        """
        Router con robustez anti-colapso + attentive pooling.

        FIX V6: x.mean(dim=1) destruía toda la información posicional y
        secuencial del chunk. Un solo token complejo escondido entre tokens
        simples era invisible al router → asignación ciega de tier.

        Ahora: cada token recibe un peso de importancia aprendido via
        pool_gate (Linear D→1 + sigmoid). El pooling es la suma ponderada
        normalizada:  x_pooled = Σ(gate_t · x_t) / Σ(gate_t)
        Si hay un token extremadamente complejo, pool_gate aprenderá a
        darle peso alto → domina la representación de routing.

        Overhead: +d_model parámetros (~256). Esencialmente gratis.

        Args:
            min_prob_floor: probabilidad mínima garantizada por tier (default=5%).
        """
        super().__init__()
        d_hidden = 32
        self.min_prob_floor = min_prob_floor
        self.n_tiers        = n_tiers
        # Attentive pooling: per-token importance gate
        self.pool_gate = nn.Linear(d_model, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, n_tiers)
        )
        # Temperatura aprendible: controla la nitidez del routing.
        # log_temp=0 → T=1 (softmax normal).
        # El scheduler puede recocerla externamente (annealing).
        # init log_temp=0.5 → T≈1.65 (soft al inicio, aprende a discriminar).
        self.log_temp = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor):
        # Attentive pooling: per-token importance weighting.
        # pool_gate: [B, S, 1] — sigmoid para importancia ∈ (0, 1).
        # Tokens complejos (alta norma, patrones raros) reciben mayor peso.
        gate = torch.sigmoid(self.pool_gate(x))       # [B, S, 1]
        x_pooled = (x * gate).sum(dim=1) / (gate.sum(dim=1) + 1e-6)  # [B, D]
        logits  = self.mlp(x_pooled)                  # [B, n_tiers]

        # Temperatura: T>1 → routing suave (más exploración al inicio).
        # T se aprende por gradiente; clamp evita T>10 o T<0.1.
        T     = torch.exp(self.log_temp).clamp(0.1, 10.0)
        probs = torch.softmax(logits / T, dim=-1)    # [B, n_tiers]

        # Anti-collapse floor: garantiza mínimo min_prob_floor por tier.
        # Equivalente a mezclar la salida del router con una distribución uniforme
        # con peso floor*n_tiers, pero más eficiente computacionalmente.
        if self.min_prob_floor > 0:
            floor = self.min_prob_floor
            probs = probs.clamp(min=floor)
            probs = probs / probs.sum(dim=-1, keepdim=True)   # re-normalizar

        return probs, logits


class _GradThrottleFn(torch.autograd.Function):
    """
    Gradient throttle — compile-safe via torch.library.custom_op.
    Identidad en forward, rescala gradiente en backward si norma > max_norm.
    """
    @staticmethod
    def forward(ctx, x, max_norm: float):
        ctx.max_norm = max_norm
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        g_norm = g.norm()
        scale  = (ctx.max_norm / (g_norm + 1e-8)).clamp(max=1.0)
        return g * scale, None


# Registrar como custom_op para que torch.compile no genere graph break
@torch.library.custom_op("chimera::grad_throttle", mutates_args=())
def _grad_throttle_op(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    return x.clone()

@_grad_throttle_op.register_fake
def _grad_throttle_fake(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    return torch.empty_like(x)

def _grad_throttle_setup(ctx, inputs, output):
    _, max_norm = inputs
    ctx.max_norm = max_norm

def _grad_throttle_bwd(ctx, g):
    g_norm = g.norm()
    scale  = (ctx.max_norm / (g_norm + 1e-8)).clamp(max=1.0)
    return g * scale, None

_grad_throttle_op.register_autograd(
    _grad_throttle_bwd, setup_context=_grad_throttle_setup,
)


def _grad_throttle(x: torch.Tensor, max_norm: float = 3.0) -> torch.Tensor:
    """Aplica throttle de gradiente en backward; identity en forward."""
    return _grad_throttle_op(x, max_norm)


class BusParallelConfig:
    """
    Config de paralelismo runtime para AsyncLightBus.

    tp_group : ProcessGroup Tensor Parallel — dimensión del modelo D sharded.
               Cada rank tiene x[:, :, D // tp_size] + pesos Column/Row-Parallel.
    sp_group : ProcessGroup Sequence Parallel — secuencia S sharded.
               Cada rank tiene x[:, S // sp_size, :] con D completa.

    Soporta las 4 combinaciones: none · TP-only · SP-only · TP+SP (3D).

    Los process groups NO se guardan en state_dict (no son tensores).
    Llamar layer.bus.set_parallel_config() una vez por proceso tras cargar
    cada checkpoint.

    Análisis de comunicación por capa:
      SP-only :  2 all-reduce × O(B·D)       — num y den del pool
      TP-only :  2 all-reduce × O(B·bus_dim) — gate_logit + publish
      TP+SP   :  3 all-reduce                — gate(TP) + num/den(SP) + publish(TP)
    En NVLink 900 GB/s con B=4, bus_dim=128: < 5 µs/layer — negligible vs scan.
    """
    __slots__ = ('tp_group', 'sp_group')

    def __init__(self, tp_group=None, sp_group=None):
        self.tp_group = tp_group
        self.sp_group = sp_group

    def _grp_size(self, grp) -> int:
        if grp is None:
            return 1
        try:
            import torch.distributed as dist
            return dist.get_world_size(grp)
        except Exception:
            return 1

    @property
    def tp_size(self) -> int:  return self._grp_size(self.tp_group)
    @property
    def sp_size(self) -> int:  return self._grp_size(self.sp_group)
    @property
    def is_tp(self) -> bool:   return self.tp_group is not None and self.tp_size > 1
    @property
    def is_sp(self) -> bool:   return self.sp_group is not None and self.sp_size > 1
    @property
    def is_distributed(self) -> bool: return self.is_tp or self.is_sp


class AsyncLightBus(nn.Module):
    """
    AsyncLightBus from OrthoSSM plan.
    Provides a fast cross-layer communication channel.
    """
    def __init__(self, d_model: int, bus_dim: int = 128):
        super().__init__()
        self.bus_dim = bus_dim
        self.publish = nn.Linear(d_model, bus_dim, bias=False)
        self.gather_q = nn.Linear(d_model, bus_dim, bias=False)
        self.modulate = nn.Linear(bus_dim, d_model, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))
        # V7: Attentive pooling para summary del bus — mismo fix que el router.
        # V8: Distribuido — pool_gate actúa como column-parallel en TP mode:
        #     gate_logit = all_reduce(pool_gate_partial) → sigmoid.
        # Overhead: +d_model parámetros (~256), una matmul [B,S,1] por capa.
        self.pool_gate = nn.Linear(d_model, 1, bias=False)
        # Paralelismo distribuido. No serializable → llamar set_parallel_config()
        # tras cada checkpoint.load(). Inmutable después de set (thread-safe).
        self._par_cfg: 'BusParallelConfig' = BusParallelConfig()

    # ── Distributed API ───────────────────────────────────────────────────────

    def set_parallel_config(self, tp_group=None, sp_group=None) -> None:
        """
        Configura paralelismo TP/SP para este bus. Llamar ANTES de entrenar:

            for layer in model.layers:
                layer.bus.set_parallel_config(tp_group=tp_pg, sp_group=sp_pg)

        tp_group : TP ProcessGroup — D sharded; all-reduces en gate/publish/q.
        sp_group : SP ProcessGroup — S sharded; all-reduces de num/den del pool.
        Crea un nuevo BusParallelConfig atómico → thread-safe.
        """
        self._par_cfg = BusParallelConfig(tp_group=tp_group, sp_group=sp_group)

    def _pool_begin(self, x: torch.Tensor) -> tuple:
        """
        Fase 1 del pooling distribuido con solapamiento de comunicación-cómputo.

        Lanza el all_reduce del gate_logit de forma ASÍNCRONA (async_op=True)
        y retorna el handle de trabajo inmediatamente. El caller puede hacer
        cómputo local (ej. _q_project) mientras la red transfiere los datos.

        Call contract:
            gl, work = self._pool_begin(x)
            q = self._q_project(x)           # ← solapado con el all_reduce
            summary = self._pool_end(gl, work, x)

        Overhead en single-device: zero — work=None, pool_gate(x) es un matmul
        normal sin comunicación. El overhead de async_op=True sobre blocking
        all_reduce en NVLink es < 2μs (deferred enqueue vs enqueue+wait).
        """
        cfg = self._par_cfg
        gl  = self.pool_gate(x)          # [B, S_local, 1]
        if cfg.is_tp:
            import torch.distributed as dist
            work = dist.all_reduce(gl, op=dist.ReduceOp.SUM,
                                   group=cfg.tp_group, async_op=True)
        else:
            work = None
        return gl, work

    def _pool_end(self, gl: torch.Tensor, work_gate, x: torch.Tensor) -> torch.Tensor:
        """
        Fase 2 del pooling distribuido.

        Espera el all_reduce del gate (work_gate.wait()) y completa:
          Paso B — pool ponderado con SP all_reduce async de num/den.
          Paso C — publish con TP all_reduce blocking (último paso, nada que solapar).

        El wait() ocurre justo antes de torch.sigmoid(gl), garantizando que gl
        esté completamente reducido entre todos los TP ranks antes de usarse.
        """
        cfg = self._par_cfg

        # Completar gate all_reduce si está en vuelo
        if work_gate is not None:
            work_gate.wait()

        if not cfg.is_distributed:
            # ── Fast path: single device — cero all-reduces ───────────────────
            gate = torch.sigmoid(gl)
            xp   = (x * gate).sum(1) / (gate.sum(1) + 1e-6)
            return F.normalize(self.publish(xp), p=2, dim=-1)

        import torch.distributed as dist

        gate = torch.sigmoid(gl)                             # [B, S_local, 1] — reduced

        # ── Paso B: pool ponderado ────────────────────────────────────────────
        num = (x * gate).sum(1)                              # [B, D_local]
        den = gate.sum(1)                                    # [B, 1]
        if cfg.is_sp:
            num, den = num.contiguous(), den.contiguous()
            w_n = dist.all_reduce(num, op=dist.ReduceOp.SUM,
                                  group=cfg.sp_group, async_op=True)
            w_d = dist.all_reduce(den, op=dist.ReduceOp.SUM,
                                  group=cfg.sp_group, async_op=True)
            w_n.wait(); w_d.wait()
        xp = num / (den + 1e-6)                              # [B, D_local] — S sumada

        # ── Paso C: publish ────────────────────────────────────────────────
        s = self.publish(xp)                                 # [B, bus_dim] parcial en TP
        if cfg.is_tp:
            dist.all_reduce(s, op=dist.ReduceOp.SUM, group=cfg.tp_group)
        return F.normalize(s, p=2, dim=-1)                   # [B, bus_dim] — idéntico

    def _attentive_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Wrapper secuencial (sin solapamiento). Usar _pool_begin/_pool_end
        directamente en forward_ring para obtener compute-comm overlap."""
        gl, work = self._pool_begin(x)
        return self._pool_end(gl, work, x)

    def _q_project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Proyección gather_q con TP all-reduce.

        SP : no all-reduce — cada rank atiende su shard de secuencia contra
             bus_ring replicado → la atención local es semánticamente correcta.
        TP : gather_q es column-parallel [D/tp → bus_dim].
             all_reduce SUM reconstruye q completo para atender bus_ring.
        """
        q = self.gather_q(x)                                # [B, S_local, bus_dim]
        if self._par_cfg.is_tp:
            import torch.distributed as dist
            dist.all_reduce(q, op=dist.ReduceOp.SUM, group=self._par_cfg.tp_group)
        return q

    # ── Forwards ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, bus_cache: torch.Tensor = None):
        """
        x: [B, S, D]
        bus_cache: [B, num_layers_so_far, bus_dim] (Optional)
        Returns:
            x_out: modulated x [B, S, D]
            new_cache: updated bus cache [B, num_layers_so_far + 1, bus_dim]
        """
        B, S, D = x.shape
        
        # 1. Publish summary — attentive pooling con all-reduces TP/SP si aplica.
        summary            = self._attentive_pool(x)       # [B, bus_dim] — consistente
        summary_unsqueezed = summary.unsqueeze(1)          # [B, 1, bus_dim]
        
        # 2. Gather from previous layers if cache exists
        if bus_cache is None or bus_cache.shape[1] == 0:
            # No hay contexto previo: auto-modulación de resguardo.
            # Garantiza que gate y modulate siempre reciban gradientes,
            # incluso en la primera capa o cuando bus_cache está vacío.
            trivial_gathered = summary.unsqueeze(1).expand(-1, S, -1)  # [B, S, bus_dim]
            trivial_mod_out = self.modulate(trivial_gathered)
            trivial_mod = trivial_mod_out * torch.sigmoid(self.gate).to(trivial_mod_out.dtype)
            return x + trivial_mod, summary_unsqueezed
            
        q = self._q_project(x)                                  # [B, S, bus_dim] — TP-aware

        # Incluir el summary DE ESTA CAPA en el pool de atención.
        # Razón: garantiza que publish.weight reciba gradiente en cualquier
        # posición del stack (incluida la última capa). También es semánticamente
        # válido: la capa puede atender a su propia compresión actual (auto-bus).
        # augmented_cache: [B, L+1, bus_dim]  donde [0] = summary actual
        augmented_cache = torch.cat([summary_unsqueezed, bus_cache], dim=1)
        
        # attention: [B, S, bus_dim] x [B, bus_dim, L+1] -> [B, S, L+1]
        scores = torch.bmm(q, augmented_cache.transpose(1, 2)) / math.sqrt(self.bus_dim)
        attn = F.softmax(scores, dim=-1)
        
        # gathered: [B, S, L+1] x [B, L+1, bus_dim] -> [B, S, bus_dim]
        gathered = torch.bmm(attn, augmented_cache)
        
        # 3. Modulate
        modulate_out_fw = self.modulate(gathered)
        modulation = modulate_out_fw * torch.sigmoid(self.gate).to(modulate_out_fw.dtype)
        
        # 4. Update cache: orden cronológico (capa 0, 1, 2, ...)
        new_cache = torch.cat([bus_cache, summary_unsqueezed], dim=1)
        
        return x + modulation, new_cache

    def forward_ring(
        self,
        x: torch.Tensor,
        bus_ring: torch.Tensor,
        write_idx: int,
    ) -> tuple:
        """
        Training-time forward con ring buffer de forma EST\u00c1TICA — CUDA Graph safe.

        Ventaja vs forward() original:
          - forward() original crecía via torch.cat: [B,1,D], [B,2,D]...
            formas diferentes en cada capa → imposible para CUDA Graphs / compile static.
          - forward_ring(): bus_ring SIEMPRE tiene forma [B, N_layers, bus_dim].
            write_idx es un int Python → baked como constante en tiempo de trace.
            El clone() preserva la semántica de gradient checkpoint:
            la capa i recibe el bus tal como estaba ANTES de que ella escribiera.

        Causal mask:
          - Slots 0..write_idx son efectivos (capas ya ejecutadas + esta).
          - Slots write_idx+1..N-1 reciben -inf → softmax los zeroes.
          - El mask es un tensor Python-computed baked como constante en el graph.

        x:         [B, S, D]
        bus_ring:  [B, N_layers, bus_dim]  — zeros para slots no escritos aún
        write_idx: int — índice de esta capa (layer._layer_idx)
        Returns:   (x_modulated [B, S, D], new_ring [B, N_layers, bus_dim])
        """
        B, S, D = x.shape
        N = bus_ring.shape[1]

        # V8.1 Compute-Comm Overlap:
        # Fase 1: lanzar gate all_reduce async (TP). La red empieza a summar
        # los gate_logits mientras esta capa sigue computando localmente.
        gl, work_gate = self._pool_begin(x)

        # OVERLAP: _q_project corre con los Tensor Cores mientras el all_reduce
        # TP del gate viaja por NVLink. En NVLink 900 GB/s con B=4, bus_dim=128
        # el reduce tarda ~3μs → la matmul [B,S,D]@[D,bus_dim] puede esconder
        # todo ese tiempo. En single-device: work_gate=None, cero overhead.
        q      = self._q_project(x)                                        # [B, S, bus_dim] — TP-aware

        # Fase 2: esperar a gate (si había reduce en vuelo) y finalizar pool.
        summary = self._pool_end(gl, work_gate, x)                         # [B, bus_dim]

        # Clone el ring para que el estado "antes de esta capa" quede preservado
        # para gradient checkpointing (la copia conserva el grafo de autograd).
        new_ring = bus_ring.clone()
        new_ring[:, write_idx, :] = summary      # escribe en slot fijo (op CUDA in-place)

        scores = torch.bmm(q, new_ring.transpose(1, 2)) / math.sqrt(self.bus_dim)  # [B, S, N]

        # Causal mask: Python int → baked como constante en CUDA Graph trace
        if write_idx < N - 1:
            causal_mask = scores.new_full((1, 1, N), 0.0)
            causal_mask[:, :, write_idx + 1:] = float('-inf')
            scores = scores + causal_mask

        attn     = F.softmax(scores, dim=-1)                               # [B, S, N]
        gathered = torch.bmm(attn, new_ring)                               # [B, S, bus_dim]

        modulate_out = self.modulate(gathered)                             # [B, S, D]
        gate_sig = torch.sigmoid(self.gate).to(modulate_out.dtype)
        modulation = modulate_out * gate_sig                               # [B, S, D]
        return x + modulation, new_ring

    def reset_cache(self):
        """Resetea el gate del bus a 0 para mitigar drift acumulado."""
        with torch.no_grad():
            self.gate.zero_()

    def step_ring(self, x: torch.Tensor, bus_ring: torch.Tensor) -> tuple:
        """
        CUDA-graph-safe decode step con ring buffer pre-alocado.

        Reemplaza forward() durante decode token-by-token. bus_ring tiene forma
        fija [B, ring_size, bus_dim], no crece con torch.cat → CUDA Graph safe.

        Algoritmo ring:
          1. torch.roll(bus_ring, -1, dims=1): evicta token más antiguo (pos 0),
             desplaza todos una ranura a la izquierda.
          2. Escribe summary nuevo en última ranura [:, -1, :] (slice estático).
          3. Cross-attention sobre TODAS las ranuras (forma fija en cada step).

        Invariantes CUDA Graph:
          - torch.roll:          forma siempre constante, sin Python branching.
          - new_ring[:, -1, :]:  slice estático, no depende de números en GPU.
          - torch.bmm:           formas fijas [B,1,bus_dim] x [B,bus_dim,ring_size].
          - F.softmax:           100% graph-safe.

        x:        [B, 1, D]              — token actual
        bus_ring: [B, ring_size, bus_dim] — ring buffer pre-alocado
        Returns: (x_out [B, 1, D], new_ring [B, ring_size, bus_dim])
        """
        # Decode single-token: S=1, pool trivial → bypass pool_gate.
        # TP: publish es column-parallel → all_reduce SUM reconstruye summary.
        # SP durante decode: tokengen es inherentemente secuencial; no SP aquí.
        x_sq   = x.squeeze(1)                                          # [B, D_local]
        s_raw  = self.publish(x_sq)                                    # [B, bus_dim] parcial en TP
        if self._par_cfg.is_tp:
            import torch.distributed as dist
            dist.all_reduce(s_raw, op=dist.ReduceOp.SUM,
                            group=self._par_cfg.tp_group)
        summary = F.normalize(s_raw, p=2, dim=-1)                      # [B, bus_dim]

        # Roll: desplaza izquierda (evicta oldest), mantiene forma exacta
        new_ring = torch.roll(bus_ring, -1, dims=1).clone()             # [B, ring_size, bus_dim]
        new_ring[:, -1, :] = summary                                    # escribe en ranura fija

        # Cross-attention de x contra todos los slots del ring
        q      = self._q_project(x)                                     # [B, 1, bus_dim] — TP-aware
        scores = torch.bmm(q, new_ring.transpose(1, 2)) / math.sqrt(self.bus_dim)  # [B,1,ring_size]
        attn   = F.softmax(scores, dim=-1)                              # [B, 1, ring_size]
        gathered     = torch.bmm(attn, new_ring)                        # [B, 1, bus_dim]
        modulate_out = self.modulate(gathered)                           # [B, 1, D]
        gate_sig     = torch.sigmoid(self.gate).to(modulate_out.dtype)
        modulation   = modulate_out * gate_sig                           # [B, 1, D]

        return x + modulation, new_ring


# ─────────────────────────────────────────────────────────────────────────────
# PagedBusCache — gestión dinámica de memoria para inferencia a gran escala
# ─────────────────────────────────────────────────────────────────────────────

class PagedBusCache:
    """
    Pool de páginas físicas para el bus_ring de decode en producción.

    PROBLEMA que resuelve:
        allocate_inference_cache() reserva bus_ring=[B, ring_size, bus_dim]
        estáticamente POR SECUENCIA. En un servidor atendiendo N secuencias
        concurrentes con tamaños heterogéneos, esto produce fragmentación:
          - Secuencia corta de 128 tokens reserva 16 páginas × bus_dim → desperdicio.
          - Secuencia larga de 8192 tokens necesita más páginas de las disponibles.
        La fragmentación reduce el throughput efectivo en ~30-50% vs el óptimo.

    SOLUCIÓN — Physical Page Pool:
        Un bloque contiguo de memoria [total_pages, page_size, bus_dim] se divide
        en páginas físicas uniformes. Cada secuencia mantiene una "tabla de páginas"
        (lista de índices físicos) que puede crecer dinámicamente. La memoria se
        asigna y libera en O(1) vía free_list.

        Uso típico:
            pool = PagedBusCache(total_pages=512, page_size=16,
                                 bus_dim=128, device='cuda')
            # Al inicio de cada secuencia:
            pool.alloc_seq(seq_id=42, n_pages=8)
            # Durante decode:
            view = pool.get_view(seq_id=42)     # [n_pages*page_size, bus_dim]
            # Al terminar:
            pool.free_seq(seq_id=42)

    COMPATIBILIDAD:
        allocate_inference_cache() acepta paged_pool=PagedBusCache como kwarg.
        step_ring() detecta automáticamente si el cache contiene una vista paginada.
        El API legacy (bus_ring tensor estático) sigue funcionando sin cambios.

    VENTAJAS sobre cache estático:
        • Fragmentación cero: páginas libres se reutilizan inmediatamente.
        • Crecimiento dinámico: append_page(seq_id) añade páginas sin realocar.
        • Batching eficiente: secuencias de tamaños diferentes comparten el pool.
        • Eviction sencillo: free_seq() marca páginas como disponibles en O(len).
    """

    def __init__(
        self,
        total_pages: int,
        page_size:   int,
        bus_dim:     int,
        device:      torch.device | str = 'cuda',
        dtype:       torch.dtype = torch.float32,
    ):
        self.page_size    = page_size
        self.bus_dim      = bus_dim
        self.total_pages  = total_pages
        # free_list: índices de páginas libres (LIFO = mejor cache locality)
        self.free_list: list[int] = list(range(total_pages))
        # seq_tables: seq_id → lista de page_ids asignados en orden lógico
        self.seq_tables: dict[int, list[int]] = {}
        # Posición de escritura circular por secuencia (dentro del ring lógico)
        self.write_ptrs:  dict[int, int]       = {}
        # ── Flat contiguous cache ─────────────────────────────────────────────
        # Para cada seq_id, mantenemos un buffer flat [n_pages*page_size, bus_dim]
        # CONTIGUO en memoria. write_slot() actualiza este buffer en paralelo con
        # el pool físico. get_view() devuelve una slice real (sin alloc, sin copia).
        # Overhead: alloc de n_pages*page_size*bus_dim*dtype bytes en alloc_seq().
        # Equivale exactamente al output del torch.cat que se elimina — sin coste extra.
        self._flat_cache: dict[int, torch.Tensor] = {}
        self._device = torch.device(device) if isinstance(device, str) else device
        self._dtype  = dtype

    # ── Gestión de secuencias ─────────────────────────────────────────────────

    def alloc_seq(self, seq_id: int, n_pages: int) -> None:
        """
        Aloca n_pages páginas para seq_id.
        Levanta RuntimeError si no hay páginas libres suficientes.
        """
        if seq_id in self.seq_tables:
            raise ValueError(f"seq_id={seq_id} ya está asignado; llamar free_seq() primero.")
        if len(self.free_list) < n_pages:
            raise RuntimeError(
                f"PagedBusCache: sin páginas libres "
                f"({len(self.free_list)} disponibles, {n_pages} requeridas)."
            )
        pages = [self.free_list.pop() for _ in range(n_pages)]
        self.seq_tables[seq_id]  = pages
        self.write_ptrs[seq_id]  = 0
        # Alocar buffer flat contiguo — get_view() devolverá una slice de este buffer
        self._flat_cache[seq_id] = torch.zeros(
            n_pages * self.page_size, self.bus_dim,
            device=self._device, dtype=self._dtype,
        )

    def append_page(self, seq_id: int) -> None:
        """Añade una página adicional a una secuencia existente (crecimiento dinámico)."""
        if not self.free_list:
            raise RuntimeError("PagedBusCache: pool agotado, no hay páginas adicionales.")
        pid = self.free_list.pop()
        self.seq_tables[seq_id].append(pid)
        # Extender el buffer flat: crear uno nuevo de tamaño actualizado y copiar
        # los datos anteriores. Esto ocurre raramente (solo en crecimiento dinámico).
        old_buf = self._flat_cache[seq_id]
        n_new   = len(self.seq_tables[seq_id]) * self.page_size
        new_buf = torch.zeros(n_new, self.bus_dim, device=self._device, dtype=self._dtype)
        new_buf[:old_buf.shape[0]].copy_(old_buf)
        self._flat_cache[seq_id] = new_buf

    def free_seq(self, seq_id: int) -> None:
        """Libera todas las páginas de seq_id y las devuelve al free_list."""
        if seq_id not in self.seq_tables:
            return
        self.free_list.extend(self.seq_tables.pop(seq_id))
        self.write_ptrs.pop(seq_id, None)
        self._flat_cache.pop(seq_id, None)   # liberar buffer flat también

    def get_view(self, seq_id: int) -> torch.Tensor:
        """
        Retorna una vista CONTIGUA [n_pages * page_size, bus_dim] de la
        memoria de seq_id. Vista REAL (sin copia ni alloc) del buffer flat
        mantenido en sincronía con el pool físico via write_slot().
        Para lectura en atención cruzada (ring-buffer virtual).
        """
        return self._flat_cache[seq_id]

    def write_slot(self, seq_id: int, summary: torch.Tensor) -> None:
        """
        Escribe summary [bus_dim] en la posición ring-buffer de seq_id.
        Actualiza AMBOS: el pool físico y el buffer flat (manteniendo la
        invariante que get_view() devuelve datos actualizados sin copia).
        Avanza el puntero circular; añade página si se necesita espacio.

        summary: [bus_dim] — resumen de la capa/token actual.
        """
        ptr   = self.write_ptrs[seq_id]
        pages = self.seq_tables[seq_id]
        total_slots = len(pages) * self.page_size
        # Si llenamos el ring, intentar crecer (si pool tiene espacio libre)
        if ptr >= total_slots:
            if self.free_list:
                self.append_page(seq_id)          # actualiza _flat_cache también
                total_slots = len(self.seq_tables[seq_id]) * self.page_size
            else:
                ptr = ptr % total_slots           # ring wrap
        page_idx = ptr // self.page_size
        slot_idx = ptr %  self.page_size
        self._flat_cache[seq_id][ptr]            = summary    # ptr antes del wrap
        self.write_ptrs[seq_id] = ptr + 1

    @property
    def free_pages(self) -> int:
        return len(self.free_list)

    @property
    def used_pages(self) -> int:
        return self.total_pages - len(self.free_list)


# ─────────────────────────────────────────────────────────────────────────────
# ChimeraMoEFFN — Sparse FFN para capacidad paramétrica eficiente
# ─────────────────────────────────────────────────────────────────────────────

class ChimeraMoEFFN(nn.Module):
    """
    Sparse Top-K Mixture-of-Experts FFN.

    MOTIVACIÓN (Gemini rec. §2.2):
      Con el router de tier (FAST/HYBRID/FULL) controlamos la PROFUNDIDAD
      del cómputo. ChimeraMoEFFN controla la RUTA PARAMÉTRICA: el mismo
      FLOPs-budget, pero el modelo puede especializarse en n_experts aspectos
      distintos del lenguaje (morfología, semántica, código, matemáticas…).

    DISEÑO:
      n_experts=8, top_k=2, d_ff=d_model*2 por experto.
      FLOPs/token = top_k × (D×d_ff + d_ff×D) = 2 × 2D² = 4D²
      vs dense FFN: 8D² (2× MENOS FLOPs que un FFN estándar de D→4D→D)
      Parámetros totales: n_experts × 2 × D × d_ff = 8 × 2 × D × 2D = 32D²
      vs dense FFN: 8D² (4× MÁS parámetros — la "inteligencia" de 3B)

    DISPATCH bajo-nivel:
      - expert_ids ordenados → matmul grupal por experto (no Python loop con .any())
      - Fallback a einsum para graph_mode / torch.compile (sin Python-if sobre tensor)
      - Load-balance auxiliary loss (Switch Transformer eq. 4) para evitar
        colapso donde un solo experto recibe todos los tokens.

    INTEGRACIÓN con la arquitectura Chimera:
      - Se aplica a `out` DESPUÉS del gated-mix y ANTES del bus.
      - En graph_mode (CUDA Graph capture): bypass completo vía torch.cond
        para evitar Python-if sobre tensores y mantener static shapes.
      - En decode (step()): bypass automático (S=1, MoE no vale la pena).
      - scale inicializado en -4.0 → sigmoid(-4) ≈ 0.018:
        contribución casi nula al inicio, crece gradualmente con entrenamiento.
    """

    def __init__(self, d_model: int, n_experts: int = 8,
                 top_k: int = 2, d_ff: int | None = None):
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = min(top_k, n_experts)
        self.d_ff      = d_ff or d_model * 2
        # Router ligero (sin bias para invarianza translacional)
        self.gate      = nn.Linear(d_model, n_experts, bias=False)
        # n_experts FFNs de dos capas con activación SiLU
        self.experts   = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, self.d_ff, bias=False),
                nn.SiLU(),
                nn.Linear(self.d_ff, d_model, bias=False),
            ) for _ in range(n_experts)
        ])
        # Escala de salida: init near-zero → contribución nula al inicio,
        # gradiente fluye desde el primer paso, aprende gradualmente.
        self.scale     = nn.Parameter(torch.full((1,), -4.0))
        # Normalización de entrada para estabilizar el routing
        self.norm      = nn.RMSNorm(d_model)

        # Init: el gate empieza con pesos pequeños → distribución uniforme inicial
        nn.init.normal_(self.gate.weight, std=0.02)

    def forward(self, x: torch.Tensor, graph_mode: bool = False) -> tuple:
        """
        x: [B, S, D]
        Returns: (out [B, S, D], lb_loss scalar)

        lb_loss: auxiliar de load-balance que el trainer debe añadir al loss.

        DISPATCH VECTORIZADO — cero CPU sync, cero argsort, cero split:
          1. Apilar pesos de todos los expertos en [E, d_ff, D] y [E, D, d_ff].
             torch.stack sobre los parámetros existentes: O(E) Python overhead,
             crea vistas concatenadas en GPU sin datos extra en el host.
          2. Para cada slot top-k:  W1_tok = W1[e_ids]  →  [T, d_ff, D]
             → torch.bmm(xf.unsqueeze(1), W1_tok.T) → [T, d_ff]
             → SiLU → torch.bmm → [T, D]
          Ventaja vs dispatch con argsort+split: elimina 1 CPU sync (cnts.tolist)
          por slot, reduce lanzamientos de kernel, preserva orden de tokens y es
          plenamente diferenciable a través de W1, W2 y gate.
        """
        B, S, D = x.shape
        T = B * S
        xn = self.norm(x)                                  # normalizar antes de routing
        xf = xn.reshape(T, D)

        logits  = self.gate(xf)                            # [T, E]
        prob    = logits.softmax(dim=-1)                   # [T, E]
        topk_w, topk_e = prob.topk(self.top_k, dim=-1)    # [T, K], [T, K]
        topk_w  = topk_w / topk_w.sum(-1, keepdim=True).clamp(min=1e-6)

        # ── Dispatch por slot top-k sin copiar pesos (No memory bloat, No CPU syncs)
        # Iteramos por experto para aplicar las FFNs a los tokens asignados,
        # sin construir buffers de peso dinámicos gigantes [T, d_ff, D].
        out = torch.zeros_like(xf)
        for ki in range(self.top_k):
            # Procesamos cada experto por separado.
            # Para tokens vacíos, PyTorch maneja BATCH_SIZE=0 silenciosa y velozmente 
            # sin necesidad de .numel() ni device-to-host syncs.
            for e in range(self.n_experts):
                mask_e = (topk_e[:, ki] == e)
                tokens_e = mask_e.nonzero(as_tuple=True)[0]
                
                # Ejecutar subgraph para los tokens
                xf_e = xf[tokens_e]                          # [N_e, D]
                h = F.linear(xf_e, self.experts[e][0].weight) # [N_e, d_ff]
                h = F.silu(h)
                o = F.linear(h, self.experts[e][2].weight)    # [N_e, D]
                
                # Ponderar y acumular
                w_e = topk_w[tokens_e, ki].unsqueeze(-1)
                out.index_add_(0, tokens_e, o * w_e)

        # ── Load-balance loss (Switch Transformer eq. 4) ─────────────────────
        # f_e: fracción de tokens que eligen experto e como top-1 (detached).
        # P_e: probabilidad media del router para experto e (diferenciable).
        # bincount no requiere CPU sync — devuelve tensor GPU.
        f_e     = topk_e[:, 0].bincount(minlength=self.n_experts).float() / T
        lb_loss = self.n_experts * (f_e.detach() * prob.mean(0)).sum()

        scale = torch.sigmoid(self.scale)                  # ∈ (0,1), crece desde ~0
        return x + scale * out.reshape(B, S, D), lb_loss


class AdvancedChimeraLayer(nn.Module):
    def __init__(self, d_model: int = 256, expand: int = 2, headdim: int = 32,
                 layer_idx: int = 0, d_state: int = 64,
                 use_spectral_vsa: bool = False,
                 spectral_K: int = 32, spectral_K_min: int = 4,
                 spectral_window: int = 256,
                 spectral_ema_alpha: float = 0.9, spectral_use_complex: bool = True,
                 spectral_n_retrieve: int = 8,
                 spectral_energy_threshold: float = 0.95,
                 spectral_lanczos_power_max: float = 4.0,
                 spectral_disc_gamma: float = 3.0,
                 spectral_error_refresh: float = 0.5,
                 use_moe: bool = False,
                 moe_n_experts: int = 8,
                 moe_top_k: int = 2,
                 moe_d_ff: int | None = None,
                 use_cas: bool = False,
                 cas_n_experts: int = 8,
                 cas_d_ff: int | None = None,
                 cas_tau_init: float = 0.3,
                 cas_target_active: float = 0.25,
                 n_layers_total: int = 4):
        super().__init__()
        self.d_model = d_model
        self._layer_idx = layer_idx
        
        # We will dynamically import mamba_ssm here to ensure it's available at runtime
        # avoiding issues if the module is still compiling globally in another process
        from mamba_ssm import Mamba2
        
        # Base Mamba 2 official — layer_idx requerido para inference_params (decode/chunked)
        self.mamba2 = Mamba2(
            d_model=d_model,
            expand=expand,
            headdim=headdim,
            d_state=d_state,
            layer_idx=layer_idx,
        )
        
        # 1. Multi-scale A init — geometric de λ_0=exp(-0.001) a λ escaladas por potencias de 2.
        # Clamp a 1e-15 para evitar domain error cuando 2**i es muy grande (modelos ≥125M).
        n_heads = self.mamba2.nheads
        lambdas = [max(math.exp(-0.001 * (2**min(i, 20))), 1e-15) for i in range(n_heads)]
        A_init = [-math.log(l) for l in lambdas]
        A_log_tensor = torch.tensor([math.log(a) for a in A_init], dtype=torch.float32)
        if self.mamba2.A_log.shape == (n_heads,):
            self.mamba2.A_log = nn.Parameter(A_log_tensor)
        else:
            self.mamba2.A_log = nn.Parameter(A_log_tensor.view(-1, 1))
            
        # OPT-3: Atención Híbrida (Sliding Window / Global)
        # Se añade en capas específicas (ej. cada 4 capas) para resolver el "needle in a haystack" >32K
        self.use_hybrid_attn = (layer_idx % 4 == 3)
        if self.use_hybrid_attn:
            # SDPA-based O(N) Hybrid Attention
            self.num_heads = d_model // headdim
            self.headdim = headdim
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.o_proj = nn.Linear(d_model, d_model, bias=False)
            self.hybrid_norm = nn.RMSNorm(d_model)


        # 2. Router
        self.router = GatedComplexityPredictor(d_model, 3)
        self.norm = nn.RMSNorm(d_model)

        # 3. SLR Differential Module (SGR top-12.5% + Triton diff-attn)
        self.slr = SLRDifferentialModule(
            d_model=d_model,
            d_head=headdim,
            window_size=64,
            top_k_frac=0.125
        )
        # 5. AsyncLightBus
        self.bus = AsyncLightBus(d_model, bus_dim=128)

        # 6. Context archive — NativeLandmarkArchive (legacy) o SpectralVSAArchive (ChebyHolo)
        self.use_spectral_vsa = use_spectral_vsa
        if use_spectral_vsa:
            self.archive = SpectralVSAArchive(
                d_model=d_model,
                K=spectral_K,
                K_min=spectral_K_min,
                window_size=spectral_window,
                ema_alpha=spectral_ema_alpha,
                use_complex_roles=spectral_use_complex,
                n_retrieve_bands=spectral_n_retrieve,
                energy_threshold=spectral_energy_threshold,
                lanczos_power_max=spectral_lanczos_power_max,
                disc_gamma=spectral_disc_gamma,
                error_refresh_ratio=spectral_error_refresh,
                use_learned_gate=_GPU_PROF.spsa_fused,  # H200: learned Chebyshev truncation
            )
        else:
            self.archive = NativeLandmarkArchive(
                d_model=d_model,
                landmark_dim=128,
                max_landmarks=64,
                ttt_err_threshold=0.3,
            )

        # 4. TTT-Lite: adapta dt_bias vía Lion (Plan §2.2.2)
        # dt_momentum siempre FP32; register_buffer → manda a GPU con .cuda(),
        # aparece en state_dict, nunca se pierde en CUDA Graph capture.
        self.register_buffer(
            'dt_momentum',
            torch.zeros(self.mamba2.dt_bias.numel(), dtype=torch.float32)
        )
        # Buffers de compensación Kahan — mantienen los bits perdidos en cada
        # acumulación de EMA y dt_bias. Inicializados en 0 (correcto para Kahan).
        # Error: O(ε·N) sin Kahan vs O(ε) con Kahan (N=pasos de entrenamiento).
        self.register_buffer(
            'dt_kahan_comp',
            torch.zeros(self.mamba2.dt_bias.numel(), dtype=torch.float32)
        )
        self.register_buffer(
            'mom_kahan_comp',
            torch.zeros(self.mamba2.dt_bias.numel(), dtype=torch.float32)
        )
        self.ttt_lr   = 1e-3
        self.ttt_beta = 0.9
        # TTT cold-start warmup: durante los primeros pasos, Mamba2 emite
        # representaciones aleatorias → TTT adapta dt_bias sobre basura.
        # Escalar ttt_lr desde ~0 hasta 1.0× evita que Lion empuje dt_bias
        # a estados irrecuperables. Acoplado al progreso de training, no al LM loss
        # (que aún no es confiable en los primeros pasos).
        self.ttt_warmup_steps = 500     # pasos lineales de warmup
        self.register_buffer(
            '_ttt_step',
            torch.tensor(0, dtype=torch.long)
        )
        # graph_mode = True → forward 100% graph-capturable:
        #   - sin .item(), sin Python-if sobre tensores
        #   - TTT se ejecuta fuera del grafo vía update_ttt_inplace()
        self.graph_mode = False
        # EMA de prob_FAST para detección de colapso del router.
        # Si fast_prob_ema > 0.92 durante logging, el trainer debe aumentar
        # supervision_weight o reiniciar el router.
        self.register_buffer(
            'fast_prob_ema',
            torch.tensor(0.333)   # init: distribución uniforme entre 3 tiers
        )
        # Umbrales dinámicos como buffers — accesibles por ChimeraAnnealer.
        # init = 0.75: SLR raramente activa al inicio (calentamiento rápido).
        # El annealer los reduce gradualmente: slr 0.75→0.50, arch 0.60→0.40.
        # En forward se modulan adicionalmente por batch_var_bonus.
        self.register_buffer('slr_threshold',  torch.tensor(0.75))
        self.register_buffer('arch_threshold', torch.tensor(0.60))

        # 4b. TTT-Full low-rank (Plan §2.2.2 TTT-FULL)
        #     W_adapt = ttt_U @ ttt_V  (rank=4)  — overhead <0.3%
        #     Corrección al scan output: out' = out + scale * (out @ Vᵀ @ Uᵀ)
        #     U: [D, rank]  V: [rank, D]
        ttt_rank = 4
        self.ttt_rank = ttt_rank
        #
        # INIT ROBUSTO contra saddle point:
        #   Si U=0 OR V=0 → h = out@V.T = 0 → correction = 0 → grad_U = h.T@g = 0
        #   → AMBOS quedan muertos aunque solo UNO sea cero.
        #   Solución: Xavier uniform para U (proyección de salida) y
        #   kaiming uniform para V (proyección de entrada).
        #   Esto garantiza varianza de activación razonable desde el paso 0.
        self.ttt_U = nn.Parameter(torch.empty(d_model, ttt_rank))
        self.ttt_V = nn.Parameter(torch.empty(ttt_rank, d_model))
        nn.init.xavier_uniform_(self.ttt_U)     # escala ∞(2/(D+rank))
        nn.init.kaiming_uniform_(self.ttt_V, a=math.sqrt(5))  # escala ∞(1/rank)
        # ttt_full_scale = -2.0 → sigmoid(-2) ≈ 0.12 (vs -4.0 → 0.018)
        # Gradiente de scale en init: σ(-2)·(1-σ(-2)) ≈ 0.105 frente a 0.018.
        # El modelo aprende a reducir o ampliar la corrección vía entrenamiento.
        self.ttt_full_scale = nn.Parameter(torch.tensor(-2.0))
        # Buffer para gradiente TTT pendiente (aplicado externamente por el trainer)
        # No register_buffer: es un estado de optimizador, no de modelo.
        self._pending_ttt_grad: torch.Tensor | None = None
        # Flag for gradient checkpoint recomputation — skips TTT/archive side effects
        # to avoid double-mutation when use_reentrant=False re-runs the forward.
        self._skip_side_effects = False

        # 7. Sparse MoE (opcional) — capacidad paramétrica con FLOPs constantes.
        # 7. Sparse MoE (opcional) — capacidad paramétrica con FLOPs constantes.
        # use_moe=False por defecto: cero overhead y retro-compatibilidad total.
        # Activar con use_moe=True para modelos ≥1B donde la capacidad paramétrica
        # es el cuello de botella (no los FLOPs).
        #
        # GUARD DE-CONFLICTO: si use_cas=True, CAS tiene prioridad TOTAL y MoE
        # no se instancia. Instanciar ambos desperdicia VRAM y crea parámetros que
        # nunca reciben gradientes (params muertos → DDP/FSDP warnings + divergencia).
        if use_moe and use_cas:
            import warnings
            warnings.warn(
                f"AdvancedChimeraLayer(layer_idx={layer_idx}): use_moe=True ignorado "
                f"porque use_cas=True tiene prioridad. MoE no se instancia para evitar "
                f"parámetros muertos ({moe_n_experts} expertos × ~{(moe_d_ff or d_model*2)*d_model*2//1000}K params).",
                UserWarning, stacklevel=2,
            )
            use_moe = False
        self.use_moe = use_moe
        if use_moe:
            self.moe = ChimeraMoEFFN(
                d_model,
                n_experts=moe_n_experts,
                top_k=moe_top_k,
                d_ff=moe_d_ff,
            )

        # 8. Chimera Autonomous Swarm (CAS) — AoE × LExI × TTT-Coupled.
        # Reemplaza funcionalmente al MoE cuando use_cas=True (ver guard arriba).
        # CAS es incompatible con graph_mode (dispatch dinámico interno).
        self.use_cas = use_cas
        if use_cas:
            self.cas = ChimeraAutonomousSwarm(
                d_model=d_model,
                n_experts=cas_n_experts,
                d_ff=cas_d_ff,
                layer_idx=layer_idx,
                n_layers=n_layers_total,
                tau_init=cas_tau_init,
                target_active=cas_target_active,
            )

    # ── API para CUDA Graph y trainer explícito ─────────────────────────────────
    @property
    def effective_ttt_lr(self) -> float:
        """TTT learning rate con linear warmup durante los primeros ttt_warmup_steps."""
        step = self._ttt_step.item()
        warmup = self.ttt_warmup_steps
        return self.ttt_lr * min(step / warmup, 1.0) if warmup > 0 else self.ttt_lr

    @torch.no_grad()
    def update_ttt_inplace(self, x_norm_chunk: torch.Tensor | None = None) -> torch.Tensor | None:
        """
        Aplica el Lion update al dt_bias con Kahan Summation.

        Dos modos de uso:
          A) x_norm_chunk=None  → aplica self._pending_ttt_grad (computado en el
             forward del entrenamiento estándar). El trainer llama:
                 loss.backward()
                 optimizer.step()
                 layer.update_ttt_inplace()   # ← modo A
             Esto garantiza que dt_bias NO cambia durante el forward → sin drift.

          B) x_norm_chunk=[B, L, D]  → recomputa el gradiente (para graph_mode,
             tests, o inference TTT). El trainer llama:
                 layer.update_ttt_inplace(x_norm.detach()[:, :64])  # ← modo B

        Returns:
            per_token_err [B, L-1] en modo B; None en modo A.
        """
        if x_norm_chunk is None:
            # ── Modo A: aplica gradiente pendiente del forward ──────────────────
            grad = self._pending_ttt_grad
            if grad is None or self.effective_ttt_lr <= 0.0:
                return None
            self._pending_ttt_grad = None

            dt_bias_fp32 = self.mamba2.dt_bias.detach().float().clone()
            lion_constrained_update_inplace(
                dt_bias_fp32,
                self.dt_momentum,
                grad,
                self.mamba2.A_log.detach().float().view(-1),
                beta=self.ttt_beta,
                lr=self.effective_ttt_lr,       # warmup-scaled
                active_prob=1.0,               # ya se filtró en forward
                mom_comp=self.mom_kahan_comp,
                dt_comp=self.dt_kahan_comp,
            )
            self.mamba2.dt_bias.data.copy_(dt_bias_fp32.to(self.mamba2.dt_bias.dtype))
            return None

        # ── Modo B: recomputa gradiente desde x_norm_chunk ─────────────────────
        B, mini_len, D = x_norm_chunk.shape
        chunk = x_norm_chunk.detach()

        with torch.enable_grad():
            dt_bias_adapt = self.mamba2.dt_bias.detach().clone().requires_grad_(True)
            with _ttt_dt_bias_override(self.mamba2, dt_bias_adapt) as dt_param:
                m_out = self.mamba2(chunk)
            pred   = m_out[:, :-1]
            target = chunk[:, 1:]
            per_token_err = compute_token_errors_triton(pred.detach(), target.detach())
            loss = F.mse_loss(pred.float(), target.float())
            grad = torch.autograd.grad(loss, dt_param)[0].float()  # float32 para Lion

        # TTT complexity is converted to a smooth value roughly [0, 1] based on error magnitude.
        # err > 4.0 ~ 1.0; err=0.0 ~ 0.0
        complexity = (1.0 - torch.exp(-0.25 * per_token_err.mean())).item()

        dt_bias_fp32 = self.mamba2.dt_bias.detach().float().clone()
        lion_constrained_update_inplace(
            dt_bias_fp32,
            self.dt_momentum,
            grad,
            self.mamba2.A_log.detach().float().view(-1),
            beta=self.ttt_beta,
            lr=self.effective_ttt_lr,          # warmup-scaled
            active_prob=complexity,
            mom_comp=self.mom_kahan_comp,
            dt_comp=self.dt_kahan_comp,
        )
        self.mamba2.dt_bias.data.copy_(dt_bias_fp32.to(self.mamba2.dt_bias.dtype))
        return per_token_err  # [B, mini_len-1]
    # ─────────────────────────────────────────────────────────────────────────────

    def reset_doc_state(self) -> None:
        """
        Reset completo de todos los estados que dependen del documento actual.
        Llamar ENTRE documentos en procesamiento secuencial (no-packing):

            for doc in dataset:
                layer.reset_doc_state()
                out, bus = layer(doc_tokens, bus_ring=ring)

        Para sequence packing (múltiples docs en una sola secuencia), usar el
        parámetro `doc_ids` de forward() en lugar de este método.

        Resets incluidos:
          • SpectralVSAArchive / NativeLandmarkArchive: todos los buffers
            espectrales (buf, V_mem, coeff_norms, etc.) → sin landmarks
          • bus.gate → 0.0 para anular modulación residual del doc anterior
          • TTT momenta y compensación Kahan → optimizador limpio
          • dt_momentum → buffer limpio para el próximo documento
        """
        # Archive completo (buf, V_mem, coeff_norms, spinor, etc.)
        if hasattr(self.archive, 'reset'):
            self.archive.reset()
        # Bus: anular la modulación acumulada del doc anterior
        with torch.no_grad():
            self.bus.gate.zero_()
        # TTT: limpiar momenta para que Lion no arrastre gradientes del doc anterior
        with torch.no_grad():
            self.dt_momentum.zero_()
            self.dt_kahan_comp.zero_()
            self.mom_kahan_comp.zero_()
        self._pending_ttt_grad = None

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int = 1,
                                   dtype=None, ring_size: int = None,
                                   paged_pool: 'PagedBusCache | None' = None,
                                   seq_id: int | None = None):
        """
        Aloca los buffers de inferencia autoregresiva (token-by-token).
        Retorna un dict cache compatible con self.step().

        cache = {
          'conv_state':   [B, d_inner+2*ngroups*d_state, d_conv] — rolling buffer
          'ssm_state':    [B, nheads, headdim, d_state]           — SSM hidden
          'bus_ring':     [B, ring_size, bus_dim]  (estático) o tensor paginado
          '_archive_ctx': None — delta de archive pre-computado (llenado por
                                  make_cuda_graph_step() tras el prefill)
          '_paged_pool':  PagedBusCache | None — pool de páginas si paged_pool!=None
          '_paged_seq_id': int | None — seq_id en el pool
        }

        ring_size:
            Tokens recientes que conserva el bus ring durante decode.
            Si None, se toma de gpu_profile.ring_size (adaptativo al hardware).
            RTX 4050 Laptop = 16; A100 = 32; H200 = 32; B200 = 64.

        paged_pool / seq_id:
            Si se pasa un PagedBusCache, el bus_ring se aloca dinámicamente
            en el pool en lugar de un tensor estático. Free con pool.free_seq(seq_id).
        """
        conv_state, ssm_state = self.mamba2.allocate_inference_cache(
            batch_size, max_seqlen=1, dtype=dtype
        )
        if ring_size is None:
            try:
                ring_size = _get_gpu_profile().ring_size
            except Exception:
                ring_size = 16   # fallback conservador

        device    = next(self.parameters()).device
        buf_dtype = dtype if dtype is not None else torch.float32

        if paged_pool is not None:
            # Modo paginado: aloca páginas desde el pool compartido.
            # El número de páginas se calcula para cubrir ring_size slots.
            n_pages = max(1, math.ceil(ring_size / paged_pool.page_size))
            sid = seq_id if seq_id is not None else id(conv_state)  # unique key
            paged_pool.alloc_seq(sid, n_pages)
            bus_ring_static = None
        else:
            bus_ring_static = torch.zeros(
                batch_size, ring_size, self.bus.bus_dim,
                device=device, dtype=buf_dtype,
            )
            sid = None

        return {
            'conv_state':    conv_state,
            'ssm_state':     ssm_state,
            'bus_ring':      bus_ring_static,    # None si paginado
            '_archive_ctx':  None,
            '_paged_pool':   paged_pool,
            '_paged_seq_id': sid,
        }

    def step(self, x: torch.Tensor, cache: dict) -> tuple:
        """
        Single-token decode (inferencia autoregresiva).

        x:     [B, 1, D]  — token actual
        cache: dict de allocate_inference_cache()
        Returns: (out [B,1,D], updated_cache)

        En decode, siempre tier FAST:
          - NO TTT (single token, no mini-chunk disponible)
          - NO SLR (K=12.5% de S=1 = 0 tokens)
          - Bus: inyecta summary de este token al cache acumulado
          - Archive retrieval: activo si hay landmarks acumulados del prefill
        """
        B = x.shape[0]
        x_norm = self.norm(x)              # [B, 1, D]

        # Mamba2 single-step decode (usa conv_state + ssm_state rolling)
        # step() retorna ([B, 1, D], conv_state, ssm_state) — ya incluye dim de secuencia
        mamba_out, cache['conv_state'], cache['ssm_state'] = self.mamba2.step(
            x_norm, cache['conv_state'], cache['ssm_state']
        )                                  # mamba_out: [B, 1, D]

        # TTT-Full decode: aplicar corrección low-rank (parámetros ya aprendidos)
        # No se actualiza U/V en decode — solo aplica la corrección estática
        scale      = torch.sigmoid(self.ttt_full_scale)
        h          = mamba_out @ self.ttt_V.T
        correction = h @ self.ttt_U.T
        mamba_out  = mamba_out + scale * correction

        # Archive retrieval: dos modos.
        #
        # Modo estándar (arc_ctx=None): retrieve() en vivo, incluye CPU sync
        #   (.item() en _get_processed_landmarks) → NO graph-safe pero exacto.
        #
        # Modo graph (arc_ctx=tensor): delta fijo pre-computado antes de capture
        #   por make_cuda_graph_step(). Sin .item(), sin Python branching → safe.
        #   Aprox: delta calculado con query=0; válido para decode estándar.
        arc_ctx = cache.get('_archive_ctx')
        if arc_ctx is not None:
            out = mamba_out + arc_ctx
        else:
            out = self.archive.retrieve(mamba_out)
            out = out + self.archive.get_compress_ctx(mamba_out)

        # Bus: ring buffer de forma fija (CUDA-graph-safe vía torch.roll).
        # Reemplaza bus(out, bus_cache) que crecía con torch.cat → no capturable.
        out, cache['bus_ring'] = self.bus.step_ring(out, cache['bus_ring'])

        # Estabilización de norma para decode multistep.
        # Problema: en un loop autoregresivo (x_tok = out_tok sin embedding table)
        # el residual x + out acumula norma indefinidamente → explosión.
        # Solución: soft-norm que mantiene la magnitud cerca de sqrt(d_model)
        # sin clipear gradientes ni alterar la dirección del vector.
        # En un LM real (x = embedding fresco), este rescale es ~1.0 → no-op.
        expected_norm = self.d_model ** 0.5          # √D ≈ 16 para D=256
        residual      = x + out
        actual_norm   = residual.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        # Escalar solo si la norma supera 4× el valor esperado
        scale_factor  = (expected_norm * 4) / actual_norm
        scale_factor  = scale_factor.clamp(max=1.0)  # solo reduce, nunca amplifica
        return residual * scale_factor, cache

    def forward(self, x, bus_ring=None, bus_cache=None, return_aux: bool = False,
                doc_ids: torch.Tensor | None = None):
        """
        Forward pass para training/prefill.

        Args:
            x:          [B, S, D]
            bus_ring:   [B, N_layers, bus_dim] ring buffer pre-alocado (nuevo API, preferred).
                        Forma ESTÁTICA — compatible con CUDA Graphs y torch.compile.
            bus_cache:  [B, L, bus_dim] cache creciente (API legacy, solo si bus_ring=None).
            return_aux: si True, retorna un dict extra con routing_probs
                        y ttt_pred/target para ChimeraLosses.
            doc_ids:    [B, S] tensor entero con IDs de documento por token.
                        Soporta sequence packing: múltiples documentos en una sola
                        secuencia. La capa zeroa automáticamente la modulación del bus
                        y del Mamba2 en las posiciones de transición entre documentos,
                        previniendo contaminación cruzada (crosstalk).
                        None → sin masking (modo predeterminado para entrenamiento
                        de documentos individuales sin packing).

        Returns:
            (out [B,S,D], new_bus_state)                 — si return_aux=False
            (out [B,S,D], new_bus_state, aux_dict)       — si return_aux=True

            new_bus_state es:
              - new_ring [B, N_layers, bus_dim]  cuando bus_ring!=None (CUDA Graph safe)
              - new_cache [B, L+1, bus_dim]      cuando bus_cache!=None (legacy, torch.cat)
        """
        B, S, D = x.shape
        # ── BF16 / mixed-precision: auto-cast al dtype del modelo ─────────────────
        param_dtype = next(self.parameters()).dtype
        x = x.to(param_dtype)
        x_norm = self.norm(x)

        # ── Document-Aware Masking (sequence packing) ──────────────────────────
        # doc_open_mask [B, S, 1]: 1.0 dentro de un documento, 0.0 en la primera
        # posición de un nuevo documento (transición doc A → doc B).
        # Al multiplicar mamba_out * doc_open_mask zeroeamos la contribución del
        # SSM en esa posición — previene que el estado oculto de A contamine B.
        # NOTA: esta es una aproximación conservadora. El estado interno del SSD
        # scan de Mamba2 aún porta info de A (sin segment_lengths nativo). Para
        # perfecta aislación, el trainer debe llamar reset_doc_state() entre docs.
        doc_open_mask = None
        if doc_ids is not None:
            boundary       = torch.zeros(B, S, dtype=x.dtype, device=x.device)
            boundary[:, 1:] = (doc_ids[:, 1:] != doc_ids[:, :-1]).to(x.dtype)
            doc_open_mask  = (1.0 - boundary).unsqueeze(-1)  # [B, S, 1]

        # Router probabilities + logits (logits para Z-loss)
        probs, router_logits = self.router(x_norm)  # [B, 3] -> 0: FAST, 1: HYBRID, 2: FULL

        # Actualizar EMA de prob_FAST para detección de colapso.
        if self.training:
            with torch.no_grad():
                cur_fast = probs[:, 0].mean()
                self.fast_prob_ema.mul_(0.99).add_(cur_fast * 0.01)

        # Vistas [B, 1, 1] para broadcasting en todas las ecuaciones de mezcla.
        prob_fast   = probs[:, 0].view(B, 1, 1)
        prob_hybrid = probs[:, 1].view(B, 1, 1)
        prob_full   = probs[:, 2].view(B, 1, 1)

        # D5: complexidad per-elemento [B] en vez de un escalar promedio de batch.
        complexity_per_elem = probs[:, 1:].sum(dim=-1)        # [B]  en [0, 2]
        total_active_prob   = complexity_per_elem.max()        # escalar
        total_active_prob_mean = complexity_per_elem.mean()    # para logging / TTT-Lite
        if complexity_per_elem.numel() > 1:
            batch_var_bonus = complexity_per_elem.std(correction=0).detach()
        else:
            batch_var_bonus = complexity_per_elem.new_zeros(())
        slr_threshold  = torch.clamp(self.slr_threshold  - 0.05 * batch_var_bonus, min=0.20)
        arch_threshold = torch.clamp(self.arch_threshold - 0.10 * batch_var_bonus, min=0.30)

        # --- TTT-LITE: adapta dt_bias usando error predictivo por mini-chunk ---
        # ARQUITECTURA ASÍNCRONA (fix drift chunked vs full):
        #   1. Computar gradiente sobre mini-chunk con dt_bias ORIGINAL
        #   2. Correr scan principal con dt_bias ORIGINAL (sin modificar)
        #   3. Aplicar Lion update DESPUÉS del scan → dt_bias para el próximo forward
        #
        # Razón: si el update se aplica antes del scan, el procesamiento chunked
        # (que re-computa el gradiente al inicio de cada chunk) ve un dt_bias
        # diferente al procesamiento full → drift T5 = 106% → eliminado con esto.
        # El modelo aprende igualmente bien: el lag de 1 forward es equivalente
        # al pipeline-parallelism estándar de optimizadores con gradient delay=1.
        ttt_importance = None   # [B, S] — error L2 por token, para SGR
        _ttt_grad_to_apply = None  # gradiente pendiente para post-scan Lion update
        # graph_mode=True: TTT se omite en forward; el trainer llama a
        # update_ttt_inplace() fuera del graph captured region.
        # self.training es Python bool (set por .train()/.eval()) → NO rompe grafo.
        if self.training and not self.graph_mode and not self._skip_side_effects:
            # Increment TTT warmup step counter
            self._ttt_step.add_(1)
            # TTT-Lite V2: SPSA (Simultaneous Perturbation Stochastic Approximation)
            #
            # V1 usaba diferencias finitas centrales coordenada-por-coordenada:
            #   2×n_heads forward passes extra (16 para n_heads=8).
            # V2 usa SPSA: perturba TODAS las coordenadas simultáneamente con
            # un vector Bernoulli ±1, necesitando solo 2 forward passes TOTALES.
            #
            # Propiedad clave: SPSA produce un estimador no-sesgado del gradiente
            # con varianza O(n²) pero Lion solo utiliza el SIGNO → la varianza
            # del estimador SPSA NO afecta al update (sign converge rápido bajo EMA).
            # Referencia: Spall (1992), "Multivariate Stochastic Approximation
            # Using a Simultaneous Perturbation Gradient Approximation"
            #
            # Reducción de FLOPs TTT: 2×n_heads → 2 forward passes (8× más rápido)
            # Para n_heads=8: 16 mamba2 mini-forwards → 3 (base + 2 SPSA)
            with torch.no_grad():
                mini_chunk_len = min(64, S)
                mini_chunk = x_norm[:, :mini_chunk_len].detach()  # [B, mini, D]

                # Forward base para per_token_err (sin perturbación)
                m_out_base = self.mamba2(mini_chunk)
                pred_base   = m_out_base[:, :-1]   # [B, mini-1, D]
                target_base = mini_chunk[:, 1:]     # [B, mini-1, D]

                per_token_err = compute_token_errors_triton(
                    pred_base, target_base
                )  # [B, mini-1]

                # SPSA: perturbación simultánea con Bernoulli ±1
                dt_bias_orig = self.mamba2.dt_bias.data.clone()
                n_bias = dt_bias_orig.numel()
                eps = 1e-3

                # V7: Bernoulli ±1 via randint: 1 tensor op vs 3 anteriores
                # (torch.rand + torch.ones + torch.ones → torch.randint * 2 - 1).
                # Elimina 2 allocaciones intermedias y un kernel CUDA adicional.
                delta = (torch.randint(0, 2, (n_bias,),
                                       device=dt_bias_orig.device,
                                       dtype=torch.float32) * 2.0 - 1.0)

                # ── SPSA forward: H200 factored path vs Ada/Ampere standard ──
                #
                # H200 (spsa_fused=True): spsa_factored_forward() hace:
                #   1× in_proj + conv (precompute) + 1× dual Triton scan kernel
                #   + 2× gate + out_proj. Elimina 2× in_proj + 2× conv redundantes.
                #   Reducción FLOPs: ~65%. Dual scan en SRAM registers (18KB/CTA).
                #
                # Ada/Ampere (spsa_fused=False): 2× mamba2.forward() estándar.
                #   Compatible con todas las GPUs. Sin dependencia de internals.
                if _GPU_PROF.spsa_fused:
                    # Factored: 1× precompute + 1× dual scan + 2× gate+proj
                    out_p, out_m = spsa_factored_forward(
                        self.mamba2, mini_chunk,
                        dt_bias_orig + eps * delta,
                        dt_bias_orig - eps * delta,
                    )
                else:
                    # Standard: 2× full mamba2 forward
                    self.mamba2.dt_bias.data.copy_(dt_bias_orig + eps * delta)
                    out_p = self.mamba2(mini_chunk)

                    self.mamba2.dt_bias.data.copy_(dt_bias_orig - eps * delta)
                    out_m = self.mamba2(mini_chunk)

                    # Restaurar dt_bias original
                    self.mamba2.dt_bias.data.copy_(dt_bias_orig)

                # H200+: fusión de las dos reducciones MSE en 1 kernel launch.
                # Lee out_p[:,:-1], out_m[:,:-1], target en UN solo pass HBM.
                # Ada/Ampere: misma ruta — spsa_mse_fused funciona en cualquier GPU.
                loss_p, loss_m = spsa_mse_fused(
                    out_p[:, :-1], out_m[:, :-1], target_base
                )

                # Estimador SPSA: g_k = (L⁺ - L⁻) / (2·ε·Δ_k)
                _ttt_grad_to_apply = (loss_p - loss_m) / (2.0 * eps * delta)
                
                # TTT Proxy Sync: En capas de Atención Híbrida, la Mamba2 no tiene toda la carga representacional.
                # Reducimos severamente el gradiente del proxy (TTT blind to attention) para evitar que
                # dt_bias sobrecompense tratando de explicar dinámicas de las que ahora se encarga el transformer.
                if getattr(self, 'use_hybrid_attn', False):
                    _ttt_grad_to_apply = _ttt_grad_to_apply * 0.1

                
                # TTT Proxy Sync: En capas de Atención Híbrida, la Mamba2 no tiene toda la carga representacional.
                # Reducimos severamente el gradiente del proxy (TTT blind to attention) para evitar que
                # dt_bias sobrecompense tratando de explicar dinámicas de las que ahora se encarga el transformer.
                if getattr(self, 'use_hybrid_attn', False):
                    _ttt_grad_to_apply = _ttt_grad_to_apply * 0.1


                # Construir mapa de importancia para S tokens completos
                err_mean = per_token_err.mean(dim=-1, keepdim=True)  # [B, 1]
                ttt_importance = err_mean.expand(B, S).clone()        # [B, S]
                ttt_importance[:, :mini_chunk_len - 1] = per_token_err
                ttt_importance = ttt_importance / (ttt_importance.mean(dim=-1, keepdim=True) + 1e-6)

        # --- Forward pass principal (scan con dt_bias ORIGINAL) ---
        # dt_bias NO fue modificado arriba. El Lion update se aplica POST-scan.
        # Esto garantiza equivalencia chunked ↔ full en el SSD scan de Mamba2.
        #
        # Selective Activation Checkpointing (SAC):
        #   Cuando _selective_ckpt_mamba=True (set por ChimeraStack para capas NO
        #   full-checkpointed), solo el scan de Mamba2 se re-computa en backward.
        #   Las activaciones internas de Mamba2 (d_inner=D×expand tensors, estados conv,
        #   etc.) se liberan tras el forward y se re-computan desde x_norm en backward.
        #   Las operaciones ligeras (router, bus, SLR, gating) conservan sus activaciones.
        #   Ahorro: ~80% de la VRAM de activación de esta capa con solo ~80% del
        #   costo de recompute de un full-layer checkpoint.
        if getattr(self, '_selective_ckpt_mamba', False):
            from torch.utils.checkpoint import checkpoint as _ckpt
            mamba_out = _ckpt(self.mamba2, x_norm, use_reentrant=False)
        else:
            mamba_out = self.mamba2(x_norm)

        # --- Local/Global Hybrid Attention (SDPA, O(N) Memory) ---
        if getattr(self, 'use_hybrid_attn', False):
            # Attention parallel to the main path, using residual x
            x_attn_norm = self.hybrid_norm(x)
            
            # Projections
            q = self.q_proj(x_attn_norm)
            k = self.k_proj(x_attn_norm)
            v = self.v_proj(x_attn_norm)
            
            B_sz, S_len, _ = q.shape
            
            q = q.view(B_sz, S_len, self.num_heads, self.headdim).transpose(1, 2)
            k = k.view(B_sz, S_len, self.num_heads, self.headdim).transpose(1, 2)
            v = v.view(B_sz, S_len, self.num_heads, self.headdim).transpose(1, 2)
            
            # FlashAttention-2 / SDPA backend
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            
            attn_out = attn_out.transpose(1, 2).contiguous().view(B_sz, S_len, self.d_model)
            x = x + self.o_proj(attn_out)

        # === Mamba2 gradient stabilization =====================================
        # mamba_out.grad es siempre None en forward (los grads solo existen en
        # backward). La forma correcta de clip local de gradiente es un backward
        # hook en el tensor de activación + clamping de A_log en espacio de datos.
        #
        # Gradient throttle en mamba_out: rescala el gradiente si su norma
        # supera 3.0. Implementado como autograd.Function (_GradThrottleFn)
        # en lugar de register_hook para ser 100% seguro con:
        #   - use_reentrant=False checkpoint (no duplica side-effects en recompute)
        #   - tensores sin requires_grad (register_hook lanzaría RuntimeError)
        #   - CUDA Graphs (graph_mode → se omite idéntico al else-branch)
        # Condicional a not self.graph_mode para CUDA Graph compatibility.
        if self.training and not self.graph_mode:
            mamba_out = _grad_throttle(mamba_out, max_norm=3.0)
        # Clamp A_log: A = exp(A_log) – debe salir negativo para estabilidad SSM.
        # Rango [-10, -0.5]: log(-λ_min=e^-0.5≈0.6) → λ_min≈0.6 nunca >1 (estable).
        with torch.no_grad():
            self.mamba2.A_log.data.clamp_(min=-10.0, max=-0.5)
        # =======================================================================

        # --- TTT-LITE: gradiente POST-scan (SOLO se computa, NO se aplica) ---
        # FIX DRIFT: El update ya no se aplica en el forward.
        # Razón: aplicar Lion dentro del forward modifica dt_bias ENTRE chunks cuando
        # el trainer divide S en trozos → cada chunk ve un dt_bias distinto → el
        # output chunked diverge del output full (drift medido: 106.8%).
        # Solución: el forward SOLO computa el gradiente y lo guarda en
        # self._pending_ttt_grad. El trainer llama update_ttt_inplace() una vez
        # al final del step, garantizando que dt_bias no cambia dentro del forward.
        if _ttt_grad_to_apply is not None:
            # Guardar float32 para que lion_constrained_update_inplace no tenga que castear
            self._pending_ttt_grad = _ttt_grad_to_apply.detach().float()
            _ttt_grad_to_apply = None

        # 1b. TTT-Full low-rank correction (Plan §2.2.2) — solo en tier FULL
        #     Corrección adaptatva: out' = out + σ(scale) * (out @ Vᵀ @ Uᵀ)
        #     Los parámetros U, V se aprenden vía gradient descent del loss principal
        #     ("outer loop" — no interfiere con el backward del scan interno de Mamba2)
        # TTT-Full: siempre activo, escalado por prob_full.
        # Antes: hard threshold > 0.35 → ttt_U/V/scale nunca recibían gradientes
        # desde init (prob_full ≈ 0.333 < 0.35 con router uniforme).
        # Ahora: prob_full actúa como soft-gate continuo → gradiente siempre fluye.
        scale      = torch.sigmoid(self.ttt_full_scale)  # escalar, en [0,1]
        h          = mamba_out @ self.ttt_V.T             # [B, S, rank]
        correction = h @ self.ttt_U.T                     # [B, S, D]
        mamba_out  = mamba_out + prob_full * scale * correction

        # 2. SGR + SLR Cross-Attention (Triton)
        #    SLR queries come from x_norm (independent gradient path),
        #    K/V come from mamba_out (sequentially-enriched context).
        #    This makes SLR a true cross-attention mechanism:
        #    "the raw input asks, Mamba2 answers".
        #    The Router is the sole controller of SLR contribution.
        slr_out, sgr_indices = self.slr(
            query_base=x_norm,
            context_base=mamba_out,
            importance=ttt_importance,
        )

        # 3. NativeLandmarkArchive — archivar cuando hay contenido complejo (FULL)
        # En graph_mode: sgr_indices siempre existe (SLR corre siempre).
        # ttt_importance puede ser None (TTT está fuera del graph); se pasa directo.
        if not self.graph_mode and ttt_importance is not None and not self._skip_side_effects:
            # FIX (checkpoint compat): maybe_archive() usa F.softplus(w_bands) cuya
            # save_for_backward es interceptada por checkpoint hooks → tensores extra en
            # el frame (171 vs 145) cuando n_triggers>0 activa el camino espectral completo.
            # La corrección: no_grad elimina estas saves. Los gradientes de w_bands/freq_bias
            # fluyen correctamente por retrieve() (que SÍ corre con grad); en maybe_archive()
            # los valores de w_bands se usan sólo para thresholding vía .item() (desconectado
            # del grafo de todas formas). h_mean tampoco necesita gradient aquí: es un update
            # de buffer de estado, no parte del camino de pérdida hacia los parámetros.
            with torch.no_grad():
                self.archive.maybe_archive(
                    scan_out=mamba_out.detach(),
                    ttt_importance=ttt_importance.detach(),
                    tier_probs=probs.detach(),
                    sgr_indices=sgr_indices,
                )
        elif self.graph_mode and not self._skip_side_effects:
            # En graph_mode siempre archivamos (usando prob_full como importancia proxy)
            # para evitar el Python-if con condición tensorial.
            # prob_full: [B, 1, 1] → squeeze a [B] → expand a [B, S]
            self.archive.maybe_archive(
                scan_out=mamba_out,
                ttt_importance=probs[:, 2].unsqueeze(1).expand(B, S).detach(),
                tier_probs=probs,
                sgr_indices=sgr_indices,
            )

        # Retrieval: soft-gate tensor (sin Python-if sobre tensor) → CUDA Graph safe.
        # arch_gate ∈ [0, 1]: 0 cuando total_active_prob < arch_threshold.
        arch_gate = ((total_active_prob - arch_threshold) /
                     (1.0 - arch_threshold + 1e-6)).clamp(0.0, 1.0)

        # FIX GRAD INVARIANT: compress_ctx se añade FUERA del arch_gate.
        # Problema anterior: retrieve() incluía compress_ctx en su retorno, pero
        # arch_gate=0 zeroeaba TODA la salida de retrieve() → compress.w dead.
        # Solución: get_compress_ctx() siempre activo (bypass arch_gate),
        # retrieve() solo devuelve la parte de landmarks (gateada por arch_gate).
        compress_ctx = self.archive.get_compress_ctx(slr_out)          # [B, S, D]
        retrieved    = self.archive.retrieve(slr_out)                  # [B, S, D]
        lm_delta     = retrieved - slr_out                             # delta de landmarks

        # --- Soft-gating por tier ---
        # FAST   → solo Mamba2 (sin overhead SLR/archive)
        # HYBRID → SLR + compress_ctx (siempre activo, sin landmarks gateados)
        # FULL   → SLR + compress_ctx + landmark retrieval (máxima información)
        #
        # FIX V5: en V4, slr_out se modificaba in-place ANTES de crear las
        # representaciones, causando que full_representation = slr_out + compress +
        # arch_gate*lm_delta recibiera esos términos DUPLICADOS (double-add bug).
        # Ahora las representaciones se construyen directamente sin mutación.
        fast_representation   = mamba_out
        hybrid_representation = slr_out + compress_ctx
        full_representation   = slr_out + compress_ctx + arch_gate * lm_delta

        out = (prob_fast   * fast_representation +
               prob_hybrid * hybrid_representation +
               prob_full   * full_representation)

        # ── Document boundary gate — SSM ─────────────────────────────────────
        # Zero la contribución del SSM/SLR/archive en la primera posición de
        # cada nuevo documento. doc_open_mask [B,S,1]: 0 en transición, 1 resto.
        if doc_open_mask is not None:
            out = out * doc_open_mask

        # ── CAS / MoE — ruta paramétrica dinámica ─────────────────────────
        # CAS tiene prioridad sobre MoE si ambos están activos.
        # Bypass en graph_mode (dispatch dinámico → incompatible con CUDA Graph).
        moe_lb_loss = None
        cas_aux = None
        if self.use_cas and not self.graph_mode:
            # TTT proxy loss: norma media del error por token como señal de sorpresa.
            # Rango típico: 0.0 (predecible) a 2.0+ (alta sorpresa).
            ttt_proxy = None
            if ttt_importance is not None:
                ttt_proxy = ttt_importance.mean()
            out, cas_aux = self.cas(out, ttt_proxy_loss=ttt_proxy, graph_mode=self.graph_mode)
        elif self.use_moe and not self.graph_mode:
            out, moe_lb_loss = self.moe(out)

        # Guardar out ANTES del bus para aislar su delta en el boundary gate.
        pre_bus_out = out

        out, new_bus_state = (
            self.bus.forward_ring(out, bus_ring, self._layer_idx)
            if bus_ring is not None
            else self.bus(out, bus_cache)
        )
        # out = pre_bus_out + bus_modulation.
        # ── Document boundary gate — Bus ──────────────────────────────────────
        # Zero la modulación del ring en transiciones: el context de doc A
        # no debe afectar el residual del primer token de doc B.
        # El ring sí se actualiza (summary de doc A queda en el ring) para que
        # capas posteriores puedan decidir ignorarlo vía su propio gate.
        if doc_open_mask is not None:
            bus_delta = out - pre_bus_out                    # [B, S, D]
            out       = pre_bus_out + bus_delta * doc_open_mask

        final = x + out
        # ── Soft-norm: estabilizador de norma residual ──────────────────────
        # Idéntico al estabilizador de step() — elimina la divergencia entre
        # la ruta de training (forward) y la ruta de inferencia (step).
        # Umbral conservador: 4×√D → solo se activa cuando el residual supera
        # 4× la norma esperada (||x+out|| ≈ √D bien inicializado → scale ≈ 1.0).
        # Protección ante gradiente explosivo, fine-tuning o init corruptos.
        _expected_norm = self.d_model ** 0.5
        _actual_norm   = final.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        _scale         = (_expected_norm * 4.0 / _actual_norm).clamp(max=1.0)
        final          = final * _scale

        if return_aux:
            aux = {
                'routing_probs':      probs.detach(),
                'routing_probs_grad': probs,
                'router_logits':      router_logits,
                'ttt_importance':     ttt_importance,
                'ttt_active':         ttt_importance is not None,
                'spectral_delta':     self.archive.get_spectral_delta() if self.use_spectral_vsa else None,
                'slr_out':            slr_out,
                'moe_lb_loss':        moe_lb_loss,
                'cas_aux':            cas_aux,
            }
            return final, new_bus_state, aux
        return final, new_bus_state


# ─────────────────────────────────────────────────────────────────────────────
# CUDA Graph Capture para decode ultrarrápido
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def make_cuda_graph_step(
    layer: AdvancedChimeraLayer,
    batch_size: int,
    device=None,
    ring_size: int = None,
    warmup_iters: int = None,
):
    """
    Captura layer.step() en un CUDA Graph para decode de baja latencia.

    REQUISITO: llamar DESPUÉS del prefill completo, cuando archive._lm_cache
    ya está poblado con los landmarks de contexto.

    Ganancia de latencia estimada (RTX 4050 Laptop):
      Sin graph:   ~2.5ms Python dispatch + ~1.15ms CUDA  = ~3.65ms/token
      Con graph:   ~0.15ms replay                          = ~1.30ms/token (≈ 2.8×)

    Ganancia estimada en H200:
      Sin graph:   ~0.8ms Python dispatch + ~0.15ms CUDA  = ~0.95ms/token
      Con graph:   ~0.02ms replay                          = ~0.17ms/token (≈ 5.6×)

    Pasos internos:
      1. Alocar tensores estáticos (x, conv_state, ssm_state, bus_ring).
      2. Pre-computar delta de archive con query=0 (aprox válida post-prefill).
      3. Warmup: N iteraciones sin captura (fuerza Triton autotune + cuDNN bench).
      4. Capturar el graph en stream privado.
      5. Retornar graphed_step(x, cache) → (out, cache).

    LIMITACIÓN PRINT ARCHIVE: en graph mode, el archive context es fijo (pre-computado
    con query=0 antes de capture). Para sesiones muy largas donde el archive crece
    durante decode (raro), re-capturar el graph periódicamente con esta função.

    Args:
        layer:        AdvancedChimeraLayer en eval() mode, post-prefill.
        batch_size:   Tamaño del batch de decode (generalmente 1).
        device:       CUDA device. None → detecta desde parámetros del layer.
        ring_size:    Tamaño del ring buffer del bus. None → GPU profile automático.
        warmup_iters: Iteraciones de warmup. None → GPU profile automático.

    Returns:
        graphed_step(x, cache) → (out, cache)
          - x:     [batch_size, 1, d_model]  (copiado al tensor estático interno)
          - cache: dict de allocate_inference_cache() (actualizado in-place)
          - out:   [batch_size, 1, d_model]  (clonado — safe para el caller)
    """
    layer.eval()
    device = device or next(layer.parameters()).device
    D      = layer.d_model

    # Perfil GPU para parámetros automáticos
    try:
        prof = _get_gpu_profile()
    except Exception:
        prof = None

    if ring_size is None:
        ring_size     = prof.ring_size          if prof else 16
    if warmup_iters is None:
        warmup_iters  = prof.graph_warmup_iters if prof else 3

    # ── 1. Tensores estáticos de entrada/salida ────────────────────────────────
    static_x     = torch.zeros(batch_size, 1, D, device=device)
    static_cache = layer.allocate_inference_cache(
        batch_size, ring_size=ring_size, dtype=static_x.dtype
    )

    # ── 2. Pre-computar delta de archive (una sola vez, post-prefill) ──────────
    # archive.retrieve() llama n_archived.item() → CPU sync → NO graph-safe.
    # Solución: calcular el delta que el archive añadiría a un query=0,
    # guardarlo como tensor fijo. En graph replay: out = mamba_out + arc_delta.
    # Exactitud: aprox válida cuando mamba_out ≈ distribución conocida post-prefill.
    dummy_x = torch.zeros(batch_size, 1, D, device=device)
    try:
        arc_full  = layer.archive.retrieve(dummy_x)          # [B, 1, D]
        arc_cmprx = layer.archive.get_compress_ctx(dummy_x)  # [B, 1, D]
        arc_delta = (arc_full - dummy_x) + arc_cmprx         # delta neto
    except Exception:
        arc_delta = torch.zeros(batch_size, 1, D, device=device)

    static_cache['_archive_ctx'] = arc_delta.to(static_x.dtype)

    # ── 3. Warmup ──────────────────────────────────────────────────────────────
    # Triton autotune y cuDNN benchmark se ejecutan en la PRIMERA llamada real.
    # Si capturamos antes del warmup, el graph captura kernels sub-óptimos.
    for _ in range(warmup_iters):
        static_x.normal_()                     # input aleatorio → fuerza autotune
        _, _ = layer.step(static_x, static_cache)
    static_x.zero_()                           # reset a cero antes de capture

    # ── 4. Captura del CUDA Graph ─────────────────────────────────────────────
    # torch.cuda.graph() usa un stream privado → no interfiere con stream default.
    # Compatible con PyTorch 2.x; si hay error, salimos con excepción explícita.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out, static_cache = layer.step(static_x, static_cache)

    # ── 5. Closure de replay ──────────────────────────────────────────────────
    def graphed_step(x: torch.Tensor, cache: dict) -> tuple:
        """
        Ejecuta el CUDA Graph capturado para un token de decode.

        Contrato:
          - x DEBE tener forma exacta [batch_size, 1, d_model].
          - cache DEBE ser el dict retornado por allocate_inference_cache().
          - No modificar cache externamente durante replay (data race).
          - out retornado es un clone seguro (el caller puede modificarlo).
        """
        static_x.copy_(x)
        g.replay()
        cache['conv_state'].copy_(static_cache['conv_state'])
        cache['ssm_state'].copy_(static_cache['ssm_state'])
        cache['bus_ring'].copy_(static_cache['bus_ring'])
        return static_out.clone(), cache

    return graphed_step


# ─────────────────────────────────────────────────────────────────────────────
# ChimeraAnnealer — Cosine annealing for router thresholds (cold-start)
#
#   Problema: con SLR potenciado (254× más gradiente), el router puede enviar
#   la mayoría de tokens a HYBRID/FULL antes de que Mamba2 aprenda la estructura
#   base del lenguaje. Solución: arrancar con umbrales altos (→ forzar FAST)
#   y reducirlos gradualmente con cosine schedule.
#
#   Uso:
#       annealer = ChimeraAnnealer(model, warmup_steps=2000)
#       for step in range(N):
#           annealer.step(step)
#           # ... forward/backward/optimizer ...
# ─────────────────────────────────────────────────────────────────────────────

class ChimeraAnnealer:
    """
    Cosine annealing de slr_threshold y arch_threshold en todas las capas.

    Schedule: high → target via cosine decay over warmup_steps.
    Después de warmup_steps, los umbrales quedan fijos en sus targets.
    """

    def __init__(
        self,
        model:         nn.Module,
        warmup_steps:  int   = 2000,
        slr_start:     float = 0.90,
        slr_target:    float = 0.30,
        arch_start:    float = 0.85,
        arch_target:   float = 0.50,
    ):
        self.warmup_steps = max(1, warmup_steps)
        self.slr_start    = slr_start
        self.slr_target   = slr_target
        self.arch_start   = arch_start
        self.arch_target  = arch_target
        # Collect all AdvancedChimeraLayer instances
        self.layers = [m for m in model.modules()
                       if isinstance(m, AdvancedChimeraLayer)]

    def step(self, current_step: int):
        """Update thresholds based on current training step."""
        if current_step >= self.warmup_steps:
            slr_val  = self.slr_target
            arch_val = self.arch_target
        else:
            # Cosine decay: 1 → 0 over warmup_steps
            progress = current_step / self.warmup_steps
            cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            slr_val  = self.slr_target  + (self.slr_start  - self.slr_target)  * cos_decay
            arch_val = self.arch_target + (self.arch_start - self.arch_target) * cos_decay
        for layer in self.layers:
            layer.slr_threshold.fill_(slr_val)
            layer.arch_threshold.fill_(arch_val)

    def get_state(self) -> dict:
        """Current threshold values for logging."""
        if not self.layers:
            return {}
        return {
            'annealer/slr_threshold':  self.layers[0].slr_threshold.item(),
            'annealer/arch_threshold': self.layers[0].arch_threshold.item(),
        }


if __name__ == "__main__":
    import time
    print("Initializing ADVANCED Mamba 2 CHIMERA...")
    
    # Needs to wait for mamba_ssm compilation, so we check importing it first

    model = AdvancedChimeraLayer(d_model=256, expand=2, headdim=32).cuda()
    model.train()

    x = torch.randn(2, 512, 256, device="cuda", requires_grad=True)

    t0 = time.time()
    out, bus_cache = model(x, bus_cache=None)
    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize()
    t1 = time.time()

    print(f"Forward + Backward Works! Shape: {out.shape}")
    print(f"Bus cache: {bus_cache.shape}")
    print(f"Time: {(t1 - t0)*1000:.2f} ms")

    tier_probs, _ = model.router(model.norm(x).detach())
    print(f"Router probabilities (FAST/HYBRID/FULL): {[f'{p:.3f}' for p in tier_probs[0].tolist()]}")
    slr = model.slr
    print(f"SGR top-K fraction: {slr.sgr.top_k_frac*100:.1f}%  ({int(slr.sgr.top_k_frac*512)}/512 tokens)")
    print(f"λ differential V2: {slr.lam_logit.sigmoid().item():.4f}")
    archive_info = model.archive.get_archive_info()
    print(f"LandmarkArchive: {archive_info}")
    ttu = model.ttt_U
    ttv = model.ttt_V
    print(f"TTT-Full U/V shapes: {tuple(ttu.shape)} / {tuple(ttv.shape)}  rank={model.ttt_rank}")
    print(f"TTT-Full scale (sigmoid): {model.ttt_full_scale.sigmoid().item():.4f}")
    print("SUCCESS: Mamba2 + Multi-scale A + TTT-Lite + Constrained-TTT + TTT-Full-LowRank +"
          " SGR + SLR-V2(Triton) + AsyncLightBus(128d) + NativeLandmarkArchive-Optimized")

    # ── Benchmark precision: TF32 vs BF16 ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Benchmark precision (TF32 vs BF16, B=2 S=512 y B=1 S=2048)")
    print("=" * 60)
    import time

    REPS = 50
    def bench(mod, xin, reps=REPS):
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps): mod(xin)
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3

    model_fp32_eval = model.eval()
    model_bf16 = AdvancedChimeraLayer(d_model=256, expand=2, headdim=32).cuda().bfloat16().eval()

    # Warmup ambos
    x_small_fp32 = torch.randn(2, 512, 256, device="cuda", dtype=torch.float32)
    x_small_bf16 = x_small_fp32.bfloat16()
    with torch.no_grad():
        for _ in range(5):
            model_fp32_eval(x_small_fp32)
            model_bf16(x_small_bf16)
    torch.cuda.synchronize()

    # Escenario 1: entrenamiento-like (B=2, S=512) — BF16 gana
    ms32_s = bench(model_fp32_eval, x_small_fp32)
    ms16_s = bench(model_bf16, x_small_bf16)
    tps32_s = 2*512 / (ms32_s / 1e3)
    tps16_s = 2*512 / (ms16_s / 1e3)
    print(f"  Training (B=2,S=512)   FP32+TF32: {ms32_s:.2f} ms  {tps32_s:,.0f} tok/s")
    print(f"  Training (B=2,S=512)   BF16:      {ms16_s:.2f} ms  {tps16_s:,.0f} tok/s  "
          f"({ms32_s/ms16_s:.2f}x {'↑ BF16 gana' if ms32_s > ms16_s else '↓ FP32+TF32 gana'})")

    # Escenario 2: inferencia B=1 largo (B=1, S=2048) — TF32 gana
    x_long_fp32 = torch.randn(1, 2048, 256, device="cuda", dtype=torch.float32)
    x_long_bf16 = x_long_fp32.bfloat16()
    ms32_l = bench(model_fp32_eval, x_long_fp32)
    ms16_l = bench(model_bf16, x_long_bf16)
    tps32_l = 2048 / (ms32_l / 1e3)
    tps16_l = 2048 / (ms16_l / 1e3)
    print(f"  Inferencia (B=1,S=2048) FP32+TF32: {ms32_l:.2f} ms  {tps32_l:,.0f} tok/s")
    print(f"  Inferencia (B=1,S=2048) BF16:       {ms16_l:.2f} ms  {tps16_l:,.0f} tok/s  "
          f"({ms32_l/ms16_l:.2f}x {'↑ BF16 gana' if ms32_l > ms16_l else '↓ FP32+TF32 gana'})")

    # Sanity: output BF16 finito
    with torch.no_grad():
        out_bf16, _ = model_bf16(x_small_bf16)
    assert out_bf16.isfinite().all(), "BF16 output tiene NaN/Inf!"
    print(f"\n  BF16 output: dtype={out_bf16.dtype}  finite=True  ✓")
    print("  Recomendación: TF32 (ya activo) para inferencia;"
          " BF16 para training con batch ≥ 2")
    print("  [BF16 + TF32 OK]")


