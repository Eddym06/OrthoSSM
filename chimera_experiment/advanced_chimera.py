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
from ttt_kernel import lion_constrained_update_inplace, compute_token_errors_triton
from sdtm_memory import SDTMMemory
from gpu_profile import get_gpu_profile as _get_gpu_profile

# Imports lazy para carry SSM — disponibles en mamba_ssm≥2.0 y causal_conv1d≥1.4
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as _mamba_scan
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn
    from einops import rearrange as _rearrange
    _MAMBA_SCAN_AVAIL = True
except ImportError:
    _MAMBA_SCAN_AVAIL = False

# ─────────────────────────────────────────────────────────────────────────────
# FP8 Linear — para H100/H200 (SM≥9.0) donde FP8 dobla throughput
# ─────────────────────────────────────────────────────────────────────────────

_FP8_DTYPE = getattr(torch, 'float8_e4m3fn', None)
_FP8_AVAIL = (_FP8_DTYPE is not None) and hasattr(torch, '_scaled_mm')
_FP8_MAX   = 448.0   # float8_e4m3fn max representable value


class Fp8Linear(nn.Module):
    """
    nn.Linear con forward en FP8 (float8_e4m3fn) para H100/H200 (SM≥9.0).

    Cuantificación dinámica por tensor: escala = amax(|x|) / 448.
    Backward en BF16/FP32 vía GradScaler estándar (FP8 solo en forward).
    Fallback transparente a BF16 si _scaled_mm no está disponible.

    Uso:
        linear_fp8 = Fp8Linear.from_linear(some_nn_linear)

    Ganancia esperada en H200:
      - BF16 GEMM: ~1,979 TFLOPS
      - FP8  GEMM: ~3,958 TFLOPS (2× throughput teórico)
      - Speedup real: ~1.5-2× (considerando memoria y overhead)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        fkw = {'device': device, 'dtype': dtype or torch.float32}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **fkw))
        self.bias   = nn.Parameter(torch.zeros(out_features, **fkw)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> 'Fp8Linear':
        """Convierte nn.Linear → Fp8Linear copiando pesos exactamente."""
        m = cls(
            linear.in_features, linear.out_features,
            bias=(linear.bias is not None),
            device=linear.weight.device, dtype=linear.weight.dtype,
        )
        m.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            m.bias.data.copy_(linear.bias.data)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _FP8_AVAIL:
            return F.linear(x, self.weight, self.bias)
        orig_dtype = x.dtype
        orig_shape = x.shape
        # Trabajar en BF16 como punto de partida para cuantificación
        w = self.weight.to(torch.bfloat16)
        m = x.reshape(-1, self.in_features).to(torch.bfloat16)
        try:
            # Escala dinámica: divide por max_abs para llevar a [-1, 1] luego ×448
            amax_x = m.float().abs().max().clamp(min=1e-12)
            amax_w = w.float().abs().max().clamp(min=1e-12)
            scale_x = (amax_x / _FP8_MAX).to(torch.float32)
            scale_w = (amax_w / _FP8_MAX).to(torch.float32)
            x_fp8 = (m / scale_x).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
            w_fp8 = (w / scale_w).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
            # _scaled_mm(A, B, sa, sb) = (A @ B) * sa * sb
            # = (x/scale_x @ (w/scale_w).T) * scale_x * scale_w = x @ w.T  ✓
            out = torch._scaled_mm(
                x_fp8, w_fp8.T,
                scale_a=scale_x, scale_b=scale_w,
                out_dtype=torch.bfloat16,
            )
        except (RuntimeError, AssertionError, AttributeError):
            # Fallback BF16 si _scaled_mm falla (versión PyTorch incompatible)
            out = m @ w.T
        out = out.to(orig_dtype)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        status = 'fp8_active' if _FP8_AVAIL else 'bf16_fallback'
        return (f'in={self.in_features}, out={self.out_features}, '
                f'bias={self.bias is not None}, mode={status}')


def apply_fp8_to_chimera(layer: 'AdvancedChimeraLayer', prof=None) -> 'AdvancedChimeraLayer':
    """
    Reemplaza nn.Linear → Fp8Linear en las proyecciones clave del layer.

    Solo activo cuando prof.use_fp8_fwd=True (H100/H200, SM≥9.0).
    En RTX 4050/A100 retorna el layer sin cambios (no-op seguro).

    Lineales convertidas (impacto alto):
      bus.publish, bus.gather_q, bus.modulate  — bus decode O(ring_size)
      slr.out_proj                             — proyección final SLR D×D
      slr.proj.proj                            — proyección fusionada 6D

    Lineales NO convertidas:
      mamba2 internos — biblioteca externa, no modificable
      router.mlp      — pequeño (D→32→3), impacto < 0.5%
      ttt_U/V         — matrices low-rank (D×4), impacto negligible
    """
    if prof is None:
        try:
            prof = _get_gpu_profile()
        except Exception:
            return layer
    if not prof.use_fp8_fwd:
        return layer   # no-op: hardware sin FP8 (RTX 4050, A100, etc.)

    # Bus: impacto máximo en decode (ejecuta cada token)
    layer.bus.publish  = Fp8Linear.from_linear(layer.bus.publish)
    layer.bus.gather_q = Fp8Linear.from_linear(layer.bus.gather_q)
    layer.bus.modulate = Fp8Linear.from_linear(layer.bus.modulate)

    # SLR: proyección de salida (D×D, ejecuta en cada forward de training)
    layer.slr.out_proj = Fp8Linear.from_linear(layer.slr.out_proj)

    # FusedProjectionSplit: si tiene una capa linear interna
    if hasattr(layer.slr, 'proj') and hasattr(layer.slr.proj, 'proj'):
        if isinstance(layer.slr.proj.proj, nn.Linear):
            layer.slr.proj.proj = Fp8Linear.from_linear(layer.slr.proj.proj)

    return layer

class GatedComplexityPredictor(nn.Module):
    def __init__(self, d_model: int, n_tiers: int = 3,
                 min_prob_floor: float = 0.05):
        """
        Router con robustez anti-colapso.

        Args:
            min_prob_floor: probabilidad mínima garantizada por tier (default=5%).
                            Previene que el router colapse a un único tier.
                            Implementado como max(softmax_out, floor) re-normalizado.
                            Efecto: con n_tiers=3 y floor=0.05, el rango real de
                            probabilidad es [0.05, 0.90] en vez de [0, 1].
        """
        super().__init__()
        d_hidden = 32
        self.min_prob_floor = min_prob_floor
        self.n_tiers        = n_tiers
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
        x_mean = x.mean(dim=1)                       # [B, D]
        logits  = self.mlp(x_mean)                   # [B, n_tiers]

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
        
    def forward(self, x: torch.Tensor, bus_cache: torch.Tensor = None):
        """
        x: [B, S, D]
        bus_cache: [B, num_layers_so_far, bus_dim] (Optional)
        Returns:
            x_out: modulated x [B, S, D]
            new_cache: updated bus cache [B, num_layers_so_far + 1, bus_dim]
        """
        B, S, D = x.shape
        
        # 1. Publish summary of current layer
        summary = self.publish(x.mean(dim=1)) # [B, bus_dim]
        summary = F.normalize(summary, p=2, dim=-1)
        summary_unsqueezed = summary.unsqueeze(1) # [B, 1, bus_dim]
        
        # 2. Gather from previous layers if cache exists
        if bus_cache is None or bus_cache.shape[1] == 0:
            # No hay contexto previo: auto-modulación de resguardo.
            # Garantiza que gate y modulate siempre reciban gradientes,
            # incluso en la primera capa o cuando bus_cache está vacío.
            trivial_mod = self.modulate(
                summary.unsqueeze(1).expand(-1, S, -1)
            ) * torch.sigmoid(self.gate)  # [B, S, D]
            return x + trivial_mod, summary_unsqueezed
            
        q = self.gather_q(x) # [B, S, bus_dim]
        
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
        modulation = self.modulate(gathered) * torch.sigmoid(self.gate)
        
        # 4. Update cache: orden cronológico (capa 0, 1, 2, ...)
        new_cache = torch.cat([bus_cache, summary_unsqueezed], dim=1)
        
        return x + modulation, new_cache

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
        x_sq    = x.squeeze(1)                                         # [B, D]
        summary = F.normalize(self.publish(x_sq), p=2, dim=-1)         # [B, bus_dim]

        # Roll: desplaza izquierda (evicta oldest), mantiene forma exacta
        # torch.roll ya retorna tensor nuevo — .clone() era redundante
        new_ring = torch.roll(bus_ring, -1, dims=1)                    # [B, ring_size, bus_dim]
        new_ring[:, -1, :] = summary                                    # escribe en ranura fija

        # Cross-attention de x contra todos los slots del ring
        q      = self.gather_q(x)                                       # [B, 1, bus_dim]
        scores = torch.bmm(q, new_ring.transpose(1, 2)) / math.sqrt(self.bus_dim)  # [B,1,ring_size]
        attn   = F.softmax(scores, dim=-1)                              # [B, 1, ring_size]
        gathered   = torch.bmm(attn, new_ring)                          # [B, 1, bus_dim]
        modulation = self.modulate(gathered) * torch.sigmoid(self.gate) # [B, 1, D]

        return x + modulation, new_ring


class AdvancedChimeraLayer(nn.Module):
    def __init__(self, d_model: int = 256, expand: int = 2, headdim: int = 32,
                 layer_idx: int = 0,
                 d_state: int = 64,
                 bus_dim: int = 128,
                 landmark_dim: int = 128,
                 max_landmarks: int = 64,
                 ttt_err_threshold: float = 0.3,
                 sdtm_n_heads: int = 1,
                 sdtm_d_mem: int = 0):
        super().__init__()
        self.d_model = d_model
        self._layer_idx = layer_idx
        
        # We will dynamically import mamba_ssm here to ensure it's available at runtime
        # avoiding issues if the module is still compiling globally in another process
        from mamba_ssm import Mamba2
        
        # Base Mamba 2 official — layer_idx requerido para inference_params (chunked_prefill)
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
        self.bus = AsyncLightBus(d_model, bus_dim=bus_dim)

        # 6. NativeLandmarkArchive — activo solo en tier FULL
        self.archive = NativeLandmarkArchive(
            d_model=d_model,
            landmark_dim=landmark_dim,
            max_landmarks=max_landmarks,
            ttt_err_threshold=ttt_err_threshold,
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

        # 7. SDTM — Surprise-Driven Dual-Timescale Memory (Multi-Head)
        #    Memoria asociativa dinámica O(1) VRAM con n_heads cabezas independientes.
        #    Cada cabeza se especializa en patrones distintos.
        #    Read gated por prob_hybrid+prob_full. Write gated por surprise signal.
        _sdtm_d_mem = sdtm_d_mem if sdtm_d_mem > 0 else max(64, d_model // 4)
        self.sdtm = SDTMMemory(
            d_model=d_model,
            d_mem=_sdtm_d_mem,
            n_heads=sdtm_n_heads,
        )

    # ── API para CUDA Graph y trainer explícito ─────────────────────────────────
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
            if grad is None or self.ttt_lr <= 0.0:
                return None
            self._pending_ttt_grad = None

            dt_bias_fp32 = self.mamba2.dt_bias.detach().float().clone()
            lion_constrained_update_inplace(
                dt_bias_fp32,
                self.dt_momentum,
                grad,
                self.mamba2.A_log.detach().float().view(-1),
                beta=self.ttt_beta,
                lr=self.ttt_lr,
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
            dt_bias_adapt = self.mamba2.dt_bias.detach().float().clone().requires_grad_(True)
            with _ttt_dt_bias_override(self.mamba2, dt_bias_adapt) as dt_param:
                m_out = self.mamba2(chunk)
            pred   = m_out[:, :-1]
            target = chunk[:, 1:]
            per_token_err = compute_token_errors_triton(pred.detach(), target.detach())
            loss = F.mse_loss(pred.float(), target.float())
            grad = torch.autograd.grad(loss, dt_param)[0]

        complexity = (per_token_err.mean() / (per_token_err.mean() + 1e-6)).item()

        dt_bias_fp32 = self.mamba2.dt_bias.detach().float().clone()
        lion_constrained_update_inplace(
            dt_bias_fp32,
            self.dt_momentum,
            grad,
            self.mamba2.A_log.detach().float().view(-1),
            beta=self.ttt_beta,
            lr=self.ttt_lr,
            active_prob=complexity,
            mom_comp=self.mom_kahan_comp,
            dt_comp=self.dt_kahan_comp,
        )
        self.mamba2.dt_bias.data.copy_(dt_bias_fp32.to(self.mamba2.dt_bias.dtype))
        return per_token_err  # [B, mini_len-1]

    @torch.no_grad()
    def update_sdtm_inplace(self, seq_len: int = 0):
        """
        Apply pending SDTM write and run maintenance cycle.

        Called by the trainer AFTER loss.backward() + optimizer.step(),
        same pattern as update_ttt_inplace().

        Steps:
          1. Apply pending Lion+Kahan write to M_fast
          2. Apply usage-weighted decay
          3. Maybe consolidate M_fast → M_slow (if interval reached)
        """
        self.sdtm.post_forward_update(seq_len=seq_len)

    @torch.no_grad()
    def archive_deferred(self):
        """
        Aplica maybe_archive() pendiente del forward con graph_mode=True.

        Patrón idéntico a update_ttt_inplace() / update_sdtm_inplace():
        el trainer llama esto DESPUÉS de loss.backward() + optimizer.step().

        Ejecuta las operaciones que causan graph breaks (.item(), buffer mutations,
        data-dependent control flow) fuera del grafo compilado.
        """
        data = getattr(self, '_pending_archive_data', None)
        if data is None:
            return
        mamba_out_det, ttt_importance_proxy, probs_det, sgr_indices_det = data
        self._pending_archive_data = None
        # maybe_archive con datos detached — seguro fuera del grafo
        if sgr_indices_det is not None:
            self.archive.maybe_archive(
                scan_out=mamba_out_det,
                ttt_importance=ttt_importance_proxy,
                tier_probs=probs_det,
                sgr_indices=sgr_indices_det,
            )
        # SDTM compute_write deferred: usar proxy importance y probs
        B, S = ttt_importance_proxy.shape
        if S > 1:
            # Necesitamos x_norm, pero solo tenemos mamba_out
            # usamos mamba_out como proxy (ya normalizado)
            self.sdtm.compute_write(
                mamba_out_det, ttt_importance_proxy,
                prob_full_mean=probs_det[:, 2].mean(),
            )
            z_usage = F.gelu(self.sdtm.W_enc(mamba_out_det))
            self.sdtm.update_usage(z_usage)

    def set_graph_mode(self, enable: bool = True):
        """Activa/desactiva graph_mode para torch.compile compatibility."""
        self.graph_mode = enable
        return self
    # ─────────────────────────────────────────────────────────────────────────────

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int = 1,
                                   dtype=None, ring_size: int = None):
        """
        Aloca los buffers de inferencia autoregresiva (token-by-token).
        Retorna un dict cache compatible con self.step().

        cache = {
          'conv_state':   [B, d_inner+2*ngroups*d_state, d_conv] — rolling buffer
          'ssm_state':    [B, nheads, headdim, d_state]           — SSM hidden
          'bus_ring':     [B, ring_size, bus_dim]                 — ring buffer fijo
          '_archive_ctx': None — delta de archive pre-computado (llenado por
                                  make_cuda_graph_step() tras el prefill)
        }

        ring_size:
            Número de tokens recientes que conserva el bus ring durante decode.
            Si None, se toma de gpu_profile.ring_size (adaptativo al hardware).
            RTX 4050 Laptop = 16; A100 = 32; H200 = 32; B200 = 64.
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
        return {
            'conv_state':   conv_state,
            'ssm_state':    ssm_state,
            'bus_ring':     torch.zeros(
                                batch_size, ring_size, self.bus.bus_dim,
                                device=device, dtype=buf_dtype,
                            ),
            '_archive_ctx': None,  # None → live retrieve; tensor → graph mode
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

        # SDTM read during decode (cheap matmuls, always active)
        sdtm_out = self.sdtm.read(out)
        out = out + sdtm_out

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

    def forward(self, x, bus_cache=None, return_aux: bool = False,
                inference_params=None):
        """
        Forward pass para training/prefill.

        Args:
            x:               [B, S, D]
            bus_cache:       [B, L, 128] cache del bus (None para primera capa)
            return_aux:      si True, retorna un dict extra con routing_probs
                             y ttt_pred/target para ChimeraLosses
            inference_params: InferenceParams de mamba_ssm para chunked_prefill
                             con carry de estado SSM entre chunks. None → scan normal.

        Returns:
            (out [B,S,D], new_bus_cache)                 — si return_aux=False
            (out [B,S,D], new_bus_cache, aux_dict)       — si return_aux=True
        """
        B, S, D = x.shape
        # ── BF16 / mixed-precision: auto-cast al dtype del modelo ─────────────────
        # Permite: model.to(torch.bfloat16); model(x.float()) → se castea aquí.
        # Sin esto, RMSNorm/Linear en BF16 recibirían input FP32 → error de dtype.
        param_dtype = next(self.parameters()).dtype
        x = x.to(param_dtype)
        x_norm = self.norm(x)
        
        # Router probabilities + logits (logits para Z-loss)
        probs, router_logits = self.router(x_norm)  # [B, 3] -> 0: FAST, 1: HYBRID, 2: FULL

        # Actualizar EMA de prob_FAST para detección de colapso.
        # La EMA se actualiza solo en training (en eval el router puede ser arbitrario).
        # Comparar fast_prob_ema > 0.90 en el trainer indica colapso inminente.
        if self.training:
            with torch.no_grad():
                cur_fast = probs[:, 0].mean()
                self.fast_prob_ema.mul_(0.99).add_(cur_fast * 0.01)

        # Vistas [B, 1, 1] para broadcasting en todas las ecuaciones de mezcla.
        prob_fast   = probs[:, 0].view(B, 1, 1)
        prob_hybrid = probs[:, 1].view(B, 1, 1)
        prob_full   = probs[:, 2].view(B, 1, 1)

        # D5: complexidad per-elemento [B] en vez de un escalar promedio de batch.
        # Permite que muestras simples eviten SLR/archive aunque el batch tenga otras complejas.
        complexity_per_elem = probs[:, 1:].sum(dim=-1)        # [B]  en [0, 2]
        total_active_prob   = complexity_per_elem.max()        # escalar — gate cuando ALGUNO es complejo
        total_active_prob_mean = complexity_per_elem.mean()    # para logging / TTT-Lite
        # Umbral dinámico: más exigente cuanto mayor es la varianza intra-batch.
        # correction=0 evita el UserWarning cuando B=1 (1 grado de libertad).
        # batch_var_bonus: std intra-batch como tensor (sin .item() → sin sync CPU)
        if complexity_per_elem.numel() > 1:
            batch_var_bonus = complexity_per_elem.std(correction=0).detach()
        else:
            batch_var_bonus = complexity_per_elem.new_zeros(())
        # Umbrales dinámicos como tensores — torch.clamp preserva autograd y
        # NO introduce sync CPU, requisito para CUDA Graph capture.
        slr_threshold  = torch.clamp(0.30 - 0.05 * batch_var_bonus, min=0.25)
        arch_threshold = torch.clamp(0.50 - 0.10 * batch_var_bonus, min=0.40)

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
        if self.training and not self.graph_mode:
            with torch.enable_grad():
                mini_chunk_len = min(64, S)
                mini_chunk = x_norm[:, :mini_chunk_len].detach()

                # FP32 siempre: autograd.grad sobre un tensor BF16 acumula en
                # FP32 internamente, pero la precisión numérica del grad es mejor
                # si el tensor proxy ya es FP32 desde el principio.
                dt_bias_adapt = self.mamba2.dt_bias.detach().float().clone().requires_grad_(True)

                # Context manager: hilo-seguro, compatible con DDP, restaura en finally.
                # dt_bias NO se modifica en el scan principal que viene a continuación:
                # el override es temporal solo dentro del `with`.
                with _ttt_dt_bias_override(self.mamba2, dt_bias_adapt) as dt_param:
                    m_out_mini = self.mamba2(mini_chunk)

                pred   = m_out_mini[:, :-1]      # [B, mini-1, D]
                target = mini_chunk[:, 1:]        # [B, mini-1, D]
                # Error por token via Triton (sin materializar tensor diferencia en HBM)
                per_token_err = compute_token_errors_triton(
                    pred.detach(), target.detach()
                )  # [B, mini-1]

                # FP32 MSE: más estable que BF16 para gradientes pequeños en TTT.
                loss = F.mse_loss(pred.float(), target.float())
                _ttt_grad_to_apply = torch.autograd.grad(loss, dt_param)[0]   # FP32

                # Construir mapa de importancia para S tokens completos
                err_mean = per_token_err.mean(dim=-1, keepdim=True)  # [B, 1]
                ttt_importance = err_mean.expand(B, S).clone()        # [B, S]
                ttt_importance[:, :mini_chunk_len - 1] = per_token_err
                # Normalizar a media=1 por muestra
                ttt_importance = ttt_importance / (ttt_importance.mean(dim=-1, keepdim=True) + 1e-6)

        # --- Forward pass principal (scan con dt_bias ORIGINAL) ---
        # dt_bias NO fue modificado arriba. El Lion update se aplica POST-scan.
        # Esto garantiza equivalencia chunked ↔ full en el SSD scan de Mamba2.
        # inference_params: cuando no es None, Mamba2 lleva conv_state+ssm_state
        # entre llamadas sucesivas (chunked_prefill). Sin inference_params: scan normal.
        mamba_out = self.mamba2(x_norm, inference_params=inference_params)

        # --- TTT-LITE: gradiente POST-scan (SOLO se computa, NO se aplica) ---
        # FIX DRIFT: El update ya no se aplica en el forward.
        # Razón: aplicar Lion dentro del forward modifica dt_bias ENTRE chunks cuando
        # el trainer divide S en trozos → cada chunk ve un dt_bias distinto → el
        # output chunked diverge del output full (drift medido: 106.8%).
        # Solución: el forward SOLO computa el gradiente y lo guarda en
        # self._pending_ttt_grad. El trainer llama update_ttt_inplace() una vez
        # al final del step, garantizando que dt_bias no cambia dentro del forward.
        if _ttt_grad_to_apply is not None:
            # Guardar para que el trainer aplique explicitamente
            self._pending_ttt_grad = _ttt_grad_to_apply.detach().clone()
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

        # 2. SGR + SLR Differential Attention V2 (Triton)
        #    D5: umbral dinámico (slr_threshold).  Se calcula SLR si AL MENOS UN
        #    elemento del batch supera el umbral de complejidad; elementos simples
        #    reciben un soft-gate ~0 que zeroes su contribución SLR.
        # SLR: siempre se ejecuta (no if-tensor → CUDA Graph compatible).
        # El soft-gate slr_elem_gate zeroes la contribución para elementos FAST.
        # Coste: SLR corre siempre; ganancia: +35-55% throughput vía CUDA Graphs.
        slr_out, sgr_indices = self.slr(mamba_out, importance=ttt_importance)
        # slr_threshold es tensor → operaciones diferenciables, sin sync CPU.
        slr_elem_gate = ((complexity_per_elem - slr_threshold) /
                         (1.0 - slr_threshold + 1e-6)).clamp(0, 1).view(B, 1, 1)
        slr_out = slr_elem_gate * slr_out + (1.0 - slr_elem_gate) * mamba_out

        # 3. NativeLandmarkArchive — archivar cuando hay contenido complejo (FULL)
        # En graph_mode: maybe_archive() se OMITE del forward.
        #   Razón: maybe_archive() tiene múltiples .item() calls, buffer mutations
        #   y control flow data-dependent → incompatible con torch.compile.
        #   El trainer llama archive_deferred() post-forward, patrón idéntico a
        #   update_ttt_inplace() y update_sdtm_inplace().
        # En eager mode: se ejecuta normalmente si hay TTT importance disponible.
        if not self.graph_mode and ttt_importance is not None:
            self.archive.maybe_archive(
                scan_out=mamba_out,
                ttt_importance=ttt_importance,
                tier_probs=probs,
                sgr_indices=sgr_indices,
            )
        elif self.graph_mode:
            # Guardar datos para archive_deferred() — sin .item(), sin mutation
            self._pending_archive_data = (
                mamba_out.detach(),
                probs[:, 2].unsqueeze(1).expand(B, S).detach(),
                probs.detach(),
                sgr_indices.detach() if sgr_indices is not None else None,
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
        # graph_mode: usar retrieve_compiled() que evita .item() y loops.
        if self.graph_mode:
            retrieved = self.archive.retrieve_compiled(slr_out)        # [B, S, D]
        else:
            retrieved = self.archive.retrieve(slr_out)                 # [B, S, D]
        lm_delta     = retrieved - slr_out                             # delta de landmarks
        slr_out      = slr_out + compress_ctx + arch_gate * lm_delta

        # 4. SDTM — Surprise-Driven Dual-Timescale Memory
        # READ: gated por HYBRID+FULL (tier FAST never pays read cost)
        sdtm_out = self.sdtm.read(slr_out)                    # [B, S, D]
        sdtm_gate = (prob_hybrid + prob_full).clamp(0, 1)      # [B, 1, 1]
        slr_out = slr_out + sdtm_gate * sdtm_out

        # WRITE: compute closed-form gradient from surprise signal
        # En graph_mode, compute_write() y update_usage() se omiten:
        #   compute_write() tiene prob_full_mean.item() → graph break
        #   El trainer llama archive_deferred() que también maneja SDTM.
        if not self.graph_mode and ttt_importance is not None:
            self.sdtm.compute_write(
                x_norm, ttt_importance,
                prob_full_mean=probs[:, 2].mean(),
            )
            # Update usage tracking for decay
            with torch.no_grad():
                z_usage = F.gelu(self.sdtm.W_enc(slr_out.detach()))
                self.sdtm.update_usage(z_usage)

        # --- Soft-gating por tier ---
        # FAST   → solo Mamba2 (sin overhead SLR/archive)
        # HYBRID → fusión SLR+Mamba2 (0.7/0.3: SLR dominante para gradiente más fuerte)
        # FULL   → SLR + archive retrieval + SDTM (máxima información)
        fast_representation   = mamba_out
        hybrid_representation = 0.7 * slr_out + 0.3 * mamba_out
        full_representation   = slr_out

        out = (prob_fast   * fast_representation +
               prob_hybrid * hybrid_representation +
               prob_full   * full_representation)

        out, new_cache = self.bus(out, bus_cache)
        final = x + out

        if return_aux:
            # En graph_mode: NUNCA llamar .item() ni float() sobre tensores CUDA.
            # Causa graph break + guard dinámico → recompilación cada step.
            # Pasar tensores directamente; el consumer los convierte fuera del grafo.
            _slr_dn = (slr_out - mamba_out).detach().norm(dim=-1).mean()
            aux = {
                'routing_probs':      probs.detach(),         # [B, 3] solo para logging
                'routing_probs_grad': probs,                  # [B, 3] con gradiente → routing loss
                'router_logits':      router_logits,          # [B, 3] para Z-loss
                'ttt_importance':     ttt_importance,         # [B, S] o None
                'ttt_active':         ttt_importance is not None,
                'sdtm_stats':         self.sdtm.memory_stats() if not self.graph_mode else {},
                'slr_delta_norm':     _slr_dn,                # tensor — convertir a float fuera del grafo
            }
            return final, new_cache, aux
        return final, new_cache


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
# CUDA Graph Pool — Pool de grafos por batch_size para serving variable
# ─────────────────────────────────────────────────────────────────────────────

class CUDAGraphPool:
    """
    Pool de CUDA Graphs pre-capturados para batch_sizes variables de decode.

    PROBLEMA: make_cuda_graph_step() captura UN ÚNICO batch_size. En serving
    real (vLLM-style continuous batching), el número de requests activos varía
    dinámicamente (1, 2, 4, 8...). Reutilizar un graph de B=1 con B=4 produce
    resultados incorrectos o errores de forma.

    SOLUCIÓN: pre-capturar un CUDAGraph por batch_size soportado y seleccionar
    en runtime. Cada graph tiene sus propios tensores estáticos independientes.

    Uso típico:
        # Post-prefill, antes de decode:
        pool = CUDAGraphPool(layer, batch_sizes=[1, 2, 4, 8])
        caches = {B: pool.allocate_cache(B) for B in pool.batch_sizes}

        # Durante decode con B variable:
        out, caches[B] = pool.step(x_tok, caches[B])

    Complejidad de memoria:
        VRAM extra = sum(cache_size(Bi)) para todos Bi en batch_sizes.
        cache_size(B): ~0.27 MB × B × n_layers → para B=8: ~2.2 MB/layer.

    Comportamiento con B no en pool:
        Si B ∈ [1,2,4,8] y B=3, usa el graph de B=4 con padding (rows 0:3 activas).
    """

    def __init__(
        self,
        layer: 'AdvancedChimeraLayer',
        batch_sizes: list = None,
        device=None,
        ring_size: int = None,
        warmup_iters: int = None,
    ):
        """
        Captura CUDA Graphs para todos los batch_sizes dados.

        Args:
            layer:        AdvancedChimeraLayer en eval() mode, post-prefill.
            batch_sizes:  Batch sizes a soportar. None → adaptativo al hardware.
            device:       CUDA device. None → detecta desde layer.
            ring_size:    Ancho del ring buffer. None → GPU profile.
            warmup_iters: Iteraciones warmup antes de capture. None → GPU profile.
        """
        layer.eval()
        device = device or next(layer.parameters()).device

        try:
            prof = _get_gpu_profile()
        except Exception:
            prof = None

        if batch_sizes is None:
            # Adaptativo: H200 (VRAM ≥ 40GB) → hasta B=16; laptops → hasta B=4
            if prof and prof.vram_gb >= 40:
                batch_sizes = [1, 2, 4, 8, 16]
            elif prof and prof.vram_gb >= 12:
                batch_sizes = [1, 2, 4, 8]
            else:
                batch_sizes = [1, 2, 4]   # RTX 4050 Laptop (6.4 GB)

        if ring_size is None:
            ring_size = prof.ring_size if prof else 16
        if warmup_iters is None:
            warmup_iters = prof.graph_warmup_iters if prof else 3

        self.layer       = layer
        self.batch_sizes = sorted(set(int(b) for b in batch_sizes))
        self.ring_size   = ring_size
        self._graphs:         dict = {}
        self._static_xs:      dict = {}
        self._static_outs:    dict = {}
        self._static_caches:  dict = {}

        for B in self.batch_sizes:
            self._capture_for_batch(B, layer, device, ring_size, warmup_iters)

    @torch.no_grad()
    def _capture_for_batch(
        self, B: int, layer: 'AdvancedChimeraLayer',
        device, ring_size: int, warmup_iters: int,
    ):
        """Captura el CUDA Graph para un batch_size específico."""
        D = layer.d_model
        static_x     = torch.zeros(B, 1, D, device=device)
        static_cache = layer.allocate_inference_cache(B, ring_size=ring_size)

        # Pre-computar delta de archive con query=0
        dummy_x = torch.zeros(B, 1, D, device=device)
        try:
            arc_full  = layer.archive.retrieve(dummy_x)
            arc_cmprx = layer.archive.get_compress_ctx(dummy_x)
            arc_delta = (arc_full - dummy_x) + arc_cmprx
        except Exception:
            arc_delta = torch.zeros(B, 1, D, device=device)
        static_cache['_archive_ctx'] = arc_delta.clone()

        # Warmup: fuerza Triton autotune + cuDNN benchmark
        for _ in range(warmup_iters):
            static_x.normal_()
            layer.step(static_x, static_cache)
        static_x.zero_()

        # Captura en stream privado
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out, static_cache = layer.step(static_x, static_cache)

        self._graphs[B]        = g
        self._static_xs[B]     = static_x
        self._static_outs[B]   = static_out
        self._static_caches[B] = static_cache

    def allocate_cache(self, batch_size: int, dtype=None) -> dict:
        """
        Aloca un cache de inferencia para el batch_size dado.
        Usa el ring_size del pool para garantizar compatibilidad con los graphs.
        """
        return self.layer.allocate_inference_cache(
            batch_size, ring_size=self.ring_size, dtype=dtype
        )

    def _select_batch_size(self, B: int) -> int:
        """Selecciona el graph más pequeño con batch_size ≥ B (padding mínimo)."""
        for bs in self.batch_sizes:
            if bs >= B:
                return bs
        return self.batch_sizes[-1]   # B > max → usa el máximo (con padding)

    @torch.no_grad()
    def step(self, x: torch.Tensor, cache: dict):
        """
        Ejecuta decode usando el CUDA Graph correspondiente a x.shape[0].

        Si x.shape[0] = B no está exactamente en el pool, usa el próximo
        batch_size disponible ≥ B (padding automático). Solo los primeros B
        elementos producen output válido.

        Args:
            x:     [B, 1, D]  — token de decode actual
            cache: dict de allocate_cache()  — actualizado in-place
        Returns:
            (out [B, 1, D], updated_cache)
        """
        B = x.shape[0]
        Bg = self._select_batch_size(B)

        sx     = self._static_xs[Bg]
        sout   = self._static_outs[Bg]
        scache = self._static_caches[Bg]

        # Copiar solo filas activas [0:B]; filas [B:Bg] son padding (ceros)
        sx[:B].copy_(x)
        if B < Bg:
            sx[B:].zero_()   # padding limpio

        self._graphs[Bg].replay()

        # Sincronizar cache del caller con el cache estático
        cache['conv_state'].copy_(scache['conv_state'][:B])
        cache['ssm_state'].copy_(scache['ssm_state'][:B])
        cache['bus_ring'].copy_(scache['bus_ring'][:B])

        return sout[:B].clone(), cache

    def __repr__(self) -> str:
        return (
            f"CUDAGraphPool("
            f"batch_sizes={self.batch_sizes}, "
            f"ring_size={self.ring_size}, "
            f"graphs={list(self._graphs.keys())})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Chunked Prefill con Carry SSM — elimina drift T5 en secuencias largas
# ─────────────────────────────────────────────────────────────────────────────

class Mamba2ChunkedPrefill:
    """
    Carry correcto de conv_state + SSM state entre chunks de prefill.

    Usa las APIs de estado explícito disponibles en mamba_ssm≥2.0:
      causal_conv1d_fn(initial_states, return_final_states)
      mamba_chunk_scan_combined(initial_states, return_final_states)

    Esto elimina el drift SSM entre chunks que existía en la versión
    anterior (bus carry solo, sin SSM carry).

    Uso:
        chunker = Mamba2ChunkedPrefill(layer.mamba2)
        conv_s, ssm_s = chunker.init_states(batch_size)
        for chunk in chunks:
            out, conv_s, ssm_s = chunker.forward_chunk(chunk, conv_s, ssm_s)
    """

    def __init__(self, mamba2: nn.Module):
        self.m = mamba2

    def init_states(self, batch_size: int):
        """Estados cero para inicio de secuencia.
        conv_state: [B, d_conv_dim, d_conv-1] con stride(1)==1
                    (requerido por causal_conv1d_fn initial_states API)
        ssm_state:  [B, nheads, headdim, d_state]
        """
        m   = self.m
        dev = m.in_proj.weight.device
        dt  = m.in_proj.weight.dtype
        # causal_conv1d_fn necesita stride(1)==1 en initial_states.
        # Creamos [B, d_conv-1, d] y transponemos → [B, d, d_conv-1] con stride(1)=1.
        conv_state = torch.zeros(
            batch_size, m.d_conv - 1, m.conv1d.weight.shape[0],
            device=dev, dtype=dt).transpose(1, 2)           # no-contiguous, stride(1)=1
        ssm_state = torch.zeros(
            batch_size, m.nheads, m.headdim, m.d_state,
            device=dev, dtype=dt)
        return conv_state, ssm_state

    def forward_chunk(self, x: torch.Tensor,
                      conv_state: torch.Tensor,
                      ssm_state: torch.Tensor):
        """
        Forward de un chunk con carry de estado.

        x          : [B, L, D]
        conv_state : [B, d_conv_dim, d_conv-1]  — lastre del conv causal
        ssm_state  : [B, nheads, headdim, d_state] — estado SSM final del chunk anterior

        Returns: (out [B, L, D], new_conv_state, new_ssm_state)
        Gradientes fluyen correctamente (no usa torch.no_grad()).
        """
        m   = self.m
        A   = -torch.exp(m.A_log.float())
        dt_lim = {} if m.dt_limit == (0.0, float("inf")) else {"dt_limit": m.dt_limit}

        zxbcdt = m.in_proj(x)   # [B, L, d_in_proj]

        d_mlp = (zxbcdt.shape[-1] - 2*m.d_ssm
                 - 2*m.ngroups*m.d_state - m.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, m.d_ssm, m.d_ssm + 2*m.ngroups*m.d_state, m.nheads],
            dim=-1,
        )

        # Causal conv con carry de estado inicial
        xBC_t, new_conv_state = _causal_conv1d_fn(
            xBC.transpose(1, 2),                          # [B, d, L]
            _rearrange(m.conv1d.weight, "d 1 w -> d w"),  # [d, d_conv]
            bias=m.conv1d.bias,
            initial_states=conv_state,                    # [B, d, d_conv-1]
            return_final_states=True,
            activation=m.activation,
        )
        xBC = xBC_t.transpose(1, 2)   # [B, L, d]

        x_s, B_s, C_s = torch.split(
            xBC, [m.d_ssm, m.ngroups*m.d_state, m.ngroups*m.d_state], dim=-1)

        D_arg = (_rearrange(m.D, "(h p) -> h p", p=m.headdim)
                 if m.D_has_hdim else m.D)
        z_arg = (_rearrange(z, "b l (h p) -> b l h p", p=m.headdim)
                 if not m.rmsnorm else None)

        # SSD scan con carry de estado SSM inicial
        y, new_ssm_state = _mamba_scan(
            _rearrange(x_s, "b l (h p) -> b l h p", p=m.headdim),
            dt, A,
            _rearrange(B_s, "b l (g n) -> b l g n", g=m.ngroups),
            _rearrange(C_s, "b l (g n) -> b l g n", g=m.ngroups),
            chunk_size=m.chunk_size,
            D=D_arg,
            z=z_arg,
            dt_bias=m.dt_bias,
            dt_softplus=True,
            initial_states=ssm_state,    # ← carry SSM del chunk anterior
            return_final_states=True,    # ← extraer estado para el siguiente
            **dt_lim,
        )

        y = _rearrange(y, "b l h p -> b l (h p)")
        if m.rmsnorm:
            y = m.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = m.out_proj(y)
        return out, new_conv_state, new_ssm_state


@contextlib.contextmanager
def _mamba2_carry_ctx(layer: 'AdvancedChimeraLayer', batch_size: int):
    """
    Context manager que parchea layer.mamba2.forward con una versión
    con estado (conv + SSM carry) mientras dura el contexto.
    Al salir restaura el forward original.

    Uso:
        with _mamba2_carry_ctx(layer, B):
            for chunk in chunks:
                out, bus_cache = layer(chunk, bus_cache=bus_cache)
    """
    if not _MAMBA_SCAN_AVAIL:
        # Sin las librerías de carry, simplemente no parchamos
        yield
        return

    chunker = Mamba2ChunkedPrefill(layer.mamba2)
    state   = list(chunker.init_states(batch_size))   # [conv_state, ssm_state]
    orig_fwd = layer.mamba2.forward

    def _stateful_fwd(x, inference_params=None):
        out, state[0], state[1] = chunker.forward_chunk(x, state[0], state[1])
        return out

    layer.mamba2.forward = _stateful_fwd
    try:
        yield
    finally:
        layer.mamba2.forward = orig_fwd


def chunked_prefill(
    layer: 'AdvancedChimeraLayer',
    x: torch.Tensor,
    chunk_size: int = None,
    bus_cache=None,
    return_aux: bool = False,
):
    """
    Prefill de secuencias largas (>chunk_size tokens) con VRAM constante
    y carry correcto de conv_state + SSM state entre chunks.

    FUENTES DEL DRIFT T5 (+108%) — ambas resueltas:
      1. Bus cache: resuelto llevando bus_cache entre chunks (antes).
      2. SSM state: resuelto con Mamba2ChunkedPrefill + _mamba2_carry_ctx.
         causal_conv1d_fn(initial_states=conv_state) propaga el estado
         de la conv causal entre chunks.
         mamba_chunk_scan_combined(initial_states=ssm_state) propaga
         el estado SSM del recurrente entre chunks.

    Requiere mamba_ssm≥2.0 y causal_conv1d≥1.4 (initial_states API).
    Si no están disponibles (_MAMBA_SCAN_AVAIL=False) usa solo bus carry.

    VRAM: O(chunk_size), nunca O(S).
      1M tokens = 245 chunks × 4.2 MB = constante en 6 GB.

    chunk_size defaults: RTX 4050=2048, A100=8192, H200=16384.

    Returns:
        (out [B, S, D], bus_cache_final)            — return_aux=False
        (out [B, S, D], bus_cache_final, aux_list)  — return_aux=True
            aux_list: list[dict], un dict por chunk (1 elemento si S<=chunk_size)
    """
    B, S, _ = x.shape

    if chunk_size is None:
        try:
            chunk_size = _get_gpu_profile().chunk_size_default
        except Exception:
            chunk_size = 4096
    chunk_size = max(chunk_size, 32)

    # Secuencias cortas: bypass directo
    if S <= chunk_size:
        if return_aux:
            out, new_cache, aux = layer(x, bus_cache=bus_cache, return_aux=True)
            return out, new_cache, [aux]   # lista de un elemento → API uniforme
        return layer(x, bus_cache=bus_cache)

    outputs  = []
    aux_list = [] if return_aux else None

    # _mamba2_carry_ctx parchea layer.mamba2.forward para llevar conv+SSM state
    with _mamba2_carry_ctx(layer, B):
        for start in range(0, S, chunk_size):
            end   = min(start + chunk_size, S)
            chunk = x[:, start:end]
            if return_aux:
                out, bus_cache, aux = layer(chunk, bus_cache=bus_cache, return_aux=True)
                aux_list.append(aux)
            else:
                out, bus_cache = layer(chunk, bus_cache=bus_cache)
            outputs.append(out)

    combined = torch.cat(outputs, dim=1)
    if return_aux:
        return combined, bus_cache, aux_list
    return combined, bus_cache


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
    print(f"SLR merge gate: {slr.merge_gate.sigmoid().item():.4f}")
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


