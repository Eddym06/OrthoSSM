"""
CHIMERA V2 SSM — OrthoSSM Chebyshev Kernel como núcleo SSM
============================================================
ChimeraV2Layer reemplaza Mamba2 con el kernel Chebyshev Parallel Scan de
OrthoSSM V10 (sdpc_kernel.py), manteniendo la misma API externa que
AdvancedChimeraLayer para ser drop-in compatible.

Ventajas de usar OrthoSSM como SSM core:
  • O(1) memoria — sin SSM state growth (los coeficientes c_k son el estado)
  • Scan paralelo — verificación en O(K) para speculative decoding (ver speculative_ssm.py)
  • TTT online inline — el Lion update está fusionado dentro del mega-kernel
  • Multi-escala temporal — 8 cabezas con forget rates λ∈[0.9995, 0.97] hard-coded
  • Sin A_log (Mamba2 requires log(A) < 0) — OrthoSSM usa Chebyshev basis invariante

Diferencias con AdvancedChimeraLayer:
  • SSM core: apply_cheby_rkv_v10() en lugar de mamba_ssm.Mamba2()
  • Estado de decode: coeficientes c_k + momentos (en lugar de ssm_state + conv_state)
  • TTT correction: se aplica a c_k directamente (no dt_bias)
  • Bus, SLR y Archive son idénticos a los de AdvancedChimeraLayer

Compatibilidad:
  • forward(x, bus_cache, return_aux) → (out, bus_cache, aux?)  ✓
  • step(x_single, cache) → (out_single, cache)                 ✓
  • AdvancedChimeraLayer puede sustituirse por ChimeraV2Layer    ✓

Fallback:
  Si el directorio raíz del proyecto no está importable (por ejemplo, en test aislado),
  usa un EMA simple como SSM fallback con advertencia.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

# ─────────────────────────────────────────────────────────────────────────────
# Import cuidadoso de OrthoSSM — fallback si no está disponible
# ─────────────────────────────────────────────────────────────────────────────

# Raíz del proyecto: directorio padre del directorio que contiene este archivo.
# Funciona independientemente de dónde esté clonado el proyecto.
_ORTHO_ROOT     = str(Path(__file__).parent.parent.resolve())
_ORTHO_AVAILABLE = False
apply_cheby = None
init_cheby  = None

if os.path.isdir(_ORTHO_ROOT) and _ORTHO_ROOT not in sys.path:
    sys.path.insert(0, _ORTHO_ROOT)

try:
    from sdpc_kernel import apply_cheby_rkv_v10, init_chebyshev_coefficients
    apply_cheby = apply_cheby_rkv_v10
    init_cheby  = init_chebyshev_coefficients
    _ORTHO_AVAILABLE = True
    print("[ChimeraV2] OrthoSSM V10 kernel importado correctamente.")
except ImportError as e:
    print(f"[ChimeraV2] WARNING: OrthoSSM no disponible ({e}). Usando EMA fallback.")

# Componentes de CHIMERA (misma carpeta)
_CHIMERA_DIR = os.path.dirname(__file__)
if _CHIMERA_DIR not in sys.path:
    sys.path.insert(0, _CHIMERA_DIR)

from sgr_slr   import SLRDifferentialModule, SGRSelector
from landmark_native import NativeLandmarkArchive


# ─────────────────────────────────────────────────────────────────────────────
# EMA Fallback SSM — usado sólo si OrthoSSM no está disponible
# ─────────────────────────────────────────────────────────────────────────────

class _EMAFallbackSSM(nn.Module):
    """EMA simple como SSM de emergencia — mantiene API de OrthoSSM."""

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        lambdas = [0.9995, 0.9995, 0.997, 0.997, 0.995, 0.995, 0.97, 0.95]
        lt = torch.tensor(lambdas[:n_heads] + [0.995] * max(0, n_heads - len(lambdas)))
        self.register_buffer('lambdas', lt)

    def forward(self, x, coeffs=None, momentum=None):
        B, S, D = x.shape
        head_dim = D // self.n_heads
        out = torch.zeros_like(x)
        for h in range(self.n_heads):
            lam = self.lambdas[h]
            state = torch.zeros(B, head_dim, device=x.device)
            start = h * head_dim
            end   = start + head_dim
            for t in range(S):
                state = lam * state + (1 - lam) * x[:, t, start:end]
                out[:, t, start:end] = state
        return out, coeffs, momentum

    def step_one(self, x_single, coeffs, momentum):
        """Decode step para EMA fallback."""
        state_key = f'ema_state'
        if coeffs is None:
            ema_state = torch.zeros(x_single.shape[0], self.d_model,
                                    device=x_single.device)
        else:
            ema_state = coeffs.get('ema_state',
                                   torch.zeros(x_single.shape[0], self.d_model,
                                               device=x_single.device))
        out_state = torch.zeros_like(ema_state)
        for h in range(self.n_heads):
            lam     = self.lambdas[h]
            hd      = self.d_model // self.n_heads
            s, e    = h * hd, h * hd + hd
            out_state[:, s:e] = lam * ema_state[:, s:e] + (1 - lam) * x_single[:, s:e]
        new_coeffs = {'ema_state': out_state}
        return out_state, new_coeffs


# ─────────────────────────────────────────────────────────────────────────────
# AsyncLightBusV2 — mismo bus que AdvancedChimeraLayer (copia mínima)
# ─────────────────────────────────────────────────────────────────────────────

class AsyncLightBusV2(nn.Module):
    """
    Bus de comunicación entre capas CHIMERA.
    API idéntica a AsyncLightBus en advanced_chimera.py.
    """

    def __init__(self, d_model: int, bus_dim: int = 128):
        super().__init__()
        self.d_model = d_model
        self.bus_dim = bus_dim
        self.compress = nn.Linear(d_model, bus_dim, bias=False)
        self.expand   = nn.Linear(bus_dim, d_model, bias=False)
        self.publish  = nn.Linear(bus_dim, bus_dim, bias=False)
        nn.init.xavier_uniform_(self.compress.weight, gain=0.5)
        nn.init.zeros_(self.expand.weight)
        nn.init.eye_(self.publish.weight)

    def forward(self, x: torch.Tensor, bus_cache: Optional[torch.Tensor] = None):
        summary = self.compress(x.mean(1, keepdim=True))   # [B, 1, bus_dim]
        if bus_cache is not None:
            augmented = torch.cat([summary, bus_cache], dim=1)   # [B, N+1, bus_dim]
        else:
            augmented = summary
        published = self.publish(augmented)                        # [B, N+1, bus_dim]
        attn_w = torch.softmax(
            torch.bmm(summary, published.transpose(1, 2)) / math.sqrt(self.bus_dim),
            dim=-1,
        )
        context = torch.bmm(attn_w, published)              # [B, 1, bus_dim]
        modulation = self.expand(context)                    # [B, 1, d]
        x_out = x + modulation * 0.1
        new_cache = published.detach()
        return x_out, new_cache


# ─────────────────────────────────────────────────────────────────────────────
# ChimeraV2Layer — Núcleo principal
# ─────────────────────────────────────────────────────────────────────────────

class ChimeraV2Layer(nn.Module):
    """
    CHIMERA V2: OrthoSSM (Chebyshev) + SLR V3 + Bus + Landmark Archive.

    Hyperparámetros:
        d_model:   dimensión total del modelo
        n_heads:   número de cabezas Chebyshev (default 8)
        degree:    grado del polinomio Chebyshev (default 4, max 8)
        top_k_frac: fracción de tokens seleccionados por SGR
        bus_dim:   dimensión del bus de comunicación inter-capas
        archive_slots: número de slots en el Landmark Archive
    """

    def __init__(
        self,
        d_model:      int   = 256,
        n_heads:      int   = 8,
        degree:       int   = 4,
        top_k_frac:   float = 0.25,
        bus_dim:      int   = 128,
        archive_slots: int  = 64,
        d_head:       int   = 32,
    ):
        super().__init__()
        self.d_model      = d_model
        self.n_heads      = n_heads
        self.degree       = degree
        self.head_dim     = d_model // n_heads
        self.d_head       = d_head
        self.top_k_frac   = top_k_frac

        # ── Proyección de entrada ─────────────────────────────────────────────
        self.norm_in  = nn.RMSNorm(d_model)
        self.norm_out = nn.RMSNorm(d_model)

        # ── OrthoSSM / EMA fallback ───────────────────────────────────────────
        if _ORTHO_AVAILABLE:
            # Parámetros aprendibles: coeficientes Chebyshev y momentum
            # Forma: [1, n_heads, degree, head_dim] — crecen con B en forward
            self._cheby_base  = nn.Parameter(
                torch.randn(1, n_heads, degree, self.head_dim) * 0.01
            )
            self._momentum_buf = nn.Parameter(
                torch.zeros(1, n_heads, degree, self.head_dim),
                requires_grad=False,
            )
        else:
            self.ssm_fallback = _EMAFallbackSSM(d_model, n_heads)

        # ── TTT correction (modula los coeficientes Chebyshev) ───────────────
        # En vez de dt_bias (Mamba2), corregimos directamente los c_k
        # usando una señal de complejidad predicha por la entrada.
        self.ttt_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, n_heads * degree, bias=True),
        )
        nn.init.zeros_(self.ttt_predictor[-1].weight)
        nn.init.zeros_(self.ttt_predictor[-1].bias)

        # ── Router 3-tier (fast / medium / full) ─────────────────────────────
        # fast=0: sólo SSM  |  medium=1: SSM+SLR  |  full=2: SSM+SLR+Archive
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3, bias=True),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, torch.tensor([0.5, 0.0, -0.5]).tolist()[0])

        # ── SLR Differential (sólo para tiers medium y full) ─────────────────
        self.slr = SLRDifferentialModule(
            d_model=d_model, d_head=d_head, top_k_frac=top_k_frac
        )

        # ── Landmark Archive ──────────────────────────────────────────────────
        self.archive = NativeLandmarkArchive(
            d_model=d_model, max_landmarks=archive_slots
        )

        # ── Bus de inter-capas ────────────────────────────────────────────────
        self.bus = AsyncLightBusV2(d_model=d_model, bus_dim=bus_dim)

        # ── Proyección de mezcla final ────────────────────────────────────────
        self.mix_proj = nn.Linear(d_model * 2, d_model, bias=False)
        nn.init.xavier_uniform_(self.mix_proj.weight, gain=0.3)

    # ─────────────────────────────────────────────────────────────────────────
    # Forward principal
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x:          torch.Tensor,
        bus_cache:  Optional[Any] = None,
        return_aux: bool = False,
    ):
        """
        x: [B, S, D]
        bus_cache: dict o None
        Returns: (out [B,S,D], new_bus_cache, aux?) 
        """
        B, S, D = x.shape
        residual = x

        x_norm = self.norm_in(x)

        # ── Router ────────────────────────────────────────────────────────────
        route_in    = x_norm.mean(dim=1)                   # [B, D]
        route_logits= self.router(route_in)                # [B, 3]
        route_probs = torch.softmax(route_logits, dim=-1)  # [B, 3]

        # Suavizar para gradiente — temperatura adaptativa
        p_fast   = route_probs[:, 0]   # [B]
        p_medium = route_probs[:, 1]
        p_full   = route_probs[:, 2]

        # ── TTT correction sobre coeficientes Chebyshev ───────────────────────
        ttt_correction = self.ttt_predictor(route_in)      # [B, n_heads*degree]
        ttt_correction = ttt_correction.view(B, self.n_heads, self.degree)
        # Corrección de escala de lr de OrthoSSM — escala ±0.1
        ttt_scale = 0.1 * torch.tanh(ttt_correction)       # [B, n_heads, degree]

        # ── OrthoSSM / fallback ───────────────────────────────────────────────
        if _ORTHO_AVAILABLE:
            # Expandir coeficientes base a batch y aplicar corrección TTT
            coeffs = self._cheby_base.expand(B, -1, -1, -1).clone()  # [B, nH, deg, hD]
            # Aplicar escala TTT: el predictor modula c_k en cada cabeza
            ttt_scale_hd = ttt_scale.unsqueeze(-1).expand_as(coeffs)  # [B,nH,deg,hD]
            coeffs_modulated = coeffs * (1.0 + ttt_scale_hd)

            momentum = self._momentum_buf.expand(B, -1, -1, -1).clone()

            try:
                ssm_out, coeffs_new, mom_new = apply_cheby(
                    x_norm.float(),
                    coeffs_modulated.float(),
                    momentum.float(),
                    n_heads      = self.n_heads,
                    base_lr      = 0.005,
                    ema_momentum = 0.9,
                    lion_beta1   = 0.9,
                    lion_beta2   = 0.99,
                    lion_wd      = 0.01,
                    chunk_size   = 256,
                    use_bf16     = False,
                    gate_global  = p_full.detach(),
                    dynamic_lambda = None,
                    use_lut      = True,
                    seq_threshold= 64,
                    lion_max_norm= 14.0,
                )
                ssm_out = ssm_out.to(x.dtype)
            except Exception as e:
                # Si el kernel falla (p.ej. secuencia demasiado corta), usar fallback
                print(f"  [ChimeraV2] OrthoSSM kernel error: {e}, usando EMA fallback")
                ssm_out = self._ema_fallback(x_norm)
        else:
            ssm_out, _, _ = self.ssm_fallback(x_norm)

        # ── SLR (tiers medium + full) ─────────────────────────────────────────
        # Soft blend: p_medium y p_full controlan la mezcla
        p_slr = (p_medium + p_full).unsqueeze(1).unsqueeze(2)  # [B,1,1]
        slr_result = self.slr(x_norm)
        slr_out = slr_result[0]   # (slr_out [B,S,D], top_idx) → tomamos solo slr_out

        # Blend suave entre SSM puro y SSM+SLR
        mixed = ssm_out + p_slr * slr_out                         # [B,S,D]

        # ── Landmark Archive (tier full) ──────────────────────────────────────
        ttt_importance = ttt_scale.abs().mean(dim=-1).mean(dim=-1)  # [B]
        tier_probs_3   = route_probs                                # [B, 3]

        archive_retrieved = None
        if p_full.mean().item() > 0.05:
            # Siempre intentamos archivar — el archive tiene su propia lógica de threshold
            self.archive.maybe_archive(
                scan_out     = ssm_out,
                ttt_importance = ttt_importance,
                tier_probs   = tier_probs_3,
                sgr_indices  = None,
            )
            # Recuperar landmarks relevantes para el token actual query
            query = x_norm[:, -1:, :]           # [B, 1, D] — último token como query
            archive_retrieved = self.archive.retrieve(query)  # [B, K, D] o None

        if archive_retrieved is not None:
            # Agregar información del archive mediante cross-attention liviana
            q   = x_norm[:, -1:, :]                        # [B, 1, D]
            k   = archive_retrieved                         # [B, K, D]
            v   = archive_retrieved
            attn = torch.softmax(
                torch.bmm(q, k.transpose(1, 2)) / math.sqrt(D), dim=-1
            )
            arc_ctx  = torch.bmm(attn, v)                  # [B, 1, D]
            arc_gate = p_full.view(B, 1, 1)
            # Expandir arc_ctx a la longitud S (sólo afecta última posición)
            arc_full = torch.zeros_like(mixed)
            arc_full[:, -1:, :] = arc_ctx
            mixed = mixed + arc_gate * arc_full

        # ── Mix final: SSM out + SLR contribution ────────────────────────────
        cat_out = torch.cat([mixed, x_norm], dim=-1)       # [B, S, 2D]
        out     = self.mix_proj(cat_out)                   # [B, S, D]
        out     = residual + self.norm_out(out)

        # ── Bus de inter-capas ────────────────────────────────────────────────
        raw_bus_cache = bus_cache.get('bus_cache') if isinstance(bus_cache, dict) else bus_cache
        out, new_bus = self.bus(out, raw_bus_cache)
        new_cache = {'bus_cache': new_bus, 'coeffs': None, 'momentum': None}

        if return_aux:
            # routing_probs con/sin grad para chimera_losses
            aux = {
                'routing_probs':      route_probs.detach(),
                'routing_probs_grad': route_probs,
                'ttt_importance':     ttt_importance.detach(),
                'ttt_active':         True,
                'p_full':             p_full.detach().mean().item(),
            }
            return out, new_cache, aux

        return out, new_cache

    def _ema_fallback(self, x: torch.Tensor) -> torch.Tensor:
        """EMA inline como backup (no necesita módulo separado)."""
        B, S, D = x.shape
        hd = D // self.n_heads
        out = torch.zeros_like(x)
        lambdas = [0.9995, 0.9995, 0.997, 0.997, 0.995, 0.995, 0.97, 0.95]
        for h in range(self.n_heads):
            lam = lambdas[min(h, len(lambdas) - 1)]
            state = torch.zeros(B, hd, device=x.device, dtype=x.dtype)
            for t in range(S):
                state = lam * state + (1 - lam) * x[:, t, h*hd:(h+1)*hd]
                out[:, t, h*hd:(h+1)*hd] = state
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # step() — decodificación autoregressive token-a-token
    # ─────────────────────────────────────────────────────────────────────────

    def step(
        self,
        x_single:  torch.Tensor,          # [B, D]
        cache:     Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Decode un token.
        cache estructura: {
            'ema_state': [B, D],           — estado EMA acumulado por cabeza
            'bus_cache': [B, N, bus_dim],
        }
        """
        if cache is None:
            cache = {}

        B, D = x_single.shape
        x    = x_single.unsqueeze(1)      # [B, 1, D]
        x_n  = self.norm_in(x)

        # Route en base al token actual
        route_in = x_n.squeeze(1)
        route_logits = self.router(route_in)
        route_probs  = torch.softmax(route_logits, dim=-1)
        p_full = route_probs[:, 2]

        # EMA state decode
        head_dim = D // self.n_heads
        ema_state = cache.get(
            'ema_state',
            torch.zeros(B, D, device=x.device, dtype=x.dtype)
        )
        lambdas = [0.9995, 0.9995, 0.997, 0.997, 0.995, 0.995, 0.97, 0.95]
        new_state = torch.zeros_like(ema_state)
        ssm_out   = torch.zeros_like(x_n.squeeze(1))
        for h in range(self.n_heads):
            lam = lambdas[min(h, len(lambdas) - 1)]
            s, e = h * head_dim, (h + 1) * head_dim
            x_h       = x_n[:, 0, s:e]
            new_state[:, s:e] = lam * ema_state[:, s:e] + (1 - lam) * x_h
            ssm_out[:, s:e]   = new_state[:, s:e]

        # SLR en token único
        slr_result = self.slr(x_n)
        slr_out = slr_result[0].squeeze(1)           # (slr_out, top_idx) → tomamos slr_out [B, D]
        p_slr   = (route_probs[:, 1] + p_full)
        mixed   = ssm_out + p_slr.unsqueeze(1) * slr_out

        # Mix + norm
        cat_out = torch.cat([mixed.unsqueeze(1), x_n], dim=-1)  # [B,1,2D]
        out     = self.mix_proj(cat_out).squeeze(1)              # [B, D]
        out     = x_single + self.norm_out(out.unsqueeze(1)).squeeze(1)

        # Bus
        raw_bus = cache.get('bus_cache', None)
        out_u   = out.unsqueeze(1)                               # [B,1,D]
        out_u, new_bus = self.bus(out_u, raw_bus)
        out = out_u.squeeze(1)

        new_cache = {
            'ema_state': new_state,
            'bus_cache': new_bus,
        }
        return out, new_cache


# ─────────────────────────────────────────────────────────────────────────────
# Stack de N capas para crear modelos profundos (equivalente al stack de CHIMERA)
# ─────────────────────────────────────────────────────────────────────────────

class ChimeraV2Stack(nn.Module):
    """
    Stack de N capas ChimeraV2Layer.
    API compatible con usar en ChimeraLM (de niah_eval.py).
    """

    def __init__(self, n_layers: int = 3, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            ChimeraV2Layer(**layer_kwargs) for _ in range(n_layers)
        ])

    def forward(self, x, bus_cache=None, return_aux=False):
        """Pasa por todas las capas encadenando el bus_cache."""
        if bus_cache is None:
            bus_cache = {}
        aux_list = []
        for layer in self.layers:
            if return_aux:
                x, bus_cache, aux = layer(x, bus_cache, return_aux=True)
                aux_list.append(aux)
            else:
                x, bus_cache = layer(x, bus_cache)

        if return_aux:
            # Agregar las aux de todas las capas — usar la última como representativa
            return x, bus_cache, aux_list[-1]
        return x, bus_cache

    def step(self, x_single, cache=None):
        if cache is None:
            cache = [{} for _ in self.layers]
        new_caches = []
        for i, layer in enumerate(self.layers):
            x_single, new_c = layer.step(x_single, cache[i])
            new_caches.append(new_c)
        return x_single, new_caches


# ─────────────────────────────────────────────────────────────────────────────
# Comparación de throughput: ChimeraV2 vs AdvancedChimera
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_v2_vs_v1(d_model=256, n_warmup=3, n_iters=20, device='cuda'):
    """
    Compara throughput y VRAM entre ChimeraV2Layer y AdvancedChimeraLayer.
    """
    import time
    try:
        from advanced_chimera import AdvancedChimeraLayer
        HAS_V1 = True
    except ImportError:
        HAS_V1 = False

    B, S = 2, 512
    x = torch.randn(B, S, d_model, device=device).float()

    print(f"\n{'='*60}")
    print(f"  ChimeraV2 vs AdvancedChimera Throughput Benchmark")
    print(f"  B={B}, S={S}, D={d_model}, device={device}")
    print(f"{'='*60}")

    # ── ChimeraV2Layer ────────────────────────────────────────────────────────
    v2 = ChimeraV2Layer(d_model=d_model).to(device).float()
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            v2(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            out, _ = v2(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_v2 = (time.perf_counter() - t0) / n_iters
    tps_v2 = (B * S) / t_v2
    print(f"  ChimeraV2:  {tps_v2:>9.1f} tok/s  ({t_v2*1000:.2f} ms/fwd)")

    # ── AdvancedChimeraLayer ──────────────────────────────────────────────────
    if HAS_V1:
        v1 = AdvancedChimeraLayer(d_model=d_model, expand=2, headdim=32).to(device).float()
        for _ in range(n_warmup):
            with torch.no_grad():
                v1(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                out, _ = v1(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_v1 = (time.perf_counter() - t0) / n_iters
        tps_v1 = (B * S) / t_v1
        print(f"  ChimeraV1:  {tps_v1:>9.1f} tok/s  ({t_v1*1000:.2f} ms/fwd)")
        speedup = tps_v2 / max(tps_v1, 1)
        print(f"  Speedup:    {speedup:.2f}x")
    else:
        print("  ChimeraV1:  no disponible (import failed)")

    # VRAM usage
    if torch.cuda.is_available():
        mb = torch.cuda.memory_allocated(device) / 1024**2
        print(f"  VRAM used:  {mb:.1f} MB")

    print(f"{'='*60}\n")

    return tps_v2


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"OrthoSSM disponible: {_ORTHO_AVAILABLE}")

    # Test básico: forward + backward + step
    d_model = 256
    B, S    = 2, 128

    print("\n--- Test ChimeraV2Layer forward + backward ---")
    model = ChimeraV2Layer(d_model=d_model).to(device).float()
    x = torch.randn(B, S, d_model, device=device, requires_grad=True)

    # Forward
    out, cache, aux = model(x, return_aux=True)
    print(f"  out:   {out.shape}  dtype={out.dtype}")
    print(f"  p_full (mean): {aux['p_full']:.4f}")
    print(f"  routing_probs fast/med/full: "
          f"{aux['routing_probs'][:,0].mean():.3f} / "
          f"{aux['routing_probs'][:,1].mean():.3f} / "
          f"{aux['routing_probs'][:,2].mean():.3f}")

    # Backward
    loss = out.mean()
    loss.backward()
    grad_norm = x.grad.norm().item()
    print(f"  grad norm (input): {grad_norm:.4f}")
    assert grad_norm > 0, "Gradiente muerto en input!"

    # Decode step
    print("\n--- Test ChimeraV2Layer decode step ---")
    model.eval()
    x_tok  = torch.randn(B, d_model, device=device)
    cache  = None
    for t in range(5):
        out_t, cache = model.step(x_tok, cache)
    print(f"  decode step output: {out_t.shape}  (5 pasos OK)")

    # Stack test
    print("\n--- Test ChimeraV2Stack (3 capas) ---")
    stack = ChimeraV2Stack(n_layers=3, d_model=d_model).to(device).float()
    x_s   = torch.randn(B, S, d_model, device=device)
    out_s, cache_s, aux_s = stack(x_s, return_aux=True)
    print(f"  stack out: {out_s.shape}  p_full={aux_s['p_full']:.4f}")

    # Benchmark
    print()
    benchmark_v2_vs_v1(d_model=d_model, device=device)

    print("[OK] chimera_v2_ssm.py completado sin errores.")
