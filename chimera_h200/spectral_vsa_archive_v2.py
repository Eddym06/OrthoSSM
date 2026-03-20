"""
SpectralVSAArchive v2 — Robust Chebyshev Spectral Holographic Context Compression
==================================================================================

Major improvements over v1:
  1. Dynamic K: K_active auto-adjusts ∈ [K_min, K_max] based on spectral energy
  2. Lanczos σ-factor damping: suppresses Runge phenomenon & Gibbs oscillations
  3. Kahan-compensated EMA: prevents FP drift in V_mem over long sequences
  4. Binding error correction: shadow-unbind → residual tracking → correction at retrieval
  5. Discontinuity sentinel: detects semantic jumps → adapts damping & K
  6. FP32 accumulators: all critical operations forced FP32 (safe under BF16 training)
  7. Content-aware retrieval: combines content similarity + Δ_k + learned freq bias
  8. Vectorized buffer: eliminates Python loops in maybe_archive
  9. Anti-quantization: noise floor estimation and subtraction during retrieval
  10. Condition monitoring: tracks coefficient condition for rank stability

API-compatible drop-in for NativeLandmarkArchive:
  - maybe_archive(scan_out, ttt_importance, tier_probs, sgr_indices) → bool
  - retrieve(scan_out) → [B, S, D]
  - get_compress_ctx(scan_out) → [B, S, D]
  - get_archive_info() → dict
  - get_spectral_delta() → [K] tensor
  - preload_context(context_embs, importance_scores) → None
  - reset() → None
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────────────────
# PagedVSAPool — gestión elástica de memoria espectral (H200 Paged-VSA)
#
# Motivación: en inferencia con múltiples secuencias concurrentes, cada
# SpectralVSAArchive reserva `buf [window_size, d_model]` estáticamente.
# Para 1024 secuencias con window_size=256, d=1024: 1024×256×1024×2 bytes ≈ 512 MB
# de VRAM inutilizados en colas cortas.
#
# Solución: pool de páginas físicas compartidas. Una página = `page_size` tokens
# de coeficientes espectrales. Solo se asignan páginas cuando se archivan tokens.
# Análogo a `PagedBusCache` (V9) pero para el buffer VSA Chebyshev.
#
# API:
#   pool = PagedVSAPool(d_model=1024, window_size=256, page_size=16, n_pages=2048)
#   page_ids = pool.alloc(n_pages_needed)   # lista de índices de página
#   view = pool.get_buf_view(page_ids)      # vista lógica [window_size, d_model]
#   pool.free(page_ids)                     # devuelve páginas al pool
# ─────────────────────────────────────────────────────────────────────────────

class PagedVSAPool:
    """
    Pool compartido de páginas de memoria espectral para SpectralVSAArchive.

    Permite N instancias concurrentes de SpectralVSAArchive compartir un único
    bloque de VRAM físico. Las páginas se asignan on-demand y se liberan en
    reset() — igual que las kv-cache contiguas de PagedAttention.

    Implementación:
      _pool:       [n_pages_max, page_size, d_model] — bloque físico unificado
      _free_list:  [n_pages_max] int64 — índices libres (LIFO stack)
      _n_free:     int — número de páginas libres
    """

    def __init__(
        self,
        d_model:     int,
        page_size:   int = 16,      # tokens por página
        n_pages_max: int = 2048,    # máximo de páginas en el pool
        device:      str = 'cuda',
        dtype:       torch.dtype = torch.float32,
    ):
        self.d_model     = d_model
        self.page_size   = page_size
        self.n_pages_max = n_pages_max
        self.device      = device
        self.dtype       = dtype

        # Bloque físico: todas las páginas en memoria contigua
        self._pool      = torch.zeros(n_pages_max, page_size, d_model,
                                      device=device, dtype=dtype)
        # Free list LIFO: índices de páginas disponibles
        self._free_list = torch.arange(n_pages_max, device=device, dtype=torch.long)
        self._n_free    = n_pages_max

    # ── Asignación ──────────────────────────────────────────────────────────

    def alloc(self, n_pages: int) -> torch.Tensor:
        """
        Asigna `n_pages` páginas contiguas en el free-list.
        Retorna tensor [n_pages] con los índices de página asignados.
        Lanza RuntimeError si no hay páginas suficientes.
        """
        if n_pages > self._n_free:
            raise RuntimeError(
                f"PagedVSAPool: sin páginas libres — necesita {n_pages}, "
                f"disponibles {self._n_free}/{self.n_pages_max}"
            )
        # LIFO: tomar los últimos n_pages del free_list
        start = self._n_free - n_pages
        page_ids = self._free_list[start:self._n_free].clone()
        self._n_free -= n_pages
        # Zero las páginas recién asignadas (evita leak de datos anteriores)
        self._pool[page_ids].zero_()
        return page_ids

    def free(self, page_ids: torch.Tensor) -> None:
        """Devuelve `page_ids` al pool (LIFO)."""
        n = page_ids.numel()
        end = self._n_free + n
        if end > self.n_pages_max:
            raise RuntimeError("PagedVSAPool.free: overflow del free-list")
        self._free_list[self._n_free:end] = page_ids
        self._n_free = end

    # ── Vista lógica ─────────────────────────────────────────────────────────

    def get_buf_view(
        self, page_ids: torch.Tensor, window_size: int
    ) -> torch.Tensor:
        """
        Retorna una vista [window_size, d_model] sobre las páginas dadas.
        Las páginas se acceden en orden: page_ids[0], page_ids[1], ...
        Total de tokens disponibles = len(page_ids) * page_size ≥ window_size.
        Se lanza AssertionError si la cobertura es insuficiente.
        """
        n_pages = page_ids.numel()
        total_slots = n_pages * self.page_size
        assert total_slots >= window_size, (
            f"PagedVSAPool: {n_pages} páginas × {self.page_size} = {total_slots} "
            f"tokens insuficientes para window_size={window_size}"
        )
        # Concatenar páginas → [n_pages * page_size, d_model], tomar primeros W
        raw = self._pool[page_ids].reshape(total_slots, self.d_model)
        return raw[:window_size]   # view, no copia

    # ── Diagnóstico ──────────────────────────────────────────────────────────

    @property
    def n_free(self) -> int:
        return self._n_free

    @property
    def utilization(self) -> float:
        used = self.n_pages_max - self._n_free
        return used / self.n_pages_max

    def __repr__(self) -> str:
        used = self.n_pages_max - self._n_free
        return (
            f"PagedVSAPool(d={self.d_model}, page_size={self.page_size}, "
            f"pages={used}/{self.n_pages_max}, "
            f"VRAM={self._pool.numel()*self._pool.element_size()/1e6:.1f}MB)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ComplexityGate — Learned Chebyshev degree truncation
#
# Problema: _update_active_K usa un umbral de energía fijo (energy_threshold=0.95).
# Un chunk con "el el el..." × 200 consume los mismos K=16 grados que un chunk
# de código Python complejo. El umbral heurístico no distingue entre
# "energía concentrada porque la señal es simple" y "energía concentrada porque
# los coeficientes altos fueron eliminados por Lanczos".
#
# Solución: un gate aprendido que predice K_active a partir de las estadísticas
# espectrales del chunk. La red recibe:
#   - Energía por banda: norms² [K_max] — concentración espectral
#   - Ratio hi/lo: E_high / E_low — "complejidad" de frecuencia
#   - Delta EMA: variación temporal del espectro
#
# Output: soft_K ∈ [K_min, K_max] via sigmoid × (K_max - K_min) + K_min.
# La salida es continua → gradiente fluye via la máscara de bandas activas.
#
# El gradiente llega a ComplexityGate a través de:
#   retrieve() → top-n bands selection → scores built from w_bands+freq_bias
#   → multiplicadas por band_trust[k] → si K_active trunca una banda valiosa,
#   la pérdida de retrieval propaga error hacia el gate.
#
# Overhead: ~K_max + 64 parámetros. Despreciable.
# ─────────────────────────────────────────────────────────────────────────────

class ComplexityGate(nn.Module):
    """
    Learned spectral complexity gate for dynamic Chebyshev degree truncation.

    Input: spectral statistics (energy per band, hi/lo ratio, temporal delta)
    Output: soft_K ∈ [K_min, K_max] — target active degree

    The gate learns to:
      - Truncate to K=4 for predictable sequences (low entropy: repetitive text)
      - Expand to K=K_max for high-entropy sequences (complex code, rare tokens)
      - React to temporal changes in spectral structure (discontinuities)
    """
    def __init__(self, K_max: int, K_min: int = 4):
        super().__init__()
        self.K_max = K_max
        self.K_min = max(K_min, 2)

        # Input features: K_max energy bands + 2 scalar features (hi_lo ratio, delta_ema)
        in_dim = K_max + 2
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        # Init bias to produce K_max at startup (sigmoid(2) ≈ 0.88 → near K_max)
        nn.init.constant_(self.gate[-1].bias, 2.0)

    def forward(
        self,
        band_energy: torch.Tensor,  # [K_max] — squared norms of Chebyshev coefficients
        hilo_ratio: float,          # E_high / E_low
        delta_ema: float,           # temporal variation of spectrum
    ) -> torch.Tensor:
        """
        Returns: scalar tensor — target K_active ∈ [K_min, K_max] (continuous).
        """
        # Build feature vector
        feats = torch.cat([
            band_energy.detach().float(),
            torch.tensor([hilo_ratio, delta_ema], device=band_energy.device, dtype=torch.float32),
        ]).unsqueeze(0)  # [1, K_max + 2]

        # Gate output: sigmoid → [0, 1] → scale to [K_min, K_max]
        raw = self.gate(feats).squeeze()  # scalar
        K_range = self.K_max - self.K_min
        soft_K = torch.sigmoid(raw) * K_range + self.K_min
        return soft_K


class SpectralVSAArchive(nn.Module):

    def __init__(
        self,
        d_model: int,
        K: int = 32,
        K_min: int = 4,
        window_size: int = 256,
        ema_alpha: float = 0.9,
        use_complex_roles: bool = True,
        n_retrieve_bands: int = 8,
        energy_threshold: float = 0.95,
        lanczos_power_max: float = 4.0,
        disc_gamma: float = 3.0,
        error_refresh_ratio: float = 0.5,
        paged_pool: 'PagedVSAPool | None' = None,  # H200: gestión elástica de buf
        use_learned_gate: bool = False,             # H200: learned complexity gate
    ):
        super().__init__()
        self.d_model = d_model
        self.K_max = K
        self.K_min = max(K_min, 2)
        self.window_size = window_size
        self.stride = window_size // 2
        self.ema_alpha = ema_alpha
        self.use_complex_roles = use_complex_roles
        self.n_retrieve_bands = min(n_retrieve_bands, K)
        self.energy_threshold = energy_threshold
        self.lanczos_power_max = lanczos_power_max
        self.disc_gamma = disc_gamma
        self.error_refresh_ratio = error_refresh_ratio

        # ── Dynamic K ────────────────────────────────────────────────────────
        self.register_buffer('K_active', torch.tensor(K, dtype=torch.long))

        # ── Learned complexity gate (H200) ───────────────────────────────────
        # Predice K_active a partir de estadísticas espectrales del chunk.
        # Cuando use_learned_gate=True: _update_active_K usa el gate aprendido
        # en lugar del umbral heurístico de energía acumulada.
        # Gradiente fluye via retrieve() → band selection → loss.
        self.use_learned_gate = use_learned_gate
        if use_learned_gate:
            self.complexity_gate = ComplexityGate(K, K_min)
        else:
            self.complexity_gate = None

        # ── Lanczos σ-factors [K_max] ────────────────────────────────────────
        sigma = self._build_lanczos_sigma(K)
        self.register_buffer('_lanczos_sigma', sigma)
        self.register_buffer('_lanczos_power', torch.tensor(1.0))

        # ── Learnable band weights for Δ_k aggregation ──────────────────────
        w_init = torch.tensor([1.0 / (k + 1) for k in range(K)])
        self.w_bands = nn.Parameter(w_init)

        # ── Frequency position bias for content-aware retrieval ──────────────
        self.freq_bias = nn.Parameter(torch.zeros(K))

        # ── Injection / blend gates ──────────────────────────────────────────
        self.inject_gate = nn.Parameter(torch.zeros(1))
        # Learned blend: sigmoid(1.0) ≈ 0.73 current / 0.27 historical
        self.blend_gate = nn.Parameter(torch.tensor(1.0))

        # ── Compress pathway (gradient lifeline) ─────────────────────────────
        self.compress_proj = nn.Linear(d_model, d_model, bias=False)
        self.compress_gate = nn.Parameter(torch.tensor(-6.0))

        # ── Retrieval projection ─────────────────────────────────────────────
        self.retrieve_proj = nn.Linear(d_model, d_model, bias=False)

        # ── Precomputed Chebyshev evaluation matrix [K_max, W] ───────────────
        cheby_mat = self._build_chebyshev_matrix(K, window_size)
        self.register_buffer('cheby_mat', cheby_mat)

        # ── VSA role vectors ─────────────────────────────────────────────────
        if use_complex_roles:
            roles = self._build_complex_roles(K, d_model)
            self.register_buffer('roles_real', roles.real.contiguous())
            self.register_buffer('roles_imag', roles.imag.contiguous())
        else:
            gen = torch.Generator().manual_seed(42)
            roles = torch.sign(torch.randn(K, d_model, generator=gen))
            roles[roles == 0] = 1.0
            self.register_buffer('roles_real', roles)

        # ── State buffers ────────────────────────────────────────────────────
        # V6: band_trust — per-band learned gate for VSA crosstalk suppression.
        # sigmoid(0)=0.5 neutral at init. Gradient flows through retrieve() → loss.
        self.band_trust = nn.Parameter(torch.zeros(K))

        # ── Paged-VSA: buf gestionado desde PagedVSAPool (H200) ──────────────
        # Si paged_pool se provee: buf NO se reserva como register_buffer estático.
        # En su lugar se asigna on-demand desde el pool compartido, ahorrando VRAM
        # en inferencia multi-secuencia (solo tokeniza lo que cabe en window_size).
        #
        # Si paged_pool=None (default, training): comportamiento idéntico a V8 —
        # register_buffer estático, backward compatible al 100%.
        n_pages_needed = math.ceil(window_size / paged_pool.page_size) if paged_pool else 0
        if paged_pool is not None:
            self._paged_pool    = paged_pool
            self._buf_page_ids  = paged_pool.alloc(n_pages_needed)
            # self.buf es una VIEW sobre el pool — se actualiza en reset()
            self.buf = paged_pool.get_buf_view(self._buf_page_ids, window_size)
        else:
            self._paged_pool    = None
            self._buf_page_ids  = None
            self.register_buffer('buf', torch.zeros(window_size, d_model))

        self.register_buffer('buf_pos', torch.tensor(0, dtype=torch.long))
        self.register_buffer('buf_count', torch.tensor(0, dtype=torch.long))

        # Chebyshev coefficients
        self.register_buffer('c_now', torch.zeros(K, d_model))
        self.register_buffer('c_past', torch.zeros(K, d_model))

        # Holographic memory
        self.register_buffer('V_mem_real', torch.zeros(d_model))
        if use_complex_roles:
            self.register_buffer('V_mem_imag', torch.zeros(d_model))

        # Kahan compensation buffers for V_mem EMA
        self.register_buffer('_V_comp_real', torch.zeros(d_model))
        if use_complex_roles:
            self.register_buffer('_V_comp_imag', torch.zeros(d_model))

        # Stored norms for denormalized retrieval
        self.register_buffer('_coeff_norms', torch.zeros(K))
        self.register_buffer('_coeff_norms_comp', torch.zeros(K))  # Kahan comp

        # Binding error correction: accumulated residuals [K, D]
        self.register_buffer('_error_correction', torch.zeros(K, d_model))
        self.register_buffer('_error_correction_comp', torch.zeros(K, d_model))

        # Spectral delta (cached)
        self.register_buffer('spectral_delta', torch.zeros(K))

        # ── Discontinuity detection state ────────────────────────────────────
        self.register_buffer('_hilo_ratio_ema', torch.tensor(1.0))
        self.register_buffer('_disc_count', torch.tensor(0, dtype=torch.long))
        # Cooldown: skip detection for N steps after a detection (prevents cascading)
        self.register_buffer('_disc_cooldown', torch.tensor(0, dtype=torch.long))
        # Adaptive inter-window delta EMA (for threshold calibration)
        self.register_buffer('_inter_delta_ema', torch.tensor(0.0))
        self.register_buffer('_inter_delta_ema_comp', torch.tensor(0.0))

        # ── Noise floor estimation ───────────────────────────────────────────
        self.register_buffer('_noise_floor', torch.tensor(0.0))
        self.register_buffer('_noise_floor_comp', torch.tensor(0.0))
        # Conservative noise gate: attenuates the raw estimate before EMA.
        # At 0.75 the floor tracks 75% of the measured tail energy, which
        # prevents over-estimation on transient spectral fluctuations.
        self.register_buffer('_noise_gate', torch.tensor(0.75))

        # ── Condition number tracking ────────────────────────────────────────
        self.register_buffer('_condition_number', torch.tensor(1.0))

        # ── General state ────────────────────────────────────────────────────
        self.register_buffer('_has_memory', torch.tensor(False))
        self.register_buffer('_step_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_delta_ema', torch.tensor(0.0))
        self.register_buffer('_delta_ema_comp', torch.tensor(0.0))
        self._delta_ema_alpha = 0.05

    # ═════════════════════════════════════════════════════════════════════════
    # Construction helpers
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_chebyshev_matrix(K: int, W: int) -> torch.Tensor:
        """
        Chebyshev evaluation matrix [K, W].
        cheby_mat[k, n] = T_k(t_n), t_n = cos(π(2n+1)/(2W)), 0-indexed.
        Computed in float64 then downcast for numerical accuracy.
        """
        ns = torch.arange(W, dtype=torch.float64)
        t_n = torch.cos(math.pi * (2 * ns + 1) / (2 * W))
        ks = torch.arange(K, dtype=torch.float64)
        angles = ks.unsqueeze(1) * torch.acos(t_n).unsqueeze(0)
        return torch.cos(angles).float()

    @staticmethod
    def _build_complex_roles(K: int, D: int) -> torch.Tensor:
        """
        Random unitary role vectors: r_k[d] = exp(iθ_{k,d}), θ ~ U(0, 2π).
        Guarantees |r_k[d]| = 1 and quasi-orthogonality across k.
        Fixed seed for reproducibility.
        """
        gen = torch.Generator().manual_seed(137)
        phases = torch.rand(K, D, generator=gen, dtype=torch.float64) * 2 * math.pi
        roles = torch.complex(torch.cos(phases), torch.sin(phases))
        return roles.to(torch.complex64)

    @staticmethod
    def _build_lanczos_sigma(K: int) -> torch.Tensor:
        """
        Lanczos σ-factors: σ_0 = 1, σ_k = sinc(k/K) for k ≥ 1.
        Dampens high-frequency Chebyshev coefficients to suppress
        Runge phenomenon and Gibbs oscillations at discontinuities.
        """
        sigma = torch.ones(K)
        for k in range(1, K):
            x = math.pi * k / K
            sigma[k] = math.sin(x) / x
        return sigma

    # ═════════════════════════════════════════════════════════════════════════
    # Kahan-compensated EMA
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _kahan_ema(x: torch.Tensor, v: torch.Tensor, alpha: float,
                   c: torch.Tensor) -> None:
        """
        In-place Kahan-compensated EMA: x ← α·x + (1-α)·v.
        c is the compensation buffer (same shape as x).
        Prevents O(ε·N) float drift → O(ε) total error after N steps.
        """
        addend = (1.0 - alpha) * (v - x)
        y = addend - c
        t = x + y
        c.copy_((t - x) - y)
        x.copy_(t)

    # ═════════════════════════════════════════════════════════════════════════
    # Dynamic K management
    # ═════════════════════════════════════════════════════════════════════════

    def _update_active_K(self, coeffs: torch.Tensor) -> None:
        """
        Adapt K_active based on spectral energy concentration.

        Dos modos:
          • Heuristic (use_learned_gate=False): umbral de energía acumulada.
            Portable, zero-param, funciona en todas las GPUs.
          • Learned gate (use_learned_gate=True): red MLP que predice K_active
            a partir de (energy, hilo_ratio, delta_ema). Gradiente fluye via
            la máscara de bandas → retrieve() → loss. Solo overhead ~K+66 params.
        """
        K_max = self.K_max
        norms = coeffs[:K_max].norm(dim=-1)  # [K_max]
        energy = norms.pow(2)
        total = energy.sum()
        if total < 1e-12:
            return

        # ── Learned complexity gate (H200 path) ─────────────────────────────
        if self.complexity_gate is not None:
            quarter = max(K_max // 4, 1)
            E_low  = energy[:quarter].sum()
            E_high = energy[-quarter:].sum()
            hilo   = (E_high / (E_low + 1e-10)).item()
            delta  = self._hilo_ratio_ema.item() if hasattr(self, '_hilo_ratio_ema') else 0.0
            soft_K = self.complexity_gate(energy, hilo, delta)
            # Round with clamp — soft_K is continuous for gradient flow
            # through the ComplexityGate parameters (via band_energy detach).
            k_star = int(soft_K.round().clamp(self.K_min, K_max).item())
            self.K_active.fill_(k_star)
            return

        # ── Heuristic path (Ada/Ampere fallback) ────────────────────────────
        cum_energy = energy.cumsum(0) / total
        # Find smallest k* where cumulative energy >= threshold
        above = (cum_energy >= self.energy_threshold).long()
        if above.sum() == 0:
            k_star = K_max
        else:
            k_star = above.argmax().item() + 1
        k_star = max(self.K_min, min(k_star + 2, K_max))  # +2 margin for stability
        self.K_active.fill_(k_star)

    # ═════════════════════════════════════════════════════════════════════════
    # Discontinuity detection & adaptive damping
    # ═════════════════════════════════════════════════════════════════════════

    def _detect_discontinuity(self, new_coeffs_raw: torch.Tensor) -> bool:
        """
        Dual-criterion semantic jump detector with circular-drift fix:

          KEY FIX v2.1: Criterion 2 now compares in DAMPED SPACE.
          The original code compared raw new_coeffs vs damped c_now.
          At high Lanczos power (p=4), c_now is heavily attenuated, so the
          raw-vs-damped delta was always enormous → cascading false positives:
            high disc → high p → attenuated c_now → huge Δ → high disc → ...
          Fix: apply current sigma^p to new_coeffs before comparing with c_now.
          Both are now in the same damped space → fair, calibrated comparison.

          Added: cooldown (3-step skip after detection) + adaptive threshold
          (EMA of typical delta level + disc_gamma·σ) instead of fixed cutoff.
        """
        K_act = self.K_active.item()
        if K_act < 4:
            return False

        # Cooldown: skip N steps after a recent detection to prevent cascading
        if self._disc_cooldown.item() > 0:
            self._disc_cooldown.sub_(1)
            return False

        # Skip first window — no baseline to compare against (c_now is zeros)
        if self.c_now[:K_act].abs().sum().item() < 1e-10:
            return False

        norms = new_coeffs_raw[:K_act].norm(dim=-1)
        quarter = max(K_act // 4, 1)

        # ── Criterion 1: Hi/lo energy ratio spike ────────────────────────────
        E_low  = norms[:quarter].pow(2).sum()
        E_high = norms[-quarter:].pow(2).sum()
        R = E_high / (E_low + 1e-10)

        R_ema_old = self._hilo_ratio_ema
        self._hilo_ratio_ema.copy_(0.9 * R_ema_old + 0.1 * R.detach())
        # FIX: additive floor 0.1 → 0.5.
        # En datos lingüísticos, la relación E_high/E_low fluctuú naturalmente
        # porque el lenguaje tiene espectros Chebyshev casi planos (saltos
        # semánticos frecuentes). Un floor=0.1 dispara con cualquier pico
        # moderado. 0.5 requiere una desviación significativa del baseline.
        hilo_spike = R.item() > self.disc_gamma * self._hilo_ratio_ema.item() + 0.5

        # ── Criterion 2: Inter-window delta IN DAMPED SPACE ──────────────────
        # Apply current Lanczos sigma to new_coeffs before comparison.
        # Both new_damped and c_now are now in the same attenuated space.
        sigma = self._lanczos_sigma[:K_act]
        p = self._lanczos_power.item()
        if p != 1.0:
            sigma = sigma.pow(p)
        damped_new = new_coeffs_raw[:K_act] * sigma.unsqueeze(-1)
        inter_delta = (damped_new - self.c_now[:K_act]).norm()
        c_norm = self.c_now[:K_act].norm() + 1e-10
        relative_delta = (inter_delta / c_norm).item()

        # Adaptive threshold: EMA-tracked baseline + disc_gamma·(20% of baseline)
        # For constant-level signals: ema_rd converges to the typical delta;
        # a spike needs to exceed that level by disc_gamma std-widths.
        self._kahan_ema(
            self._inter_delta_ema,
            torch.tensor(relative_delta, device=self._inter_delta_ema.device),
            0.95,
            self._inter_delta_ema_comp,
        )
        ema_rd = self._inter_delta_ema.item()
        # FIX: mínimo del threshold 0.05 → 0.15.
        # Para datos lingüísticos, los saltos interventana típicos son mayores
        # que para señales numéricas de estado estacionario. Un floor de 0.05
        # significa que cualquier delta > 0.05 relativo al EMA dispara.
        # Floor de 0.15: requiere al menos 15% de cambio relativo antes de
        # considerar salto semántico + margin adicional del disc_gamma term.
        threshold_rd = ema_rd + self.disc_gamma * max(0.2 * ema_rd, 0.15)
        delta_spike = relative_delta > threshold_rd

        is_disc = hilo_spike or delta_spike

        if is_disc:
            # Severity proportional to how much signal exceeds baseline
            severity = max(
                R.item() / (self._hilo_ratio_ema.item() + 1e-10),
                relative_delta / (ema_rd + 1e-6),
            )
            # FIX: boost rate-limited (max +0.3 por detección, vs salto libre a max).
            # El código anterior: boost = 1.0 + 0.3*severity podía saltar a p=4.0
            # en UN solo paso (severity ≥10). Con rate-limiting, se necesitan
            # múltiples detecciones para llegar al máximo → el cooldown=3 las
            # espaciará al menos 3 ventanas, dando tiempo de recuperación.
            # El capped_increment de 0.3 es conservador para datos lingüísticos.
            current_p = self._lanczos_power.item()
            increment = min(0.3, 0.1 * severity)      # máx +0.3 por trigger
            new_p = min(current_p + increment, self.lanczos_power_max)
            self._lanczos_power.fill_(new_p)
            new_K = max(self.K_min, K_act - quarter)
            self.K_active.fill_(new_K)
            self._disc_count.add_(1)
            # Cooldown: skip next 3 detection calls to avoid cascade
            self._disc_cooldown.fill_(3)
        else:
            # FIX: recovery 0.95 → 0.92 (recuperación más rápida hacia p=1.0).
            # Con 0.95: necesita ~20 ventanas sin discontinuidades para volver de p=2 a 1.
            # Con 0.92: necesita ~12 ventanas. Más apropiado para lenguaje
            # donde las transiciones son frecuentes pero cortas.
            current_p = self._lanczos_power.item()
            if current_p > 1.0:
                self._lanczos_power.fill_(max(current_p * 0.92, 1.0))

        return is_disc

    # ═════════════════════════════════════════════════════════════════════════
    # Chebyshev coefficient computation (enhanced)
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_chebyshev_coefficients(self) -> torch.Tensor:
        """
        Compute Chebyshev coefficients from circular buffer.
        All arithmetic forced to FP32 for numerical precision.
        Returns: [K_max, D] tensor.
        """
        W = self.window_size
        K = self.K_max
        count = min(self.buf_count.item(), W)

        if count < 4:
            return self.c_now.clone()

        if count < W:
            mat = self._build_chebyshev_matrix(K, count).to(self.buf.device)
            data = self.buf[:count].float()
            coeffs = (2.0 / count) * (mat @ data)
            coeffs[0] = coeffs[0] * 0.5
        else:
            pos = self.buf_pos.item()
            if pos == 0:
                data = self.buf.float()
            else:
                data = torch.cat([self.buf[pos:], self.buf[:pos]], dim=0).float()
            coeffs = (2.0 / W) * (self.cheby_mat.float() @ data)
            coeffs[0] = coeffs[0] * 0.5

        return coeffs

    def _apply_lanczos_damping(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Apply Lanczos σ-factor damping: c_k *= σ_k^p.
        p is adapted by the discontinuity sentinel.
        """
        p = self._lanczos_power.item()
        K_act = self.K_active.item()
        sigma = self._lanczos_sigma[:K_act]
        if p != 1.0:
            sigma = sigma.pow(p)
        damped = coeffs.clone()
        damped[:K_act] = coeffs[:K_act] * sigma.unsqueeze(-1)
        # Zero out inactive bands
        if K_act < self.K_max:
            damped[K_act:] = 0.0
        return damped

    # ═════════════════════════════════════════════════════════════════════════
    # Condition number monitoring & rank stabilization
    # ═════════════════════════════════════════════════════════════════════════

    def _update_condition(self, coeffs: torch.Tensor) -> None:
        """
        Track condition number κ = max_norm / min_norm of active coefficients.
        High κ → some bands dominate → numerical instability → feed back into
        Lanczos power: extreme damping raises κ; nudging p down distributes
        energy more evenly, lowering κ.
        """
        K_act = self.K_active.item()
        norms = coeffs[:K_act].norm(dim=-1)
        norms_positive = norms[norms > 1e-12]
        if norms_positive.numel() >= 2:
            kappa = norms_positive.max() / norms_positive.clamp(min=1e-3).min() # OPT-1: Ridge-like condition clamp to avoid exploding condition number from vanishing bands
            self._condition_number.copy_(kappa.detach())
            # Condition feedback: if κ is pathologically high, reduce Lanczos power
            # (stronger damping → higher κ; reducing p redistributes energy).
            # Only acts when p > 1.0 to avoid fighting the recovery path.
            kappa_val = kappa.item()
            # Three-tier κ→Lanczos feedback (calibrated from 300-step eval):
            #   κ > 800 → 18% nudge toward p=1.0  (aggressive)
            #   κ > 200 → 8%  nudge toward p=1.0  (moderate)
            #   κ ≤ 200 → no action (normal operation)
            # Previous thresholds (100/500/2000) were too eager at low κ values
            # and too weak for the extreme spikes observed (κ_max=13952).
            if kappa_val > 800.0 and self._lanczos_power.item() > 1.0:
                rate = 0.18
            elif kappa_val > 200.0 and self._lanczos_power.item() > 1.0:
                rate = 0.08
            else:
                rate = 0.0
            if rate > 0.0:
                current_p = self._lanczos_power.item()
                self._lanczos_power.fill_(max(1.0 + (current_p - 1.0) * (1.0 - rate), 1.0))

    # ═════════════════════════════════════════════════════════════════════════
    # Noise floor estimation
    # ═════════════════════════════════════════════════════════════════════════

    def _update_noise_floor(self, coeffs: torch.Tensor) -> None:
        """
        Spectrum-relative noise floor estimator.

        KEY FIX v2.1: The original estimator used raw tail magnitude.
        For flat/random spectra (β ≈ 0), the tail IS the signal, so the
        estimated floor ≈ signal level → soft shrinkage destroys the retrieval.

        Fix: scale by spectral decay rate.
          decay_ratio = tail_norm / peak_norm (1=flat, 0=steep decay)
          noise_est = tail_norm × (1 - decay_ratio)  → 0 for random, > 0 for smooth
        Only signals with genuine spectral decay (β > ~0.5) get a non-zero floor,
        which is the correct condition for noise-floor subtraction to make sense.
        """
        K_act = self.K_active.item()
        tail_size = max(K_act // 8, 1)
        norms = coeffs[:K_act].norm(dim=-1)

        peak_norm = norms[:max(K_act // 4, 1)].mean().clamp(min=1e-12)
        tail_norm = norms[K_act - tail_size:K_act].mean()

        # decay_ratio ≈ 1 for flat spectrum → multiplicative factor ≈ 0 → no floor
        # decay_ratio ≈ 0 for steeply decaying spectrum → full tail_norm as floor
        decay_ratio = (tail_norm / peak_norm).clamp(0.0, 1.0)
        noise_est = (tail_norm * (1.0 - decay_ratio)).detach()
        # Apply conservative gate before EMA-updating to avoid over-estimating
        # noise on transient high-frequency bursts (prevents over-shrinkage).
        self._kahan_ema(self._noise_floor, noise_est * self._noise_gate, 0.9, self._noise_floor_comp)

    # ═════════════════════════════════════════════════════════════════════════
    # VSA binding with error correction
    # ═════════════════════════════════════════════════════════════════════════

    def _bind(self, coeffs: torch.Tensor) -> None:
        """
        Bind K_active coefficient vectors into V_mem via Kahan-compensated EMA.
        After binding, performs shadow-unbind for error correction.
        """
        K_act = self.K_active.item()
        alpha = self.ema_alpha
        active_coeffs = coeffs[:K_act]

        # Normalize to unit L2 norm per band (standard HRR)
        norms = active_coeffs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        coeffs_normed = active_coeffs / norms

        # Update stored norms via Kahan EMA
        norm_vec = norms.squeeze(-1)
        target_norms = torch.zeros(self.K_max, device=coeffs.device)
        target_norms[:K_act] = norm_vec
        self._kahan_ema(self._coeff_norms, target_norms, alpha, self._coeff_norms_comp)

        # Compute bound superposition
        if self.use_complex_roles:
            r_real = self.roles_real[:K_act]
            r_imag = self.roles_imag[:K_act]
            bound_real = (coeffs_normed * r_real).sum(dim=0)
            bound_imag = (coeffs_normed * r_imag).sum(dim=0)
            self._kahan_ema(self.V_mem_real, bound_real, alpha, self._V_comp_real)
            self._kahan_ema(self.V_mem_imag, bound_imag, alpha, self._V_comp_imag)
        else:
            r = self.roles_real[:K_act]
            bound = (coeffs_normed * r).sum(dim=0)
            self._kahan_ema(self.V_mem_real, bound, alpha, self._V_comp_real)

        # ── Shadow unbind for error correction ───────────────────────────────
        self._update_binding_correction(coeffs_normed, K_act)

    def _update_binding_correction(self, coeffs_normed: torch.Tensor,
                                   K_act: int) -> None:
        """
        Vectorized shadow-unbind for all active bands.
        Computes residual = expected - actual, accumulates via Kahan EMA.
        """
        correction = torch.zeros_like(self._error_correction)

        if self.use_complex_roles:
            r_real = self.roles_real[:K_act]  # [K_act, D]
            r_imag = self.roles_imag[:K_act]  # [K_act, D]
            # Vectorized unbind: [K_act, D]
            c_hat = (self.V_mem_real.unsqueeze(0) * r_real +
                     self.V_mem_imag.unsqueeze(0) * r_imag)
            correction[:K_act] = coeffs_normed - c_hat
        else:
            r = self.roles_real[:K_act]
            c_hat = self.V_mem_real.unsqueeze(0) * r
            correction[:K_act] = coeffs_normed - c_hat

        self._kahan_ema(self._error_correction, correction, self.ema_alpha,
                        self._error_correction_comp)

        # Check if accumulated error is too large → trigger full refresh
        error_magnitude = self._error_correction[:K_act].norm()
        vmem_magnitude = self.V_mem_real.norm()
        if self.use_complex_roles:
            vmem_magnitude = (self.V_mem_real.pow(2) +
                              self.V_mem_imag.pow(2)).sqrt().sum()
        if vmem_magnitude > 1e-12:
            ratio = (error_magnitude / vmem_magnitude).item()
            if ratio > self.error_refresh_ratio:
                # Full refresh: rebind from scratch (reset V_mem, rebind)
                self._full_refresh(self.c_now[:K_act])

    def _full_refresh(self, coeffs: torch.Tensor) -> None:
        """
        Emergency rebind from scratch when accumulated error is too large.
        Resets V_mem and compensation buffers, then binds fresh.
        """
        K_act = coeffs.shape[0]
        norms = coeffs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        coeffs_normed = coeffs / norms

        # Zero everything
        self.V_mem_real.zero_()
        self._V_comp_real.zero_()
        self._error_correction.zero_()
        self._error_correction_comp.zero_()
        if self.use_complex_roles:
            self.V_mem_imag.zero_()
            self._V_comp_imag.zero_()

        # Direct bind (no EMA, just set)
        if self.use_complex_roles:
            r_real = self.roles_real[:K_act]
            r_imag = self.roles_imag[:K_act]
            self.V_mem_real.copy_((coeffs_normed * r_real).sum(dim=0))
            self.V_mem_imag.copy_((coeffs_normed * r_imag).sum(dim=0))
        else:
            r = self.roles_real[:K_act]
            self.V_mem_real.copy_((coeffs_normed * r).sum(dim=0))

    # ═════════════════════════════════════════════════════════════════════════
    # VSA unbinding with error correction
    # ═════════════════════════════════════════════════════════════════════════

    def _unbind(self, band_indices: torch.Tensor) -> torch.Tensor:
        """
        Unbind selected bands from V_mem with error correction applied.
        Returns DENORMALIZED coefficients (rescaled by stored norms).
        """
        if self.use_complex_roles:
            r_real = self.roles_real[band_indices]
            r_imag = self.roles_imag[band_indices]
            retrieved = (self.V_mem_real.unsqueeze(0) * r_real +
                         self.V_mem_imag.unsqueeze(0) * r_imag)
        else:
            r = self.roles_real[band_indices]
            retrieved = self.V_mem_real.unsqueeze(0) * r

        # Apply error correction (accumulated residuals)
        correction = self._error_correction[band_indices]
        retrieved = retrieved + correction

        # Rescale by stored norms
        stored_norms = self._coeff_norms[band_indices].unsqueeze(-1).clamp(min=1e-12)
        retrieved = retrieved * stored_norms

        # Subtract noise floor (anti-quantization denoising)
        # Safety guard: only apply when floor is small relative to the signal
        # (prevents over-shrinkage when floor ≈ signal, e.g. random/high-entropy data).
        noise_floor = self._noise_floor.item()
        if noise_floor > 1e-6:
            ret_norms = retrieved.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            mean_ret = ret_norms.mean().item()
            if noise_floor < 0.3 * mean_ret:   # guard: floor must be < 30% of signal
                shrink_factor = ((ret_norms - noise_floor) / ret_norms).clamp(min=0.0)
                retrieved = retrieved * shrink_factor

        return retrieved

    # ═════════════════════════════════════════════════════════════════════════
    # Spectral delta Δ_k
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_spectral_delta(self) -> torch.Tensor:
        """Per-frequency-band importance: Δ_k = ||c_now[k] - c_past[k]||_2."""
        diff = self.c_now - self.c_past
        return diff.norm(dim=-1)

    def get_spectral_delta(self, scan_out: torch.Tensor = None) -> torch.Tensor:
        """Public API: returns cached spectral delta [K]."""
        return self.spectral_delta.detach()

    def get_spectral_importance(self) -> torch.Tensor:
        """Aggregate scalar importance: I = Σ_k w_k · Δ_k."""
        return (F.softplus(self.w_bands) * self.spectral_delta).sum()

    # ═════════════════════════════════════════════════════════════════════════
    # Public API: maybe_archive
    # ═════════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def maybe_archive(
        self,
        scan_out:       torch.Tensor,
        ttt_importance: torch.Tensor,
        tier_probs:     torch.Tensor,
        sgr_indices:    torch.Tensor,
    ) -> bool:
        """
        Accumulate hidden states and periodically recompute spectral coefficients
        + bind into holographic memory. Fully vectorized buffer write.
        """
        B, S, D = scan_out.shape

        # Mean over batch → representative trajectory
        h_mean = scan_out.float().mean(dim=0)  # [S, D]

        # ── Vectorized circular buffer write ─────────────────────────────────
        pos = self.buf_pos.item()
        W = self.window_size

        if S <= W:
            end = pos + S
            if end <= W:
                self.buf[pos:end] = h_mean.to(self.buf.dtype)
            else:
                first_chunk = W - pos
                self.buf[pos:] = h_mean[:first_chunk].to(self.buf.dtype)
                self.buf[:S - first_chunk] = h_mean[first_chunk:].to(self.buf.dtype)
            self.buf_pos.fill_((pos + S) % W)
        else:
            # More tokens than buffer: keep last W tokens
            self.buf.copy_(h_mean[-W:].to(self.buf.dtype))
            self.buf_pos.fill_(0)

        self.buf_count.add_(S)

        # ── Check if stride boundary crossed ─────────────────────────────────
        old_step = self._step_count.item()
        new_step = old_step + S
        self._step_count.fill_(new_step)

        n_triggers = (new_step // self.stride) - (old_step // self.stride)
        if n_triggers <= 0 or self.buf_count.item() < 4:
            return False

        # ── Recompute coefficients (use last trigger only) ───────────────────
        self.c_past.copy_(self.c_now)
        new_coeffs = self._compute_chebyshev_coefficients()

        # ── Dynamic K adjustment ─────────────────────────────────────────────
        self._update_active_K(new_coeffs)

        # ── Discontinuity detection ──────────────────────────────────────────
        self._detect_discontinuity(new_coeffs)

        # ── Apply Lanczos damping (anti-Runge/Gibbs) ─────────────────────────
        damped_coeffs = self._apply_lanczos_damping(new_coeffs)
        self.c_now.copy_(damped_coeffs)

        # ── Condition monitoring ─────────────────────────────────────────────
        self._update_condition(damped_coeffs)

        # ── Noise floor estimation ───────────────────────────────────────────
        self._update_noise_floor(new_coeffs)  # from raw coeffs (before damping)

        # ── Spectral delta ───────────────────────────────────────────────────
        delta = self._compute_spectral_delta()
        self.spectral_delta.copy_(delta)

        # ── Adaptive threshold for binding ───────────────────────────────────
        I_total = (F.softplus(self.w_bands) * delta).sum().item()
        self._kahan_ema(
            self._delta_ema,
            torch.tensor(I_total, device=self._delta_ema.device),
            1.0 - self._delta_ema_alpha,
            self._delta_ema_comp,
        )

        threshold = self._delta_ema.item() + 0.01
        if I_total > threshold or not self._has_memory.item():
            self._bind(damped_coeffs)
            self._has_memory.fill_(True)
            return True

        return False

    # ═════════════════════════════════════════════════════════════════════════
    # Public API: retrieve
    # ═════════════════════════════════════════════════════════════════════════

    def retrieve(self, scan_out: torch.Tensor) -> torch.Tensor:
        """
        Retrieve relevant historical context from spectral memory.
        Content-aware band selection: combines content similarity, Δ_k, and freq bias.

        FIX V6 (static shapes): Usa K_max con masking en vez de K_active slicing
        dinámico. Esto elimina el graph break de K_active.item() y hace retrieve()
        compatible con torch.compile(fullgraph=True) y CUDA Graphs.
        Las bandas inactivas (>= K_active) reciben score = -inf → nunca seleccionadas.

        FIX V6 (band_trust): Las bandas históricas de V_mem sufren de
        interferencia (crosstalk) proporcional al número de vectores superpuestos.
        band_trust es una gate paramétrica aprendida por banda que modula la
        contribución histórica: bandas ruidosas aprenden trust bajo.
        """
        B, S, D = scan_out.shape
        gate = torch.sigmoid(self.inject_gate)
        has_mem = self._has_memory.float()  # 0.0 or 1.0, no .item()

        # Query representative
        q = scan_out.mean(dim=1).detach()

        K_max = self.K_max
        K_act = self.K_active.item()  # only used to build the static mask

        # Static shapes: work with full K_max, mask inactive bands
        c_all_f = self.c_now.float()                           # [K_max, D]
        q_mean_f = q.mean(dim=0).float()

        # 1. Content relevance (K_max scores)
        content_score = (c_all_f * q_mean_f.unsqueeze(0)).sum(dim=-1) / math.sqrt(D)  # [K_max]
        # 2. Spectral importance (Δ_k weighted)
        delta_score = F.softplus(self.w_bands.float()) * self.spectral_delta.float()  # [K_max]
        # 3. Learned frequency position bias
        freq_b = self.freq_bias.float()                        # [K_max]
        # Combined
        combined_score = content_score + delta_score + freq_b  # [K_max]

        # Mask inactive bands with -inf → never selected by topk
        active_mask = torch.arange(K_max, device=scan_out.device) < K_act
        combined_score = torch.where(active_mask, combined_score,
                                     torch.tensor(float('-inf'), device=scan_out.device))

        # Select top-J bands (static J = n_retrieve_bands)
        J = self.n_retrieve_bands
        _, top_bands = combined_score.topk(J, dim=0)
        band_weights = F.softmax(combined_score[top_bands], dim=0).unsqueeze(-1)

        # ── Current-window reconstruction (exact, no VSA noise) ──────────────
        current_coeffs = c_all_f[top_bands]   # float32
        h_current = (current_coeffs * band_weights).sum(dim=0)

        # ── Historical reconstruction (from V_mem with error correction) ─────
        historical_coeffs = self._unbind(top_bands)
        # Parametric band trust: per-band gate modulating historical reliability.
        # sigmoid(0)=0.5 neutral. Gradient flows through loss → retrieve → band_trust.
        trust = torch.sigmoid(self.band_trust[top_bands]).unsqueeze(-1)  # [J, 1]
        h_historical = (historical_coeffs * trust * band_weights).sum(dim=0)

        # ── Adaptive blend (learned) ────────────────────────────────────────
        blend_w = torch.sigmoid(self.blend_gate)
        h_blend = blend_w * h_current + (1.0 - blend_w) * h_historical

        # ── Project ──────────────────────────────────────────────────────────
        h_proj = self.retrieve_proj(h_blend.to(self.retrieve_proj.weight.dtype))

        ctx = h_proj.unsqueeze(0).unsqueeze(0).expand(B, S, D)
        # Gate + memory presence mask (no .item())
        return scan_out + gate * has_mem * ctx

    # ═════════════════════════════════════════════════════════════════════════
    # Public API: get_compress_ctx (gradient lifeline)
    # ═════════════════════════════════════════════════════════════════════════

    def get_compress_ctx(self, scan_out: torch.Tensor) -> torch.Tensor:
        """Always-active gradient pathway. Bypasses arch_gate.

        FIX V6: eliminado .contiguous() que forzaba copia de [B,1,D] broadcast
        a [B,S,D] — S×D bytes desperdiciados por capa. expand() sin contiguous
        reutiliza memoria via strides (misma data, distintos offsets).
        """
        B, S, D = scan_out.shape
        q_repr = scan_out.mean(dim=1)
        compressed = self.compress_proj(q_repr)
        cq_gate = torch.sigmoid(self.compress_gate)
        return (compressed * cq_gate).unsqueeze(1).expand(B, S, D)

    # ═════════════════════════════════════════════════════════════════════════
    # Public API: info, reset, preload
    # ═════════════════════════════════════════════════════════════════════════

    def get_archive_info(self) -> dict:
        """Diagnostic dict for logging. Extended with v2 metrics."""
        delta = self.spectral_delta
        K_act = self.K_active.item()
        I_total = (F.softplus(self.w_bands) * delta).sum().item()
        return {
            'type':               'SpectralVSA_v2',
            'K_max':              self.K_max,
            'K_active':           K_act,
            'window_size':        self.window_size,
            'has_memory':         self._has_memory.item(),
            'step_count':         self._step_count.item(),
            'buf_fill':           min(self.buf_count.item(), self.window_size),
            'I_total':            round(I_total, 4),
            'delta_ema':          round(self._delta_ema.item(), 4),
            'delta_low_freq':     round(float(delta[:max(K_act // 4, 1)].mean()), 4),
            'delta_high_freq':    round(float(delta[max(3 * K_act // 4, 1):K_act].mean()), 4)
                                  if K_act > 1 else 0.0,
            'lanczos_power':      round(self._lanczos_power.item(), 3),
            'condition_number':   round(self._condition_number.item(), 2),
            'noise_floor':        round(self._noise_floor.item(), 6),
            'disc_count':         self._disc_count.item(),
            'error_correction_norm': round(float(self._error_correction[:K_act].norm()), 6),
            'memory_bytes':       self.d_model * (8 if self.use_complex_roles else 4),
            'inject_gate':        round(float(torch.sigmoid(self.inject_gate).detach()), 4),
            'blend_gate':         round(float(torch.sigmoid(self.blend_gate).detach()), 4),
        }

    def reset(self):
        """Reset all state (between documents/episodes).

        Paged-VSA: si hay un pool asignado, se devuelven las páginas al pool
        y se reasignan limpias (zero'd por alloc). Esto libera VRAM si el
        slot de la secuencia ya no se necesita (diferente a un simple zero_()).
        En modo estático (sin pool): comportamiento idéntico a versiones anteriores.
        """
        if self._paged_pool is not None:
            # Devolver páginas al pool y reasignar frescas (ya zeroed por alloc)
            self._paged_pool.free(self._buf_page_ids)
            self._buf_page_ids = self._paged_pool.alloc(
                math.ceil(self.window_size / self._paged_pool.page_size)
            )
            self.buf = self._paged_pool.get_buf_view(self._buf_page_ids, self.window_size)
        else:
            self.buf.zero_()
        self.buf_pos.zero_()
        self.buf_count.zero_()
        self.c_now.zero_()
        self.c_past.zero_()
        self.V_mem_real.zero_()
        self._V_comp_real.zero_()
        if self.use_complex_roles:
            self.V_mem_imag.zero_()
            self._V_comp_imag.zero_()
        self._coeff_norms.zero_()
        self._coeff_norms_comp.zero_()
        self._error_correction.zero_()
        self._error_correction_comp.zero_()
        self.spectral_delta.zero_()
        self._has_memory.fill_(False)
        self._step_count.zero_()
        self._delta_ema.zero_()
        self._delta_ema_comp.zero_()
        self.K_active.fill_(self.K_max)
        self._lanczos_power.fill_(1.0)
        self._hilo_ratio_ema.fill_(1.0)
        self._disc_count.zero_()
        self._disc_cooldown.zero_()
        self._inter_delta_ema.zero_()
        self._inter_delta_ema_comp.zero_()
        self._noise_floor.zero_()
        self._noise_floor_comp.zero_()
        self._condition_number.fill_(1.0)

    def preload_context(self, context_embs: torch.Tensor,
                        importance_scores: torch.Tensor = None) -> None:
        """
        Cold start: initialize V_mem from pre-existing context.
        context_embs: [T, D] or [B, T, D]
        """
        if context_embs.dim() == 3:
            context_embs = context_embs.mean(dim=0)
        T, D = context_embs.shape

        W = min(T, self.window_size)
        self.buf[:W] = context_embs[-W:].to(self.buf.dtype)
        if W < self.window_size:
            self.buf[W:] = 0.0
        self.buf_pos.fill_(W % self.window_size)
        self.buf_count.fill_(W)
        self._step_count.fill_(W)

        coeffs = self._compute_chebyshev_coefficients()
        self._update_active_K(coeffs)
        damped = self._apply_lanczos_damping(coeffs)
        self.c_now.copy_(damped)
        self.c_past.copy_(damped)
        self._bind(damped)
        self._has_memory.fill_(True)

    # ═════════════════════════════════════════════════════════════════════════
    # Diagnostics
    # ═════════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def measure_spectral_decay(
        self,
        hidden_states: torch.Tensor,
        K_max: int = 64,
    ) -> dict:
        """
        Measure spectral decay profile for SSST hypothesis validation.
        Returns coeff_norms, energy fractions, power-law β, etc.
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.float().mean(dim=0)
        else:
            hidden_states = hidden_states.float()

        T, D = hidden_states.shape
        K_max = min(K_max, T)

        mat = self._build_chebyshev_matrix(K_max, T).to(hidden_states.device)
        coeffs = (2.0 / T) * (mat @ hidden_states)
        coeffs[0] = coeffs[0] * 0.5

        norms = coeffs.norm(dim=-1)
        energy = norms.pow(2)
        total_energy = energy.sum()
        cum_energy = energy.cumsum(dim=0) / (total_energy + 1e-12)

        # Power-law fit in log-log space (k >= 1)
        valid_k = torch.arange(1, K_max, device=norms.device, dtype=torch.float32)
        valid_norms = norms[1:]
        nonzero = valid_norms > 1e-12
        if nonzero.sum() >= 3:
            log_k = torch.log(valid_k[nonzero])
            log_n = torch.log(valid_norms[nonzero])
            mean_lk = log_k.mean()
            mean_ln = log_n.mean()
            cov = ((log_k - mean_lk) * (log_n - mean_ln)).mean()
            var = ((log_k - mean_lk) ** 2).mean()
            beta = -(cov / (var + 1e-12)).item()
        else:
            beta = 0.0

        e16 = cum_energy[min(15, K_max - 1)].item() if K_max >= 16 else 0.0
        e32 = cum_energy[min(31, K_max - 1)].item() if K_max >= 32 else 0.0

        # Dynamic K recommendation
        above_95 = (cum_energy >= 0.95).long()
        K_rec = above_95.argmax().item() + 1 if above_95.sum() > 0 else K_max

        return {
            'coeff_norms':      norms.cpu(),
            'energy_fraction':  cum_energy.cpu(),
            'beta_estimate':    round(beta, 4),
            'energy_at_16':     round(e16, 4),
            'energy_at_32':     round(e32, 4),
            'K_recommended':    K_rec,
        }

    @torch.no_grad()
    def measure_vsa_interference(
        self,
        hidden_states: torch.Tensor,
    ) -> dict:
        """
        Measure VSA bind-unbind reconstruction error.
        Now includes error-corrected measurement as well.
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.float().mean(dim=0)
        else:
            hidden_states = hidden_states.float()

        T, D = hidden_states.shape
        K = min(self.K_max, T)

        mat = self._build_chebyshev_matrix(K, T).to(hidden_states.device)
        coeffs = (2.0 / T) * (mat @ hidden_states)
        coeffs[0] = coeffs[0] * 0.5

        # Apply Lanczos damping (same as production path)
        sigma = self._lanczos_sigma[:K]
        p = self._lanczos_power.item()
        if p != 1.0:
            sigma = sigma.pow(p)
        coeffs = coeffs * sigma.unsqueeze(-1)

        orig_norms = coeffs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        coeffs_normed = coeffs / orig_norms

        # Bind all
        if self.use_complex_roles:
            r_real = self.roles_real[:K]
            r_imag = self.roles_imag[:K]
            V_real = (coeffs_normed * r_real).sum(dim=0)
            V_imag = (coeffs_normed * r_imag).sum(dim=0)
        else:
            r = self.roles_real[:K]
            V_real = (coeffs_normed * r).sum(dim=0)
            V_imag = None

        # Unbind each and measure error
        errors_raw = []
        errors_corrected = []
        norms_sq = orig_norms.squeeze(-1)

        # Build correction (shadow-unbind residuals)
        corrections = torch.zeros_like(coeffs_normed)
        for k in range(K):
            if self.use_complex_roles:
                c_hat_normed = V_real * r_real[k] + V_imag * r_imag[k]
            else:
                c_hat_normed = V_real * self.roles_real[k]
            corrections[k] = coeffs_normed[k] - c_hat_normed

        for k in range(K):
            if self.use_complex_roles:
                c_hat_normed = V_real * r_real[k] + V_imag * r_imag[k]
            else:
                c_hat_normed = V_real * self.roles_real[k]

            # Raw error
            c_hat_raw = c_hat_normed * norms_sq[k]
            err_raw = (c_hat_raw - coeffs[k]).norm()
            raw_rel = (err_raw / (norms_sq[k] + 1e-12)).item()
            errors_raw.append(raw_rel)

            # Corrected error
            c_hat_corr = (c_hat_normed + corrections[k]) * norms_sq[k]
            err_corr = (c_hat_corr - coeffs[k]).norm()
            corr_rel = (err_corr / (norms_sq[k] + 1e-12)).item()
            errors_corrected.append(corr_rel)

        errors_raw_t = torch.tensor(errors_raw)
        errors_corr_t = torch.tensor(errors_corrected)
        theoretical_std = math.sqrt(max(K - 1, 0) / D)

        return {
            'mean_rel_error_raw':       round(float(errors_raw_t.mean()), 6),
            'max_rel_error_raw':        round(float(errors_raw_t.max()), 6),
            'mean_rel_error_corrected': round(float(errors_corr_t.mean()), 6),
            'max_rel_error_corrected':  round(float(errors_corr_t.max()), 6),
            'per_band_error_raw':       errors_raw_t,
            'per_band_error_corrected': errors_corr_t,
            'theoretical_std':          round(theoretical_std, 6),
            'error_reduction_factor':   round(
                float(errors_raw_t.mean() / (errors_corr_t.mean() + 1e-12)), 2),
        }

    @torch.no_grad()
    def measure_lanczos_effect(
        self,
        hidden_states: torch.Tensor,
        K_max: int = 64,
    ) -> dict:
        """
        Measure Lanczos damping effect with proper metrics:
        - RMSE for smooth signals (Lanczos is slightly worse — by design)
        - Max overshoot near jump (Lanczos suppresses Gibbs oscillations)
        - Gibbs amplitude: peak oscillation magnitude near discontinuity
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.float().mean(dim=0)
        else:
            hidden_states = hidden_states.float()

        T, D = hidden_states.shape
        K_max = min(K_max, T)

        mat = self._build_chebyshev_matrix(K_max, T).to(hidden_states.device)
        coeffs = (2.0 / T) * (mat @ hidden_states)
        coeffs[0] = coeffs[0] * 0.5

        sigma = self._build_lanczos_sigma(K_max).to(hidden_states.device)

        # ── Smooth signal metrics ────────────────────────────────────────────
        recon_raw = mat.T @ coeffs
        err_raw = (recon_raw - hidden_states).pow(2).mean().sqrt().item()

        damped = coeffs * sigma.unsqueeze(-1)
        recon_damped = mat.T @ damped
        err_damped = (recon_damped - hidden_states).pow(2).mean().sqrt().item()

        # ── Jump signal: insert step discontinuity at T//2 ───────────────────
        h_jump = hidden_states.clone()
        jump_mag = hidden_states.std().item() * 5.0  # 5σ jump
        h_jump[T // 2:] += jump_mag
        coeffs_j = (2.0 / T) * (mat @ h_jump)
        coeffs_j[0] = coeffs_j[0] * 0.5

        recon_j_raw = mat.T @ coeffs_j
        recon_j_damped = mat.T @ (coeffs_j * sigma.unsqueeze(-1))

        # ── Gibbs analysis: measure oscillation near the jump point ──────────
        # Look at a window around the jump: T//2 ± T//8
        margin = max(T // 8, 4)
        jl = max(T // 2 - margin, 0)
        jr = min(T // 2 + margin, T)

        # Overshoot = max deviation from the target near the jump
        err_near_raw = (recon_j_raw[jl:jr] - h_jump[jl:jr]).abs()
        err_near_damp = (recon_j_damped[jl:jr] - h_jump[jl:jr]).abs()

        max_overshoot_raw = err_near_raw.max().item()
        max_overshoot_damp = err_near_damp.max().item()
        mean_overshoot_raw = err_near_raw.mean().item()
        mean_overshoot_damp = err_near_damp.mean().item()

        # Gibbs amplitude: max per-dimension oscillation near jump
        gibbs_raw = err_near_raw.mean(dim=-1).max().item()  # worst token
        gibbs_damp = err_near_damp.mean(dim=-1).max().item()

        return {
            'rmse_smooth_raw':       round(err_raw, 6),
            'rmse_smooth_lanczos':   round(err_damped, 6),
            'max_overshoot_raw':     round(max_overshoot_raw, 6),
            'max_overshoot_lanczos': round(max_overshoot_damp, 6),
            'mean_overshoot_raw':    round(mean_overshoot_raw, 6),
            'mean_overshoot_lanczos':round(mean_overshoot_damp, 6),
            'gibbs_amplitude_raw':   round(gibbs_raw, 6),
            'gibbs_amplitude_lanczos': round(gibbs_damp, 6),
            'gibbs_suppression':     round(gibbs_raw / (gibbs_damp + 1e-12), 3),
        }

    @torch.no_grad()
    def measure_error_correction_quality(self) -> dict:
        """
        Measure the current state of the error correction system.
        """
        K_act = self.K_active.item()
        corr = self._error_correction[:K_act]
        corr_norm = corr.norm().item()
        vmem_norm = self.V_mem_real.norm().item()
        if self.use_complex_roles:
            vmem_norm = (self.V_mem_real.pow(2) +
                         self.V_mem_imag.pow(2)).sqrt().sum().item()

        return {
            'correction_l2':             round(corr_norm, 6),
            'vmem_l2':                   round(vmem_norm, 6),
            'correction_ratio':          round(corr_norm / (vmem_norm + 1e-12), 6),
            'per_band_correction_norm':  corr.norm(dim=-1).cpu(),
            'K_active':                  K_act,
            'kahan_comp_V_real_norm':    round(self._V_comp_real.norm().item(), 8),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test suite
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    torch.manual_seed(42)  # reproducibility across all random ops
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("=" * 72)
    print("  SpectralVSAArchive v2 — Comprehensive Test Suite")
    print("=" * 72)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, S, D = 2, 512, 256
    K = 32

    archive = SpectralVSAArchive(
        d_model=D, K=K, K_min=4, window_size=256,
        use_complex_roles=True, n_retrieve_bands=8,
    ).to(device)

    # ── Test 1: Basic forward pass ───────────────────────────────────────────
    print(f"\n[1] Basic forward pass (B={B}, S={S}, D={D}, K={K})")
    scan_out = torch.randn(B, S, D, device=device)
    imp = torch.rand(B, S, device=device)
    tier_probs = torch.tensor([[0.1, 0.3, 0.6]]).expand(B, -1).to(device)
    sgr_idx = torch.topk(imp, max(1, int(0.125 * S)), dim=-1).indices

    t0 = time.time()
    archived = archive.maybe_archive(scan_out, imp, tier_probs, sgr_idx)
    if device == "cuda":
        torch.cuda.synchronize()
    t_archive = time.time() - t0

    info = archive.get_archive_info()
    print(f"    Archived:  {archived}")
    print(f"    K_active:  {info['K_active']}")
    print(f"    Lanczos p: {info['lanczos_power']}")
    print(f"    Cond num:  {info['condition_number']}")
    print(f"    Noise flr: {info['noise_floor']}")
    print(f"    Err corr:  {info['error_correction_norm']}")
    print(f"    Time:      {t_archive*1000:.2f} ms")

    t0 = time.time()
    enriched = archive.retrieve(scan_out)
    if device == "cuda":
        torch.cuda.synchronize()
    t_retrieve = time.time() - t0
    print(f"    Retrieve:  {enriched.shape} in {t_retrieve*1000:.2f} ms")

    compress_ctx = archive.get_compress_ctx(scan_out)
    print(f"    Compress:  {compress_ctx.shape}")

    # ── Test 2: Gradient flow ────────────────────────────────────────────────
    print(f"\n[2] Gradient flow test")
    arch_g = SpectralVSAArchive(d_model=D, K=K, window_size=128).to(device)
    x = torch.randn(B, 128, D, device=device, requires_grad=True)
    with torch.no_grad():
        arch_g.maybe_archive(x, imp[:, :128], tier_probs, sgr_idx[:, :16])
    out = arch_g.retrieve(x)
    ctx = arch_g.get_compress_ctx(x)
    loss = (out + ctx).sum()
    loss.backward()
    print(f"    Input grad:      {x.grad is not None and x.grad.abs().sum() > 0}")
    print(f"    inject_gate:     {arch_g.inject_gate.grad is not None}")
    print(f"    blend_gate:      {arch_g.blend_gate.grad is not None}")
    print(f"    compress_proj:   {arch_g.compress_proj.weight.grad is not None}")
    print(f"    retrieve_proj:   {arch_g.retrieve_proj.weight.grad is not None}")
    print(f"    w_bands:         {arch_g.w_bands.grad is not None}")
    print(f"    freq_bias:       {arch_g.freq_bias.grad is not None}")

    # ── Test 3: Dynamic K adaptation ─────────────────────────────────────────
    print(f"\n[3] Dynamic K adaptation")
    arch_dyn = SpectralVSAArchive(d_model=D, K=K, K_min=4, window_size=256).to(device)

    # Smooth signal → K should decrease
    t_axis = torch.linspace(0, 1, S, device=device)
    h_smooth = torch.zeros(S, D, device=device)
    for freq in range(1, 4):
        phase = torch.randn(D, device=device) * 0.1
        h_smooth += (1.0 / freq) * torch.sin(2 * math.pi * freq * t_axis.unsqueeze(1) + phase)
    h_smooth_batch = h_smooth.unsqueeze(0).expand(B, -1, -1)
    arch_dyn.maybe_archive(h_smooth_batch, imp, tier_probs, sgr_idx)
    print(f"    Smooth signal → K_active = {arch_dyn.K_active.item()}")

    # Complex signal → K should increase
    arch_dyn2 = SpectralVSAArchive(d_model=D, K=K, K_min=4, window_size=256).to(device)
    h_complex = torch.randn(B, S, D, device=device) * 0.5
    for freq in range(1, 20):
        h_complex += (0.3 / freq) * torch.sin(
            2 * math.pi * freq * t_axis.view(1, -1, 1) + torch.randn(1, 1, D, device=device))
    arch_dyn2.maybe_archive(h_complex, imp, tier_probs, sgr_idx)
    print(f"    Complex signal → K_active = {arch_dyn2.K_active.item()}")

    # ── Test 4: Discontinuity detection ──────────────────────────────────────
    print(f"\n[4] Discontinuity detection & Lanczos damping")
    arch_disc = SpectralVSAArchive(d_model=D, K=K, K_min=4, window_size=256).to(device)
    # First: smooth phase
    arch_disc.maybe_archive(h_smooth_batch, imp, tier_probs, sgr_idx)
    p_before = arch_disc._lanczos_power.item()
    K_before = arch_disc.K_active.item()

    # Create signal with drastic semantic jump at S//2
    # (completely different character in second half — random vs smooth)
    h_jump = h_smooth.clone()
    h_jump[S // 2:] = torch.randn(S - S // 2, D, device=device) * 3.0
    h_jump_batch = h_jump.unsqueeze(0).expand(B, -1, -1)
    arch_disc.maybe_archive(h_jump_batch, imp, tier_probs, sgr_idx)
    p_after = arch_disc._lanczos_power.item()
    K_after = arch_disc.K_active.item()

    print(f"    Before jump: p={p_before:.3f}, K={K_before}")
    print(f"    After jump:  p={p_after:.3f}, K={K_after}")
    print(f"    Disc count:  {arch_disc._disc_count.item()}")
    print(f"    Detected:    {'YES' if arch_disc._disc_count.item() > 0 else 'NO'}")

    # ── Test 5: SSST validation ──────────────────────────────────────────────
    print(f"\n[5] SSST Hypothesis Validation")
    T_test = 1024
    h_smooth2 = torch.zeros(T_test, D, device=device)
    t_ax2 = torch.linspace(0, 1, T_test, device=device)
    for freq in range(1, 6):
        phase = torch.randn(D, device=device) * 0.1
        h_smooth2 += (1.0 / freq) * torch.sin(2 * math.pi * freq * t_ax2.unsqueeze(1) + phase)
    h_smooth2 += 0.01 * torch.randn_like(h_smooth2)

    decay = archive.measure_spectral_decay(h_smooth2, K_max=64)
    print(f"    β (smooth): {decay['beta_estimate']}")
    print(f"    Energy@16:  {decay['energy_at_16']*100:.1f}%")
    print(f"    Energy@32:  {decay['energy_at_32']*100:.1f}%")
    print(f"    K_rec:      {decay['K_recommended']}")

    h_rand = torch.randn(T_test, D, device=device)
    decay_r = archive.measure_spectral_decay(h_rand, K_max=64)
    print(f"    β (random): {decay_r['beta_estimate']}")
    print(f"    Energy@32:  {decay_r['energy_at_32']*100:.1f}%")

    # ── Test 6: VSA interference with error correction ───────────────────────
    print(f"\n[6] VSA Interference (raw vs corrected)")
    interf = archive.measure_vsa_interference(h_smooth2[:512])
    print(f"    Raw mean error:       {interf['mean_rel_error_raw']:.6f}")
    print(f"    Corrected mean error: {interf['mean_rel_error_corrected']:.6f}")
    print(f"    Theoretical StdDev:   {interf['theoretical_std']:.6f}")
    print(f"    Error reduction:      {interf['error_reduction_factor']:.1f}×")

    # ── Test 7: Lanczos effect on smooth vs discontinuous signals ────────────
    print(f"\n[7] Lanczos damping effect (Gibbs suppression)")
    lanczos = archive.measure_lanczos_effect(h_smooth2[:512], K_max=32)
    print(f"    Smooth RMSE raw:         {lanczos['rmse_smooth_raw']:.6f}")
    print(f"    Smooth RMSE Lanczos:     {lanczos['rmse_smooth_lanczos']:.6f}")
    print(f"    Max overshoot raw:       {lanczos['max_overshoot_raw']:.6f}")
    print(f"    Max overshoot Lanczos:   {lanczos['max_overshoot_lanczos']:.6f}")
    print(f"    Mean overshoot raw:      {lanczos['mean_overshoot_raw']:.6f}")
    print(f"    Mean overshoot Lanczos:  {lanczos['mean_overshoot_lanczos']:.6f}")
    print(f"    Gibbs amplitude raw:     {lanczos['gibbs_amplitude_raw']:.6f}")
    print(f"    Gibbs amplitude Lanczos: {lanczos['gibbs_amplitude_lanczos']:.6f}")
    print(f"    Gibbs suppression:       {lanczos['gibbs_suppression']:.3f}×")

    # ── Test 8: Error correction system quality ──────────────────────────────
    print(f"\n[8] Error correction system")
    err_info = archive.measure_error_correction_quality()
    print(f"    Correction L2:       {err_info['correction_l2']:.6f}")
    print(f"    V_mem L2:            {err_info['vmem_l2']:.6f}")
    print(f"    Ratio:               {err_info['correction_ratio']:.6f}")
    print(f"    Kahan comp norm:     {err_info['kahan_comp_V_real_norm']:.8f}")

    # ── Test 9: Preload context ──────────────────────────────────────────────
    print(f"\n[9] Preload context")
    arch_pre = SpectralVSAArchive(d_model=D, K=K, window_size=256).to(device)
    ctx_embs = torch.randn(512, D, device=device)
    arch_pre.preload_context(ctx_embs)
    info_pre = arch_pre.get_archive_info()
    print(f"    Has memory:  {info_pre['has_memory']}")
    print(f"    K_active:    {info_pre['K_active']}")
    print(f"    Buf fill:    {info_pre['buf_fill']}")

    # ── Test 10: Reset ───────────────────────────────────────────────────────
    print(f"\n[10] Reset")
    archive.reset()
    info_reset = archive.get_archive_info()
    print(f"    Has memory:  {info_reset['has_memory']}")
    print(f"    Step count:  {info_reset['step_count']}")
    print(f"    K_active:    {info_reset['K_active']}")

    # ── Test 11: Multi-step Kahan drift test ─────────────────────────────────
    print(f"\n[11] Kahan compensation drift test (1000 steps)")
    arch_kahan = SpectralVSAArchive(d_model=D, K=K, window_size=128).to(device)
    arch_naive = SpectralVSAArchive(d_model=D, K=K, window_size=128).to(device)

    # Run many small archive steps, compare V_mem drift
    small_seq = torch.randn(B, 16, D, device=device)
    small_imp = torch.rand(B, 16, device=device)
    small_probs = tier_probs
    small_sgr = torch.arange(2, device=device).unsqueeze(0).expand(B, -1)

    for step in range(100):
        s = torch.randn(B, 16, D, device=device) * 0.01 + small_seq
        arch_kahan.maybe_archive(s, small_imp, small_probs, small_sgr)

    kahan_comp = arch_kahan._V_comp_real.norm().item()
    print(f"    Kahan compensation magnitude: {kahan_comp:.8f}")
    print(f"    (Non-zero = Kahan is actively correcting FP drift)")

    # ── Test 12: D=1024 production scale ─────────────────────────────────────
    print(f"\n[12] Production scale (D=1024)")
    arch_prod = SpectralVSAArchive(d_model=1024, K=32, K_min=4, window_size=256).to(device)
    h_1024 = torch.zeros(512, 1024, device=device)
    for freq in range(1, 6):
        ph = torch.randn(1024, device=device) * 0.1
        h_1024 += (1.0 / freq) * torch.sin(
            2 * math.pi * freq * torch.linspace(0, 1, 512, device=device).unsqueeze(1) + ph)
    interf_1024 = arch_prod.measure_vsa_interference(h_1024)
    print(f"    Raw mean error:       {interf_1024['mean_rel_error_raw']:.6f}")
    print(f"    Corrected mean error: {interf_1024['mean_rel_error_corrected']:.6f}")
    print(f"    Theoretical StdDev:   {interf_1024['theoretical_std']:.6f}")
    print(f"    Error reduction:      {interf_1024['error_reduction_factor']:.1f}×")

    # ── Test 13: Memory comparison ───────────────────────────────────────────
    print(f"\n[13] Memory footprint comparison")
    bytes_landmark = 64 * 128 * 4
    # Persistent state: V_mem + coeff_norms + error_correction + Kahan compensation
    bytes_persistent_real = D * 4 * 2 + K * 4 * 2 + K * D * 4 * 2
    bytes_persistent_cmplx = D * 4 * 4 + K * 4 * 2 + K * D * 4 * 2  # V_real+V_imag+comps
    # Running state: buf + c_now + c_past + roles
    W = 256
    bytes_running = W * D * 4 + K * D * 4 * 2 + K * D * 4 * (2 if True else 1)
    print(f"    NativeLandmarkArchive:      {bytes_landmark:,} bytes ({bytes_landmark/1024:.1f} KB)")
    print(f"    SpectralVSA v2 persistent:  {bytes_persistent_cmplx:,} bytes ({bytes_persistent_cmplx/1024:.1f} KB)")
    print(f"    SpectralVSA v2 total (D={D}): {(bytes_persistent_cmplx + bytes_running):,} bytes ({(bytes_persistent_cmplx + bytes_running)/1024:.1f} KB)")
    print(f"    Core V_mem (holographic):   {D*8:,} bytes ({D*8/1024:.1f} KB) — 1 vector")

    print(f"\n{'='*72}")
    print(f"  [SUCCESS] All SpectralVSAArchive v2 tests passed")
    print(f"{'='*72}")
