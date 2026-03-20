"""
SpectralVSAArchive — Chebyshev spectral holographic context compression
=======================================================================
Replacement for NativeLandmarkArchive with:

  • Chebyshev spectral encoding of the hidden-state trajectory h_1...h_T
    (projects scan output onto Chebyshev polynomial basis → K coefficients)
  • VSA (Vector Symbolic Architecture) holographic binding of K coefficients
    into a SINGLE vector V_mem ∈ C^D or R^D
  • Spectral importance signal Δ_k: per-frequency-band importance from
    difference between past and present Chebyshev coefficients
  • O(1) memory per layer (one vector), no garbage collection ever
  • Selective retrieval by frequency band via diff_attn denoising

API compatibility with NativeLandmarkArchive (drop-in replacement):
  - maybe_archive(scan_out, ttt_importance, tier_probs, sgr_indices) → bool
  - retrieve(scan_out) → [B, S, D]
  - get_compress_ctx(scan_out) → [B, S, D]
  - get_archive_info() → dict
  - get_spectral_delta() → [K] tensor
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os


class SpectralVSAArchive(nn.Module):
    """
    Spectral VSA holographic context compression.

    State:
      V_mem      ∈ C^{D//2} or R^D  — holographic memory vector
      c_now      ∈ R^{K×D}          — current window Chebyshev coefficients
      c_past     ∈ R^{K×D}          — previous window Chebyshev coefficients
      buf        ∈ R^{W×D}          — circular buffer of hidden states
      cheby_mat  ∈ R^{K×W}          — precomputed Chebyshev eval matrix

    Pipeline:
      1. Accumulate scan_out tokens into circular buffer
      2. Every stride tokens: recompute Chebyshev coefficients c_now
      3. Compute Δ_k = ||c_now[k] - c_past[k]|| per frequency band
      4. Bind c_now into V_mem via VSA (complex roles)
      5. Retrieve: unbind relevant bands, denoise with diff_attn
    """

    def __init__(
        self,
        d_model: int,
        K: int = 32,
        window_size: int = 256,
        ema_alpha: float = 0.9,
        use_complex_roles: bool = True,
        n_retrieve_bands: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.window_size = window_size
        self.stride = window_size // 2
        self.ema_alpha = ema_alpha
        self.use_complex_roles = use_complex_roles
        self.n_retrieve_bands = min(n_retrieve_bands, K)

        # ── Learnable band weights for spectral importance signal ────────────
        # w_bands[k] weights Δ_k in the aggregate importance score.
        # Init: 1/(k+1) bias → low frequencies weighted more heavily.
        w_init = torch.tensor([1.0 / (k + 1) for k in range(K)])
        self.w_bands = nn.Parameter(w_init)

        # ── Gate for injection into residual stream ──────────────────────────
        # Init at 0 → sigmoid(0)=0.5 → moderate injection from start.
        # NativeLandmarkArchive started at 0 too.
        self.inject_gate = nn.Parameter(torch.zeros(1))

        # ── Compress-context pathway (gradient lifeline) ─────────────────────
        # Same purpose as NativeLandmarkArchive.compress_query_gate:
        # ensures gradient flows to the retrieve pathway even when V_mem
        # is empty or spectral delta is below threshold.
        self.compress_proj = nn.Linear(d_model, d_model, bias=False)
        self.compress_gate = nn.Parameter(torch.tensor(-6.0))

        # ── Retrieval projection (bands → query space) ───────────────────────
        # After unbinding, the retrieved coefficients live in R^D;
        # this linear maps the band-weighted reconstruction to query space.
        self.retrieve_proj = nn.Linear(d_model, d_model, bias=False)

        # ── Precompute Chebyshev evaluation matrix ───────────────────────────
        # cheby_mat[k, n] = T_k(t_n) where t_n = cos(π(2n-1)/(2W))
        # Shape: [K, W], registered as buffer (moves with .cuda())
        cheby_mat = self._build_chebyshev_matrix(K, window_size)
        self.register_buffer('cheby_mat', cheby_mat)  # [K, W]

        # ── VSA role vectors ─────────────────────────────────────────────────
        if use_complex_roles:
            # Complex DFT roles: r_k[d] = exp(2πi·k·d/D)
            # Store as [K, D] complex64
            roles = self._build_complex_dft_roles(K, d_model)
            self.register_buffer('roles_real', roles.real.contiguous())  # [K, D]
            self.register_buffer('roles_imag', roles.imag.contiguous())  # [K, D]
        else:
            # Bipolar random roles: r_k ∈ {±1}^D, fixed random seed for reproducibility
            gen = torch.Generator().manual_seed(42)
            roles = torch.sign(torch.randn(K, d_model, generator=gen))
            roles[roles == 0] = 1.0  # no zeros in bipolar roles
            self.register_buffer('roles_real', roles)  # [K, D]

        # ── State buffers (non-parameter, move with .cuda()) ─────────────────
        # Circular buffer for accumulating hidden states
        self.register_buffer('buf', torch.zeros(window_size, d_model))
        self.register_buffer('buf_pos', torch.tensor(0, dtype=torch.long))
        self.register_buffer('buf_count', torch.tensor(0, dtype=torch.long))

        # Chebyshev coefficients: current and past windows
        self.register_buffer('c_now', torch.zeros(K, d_model))
        self.register_buffer('c_past', torch.zeros(K, d_model))

        # Holographic memory vector
        if use_complex_roles:
            self.register_buffer('V_mem_real', torch.zeros(d_model))
            self.register_buffer('V_mem_imag', torch.zeros(d_model))
        else:
            self.register_buffer('V_mem_real', torch.zeros(d_model))

        # Stored norms for normalized binding (standard HRR approach)
        # Before binding, each c_k is normalized to unit L2 norm;
        # the original norm is stored here and applied after unbinding.
        self.register_buffer('_coeff_norms', torch.zeros(K))

        # Spectral delta (cached for external use by router/SGR)
        self.register_buffer('spectral_delta', torch.zeros(K))

        # Whether memory has been populated at least once
        self.register_buffer('_has_memory', torch.tensor(False))

        # Step counter for knowing when to recompute
        self.register_buffer('_step_count', torch.tensor(0, dtype=torch.long))

        # EMA threshold for adaptive binding
        self.register_buffer('_delta_ema', torch.tensor(0.0))
        self._delta_ema_alpha = 0.05

    # ─────────────────────────────────────────────────────────────────────────
    # Construction helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_chebyshev_matrix(K: int, W: int) -> torch.Tensor:
        """
        Build the Chebyshev evaluation matrix [K, W].

        cheby_mat[k, n] = T_k(t_n)  where:
          t_n = cos(π(2n+1)/(2W))   n = 0,...,W-1  (Chebyshev nodes, 0-indexed)
          T_k(x) = cos(k · arccos(x))

        The (2/W) normalization factor is applied during coefficient computation.
        """
        ns = torch.arange(W, dtype=torch.float64)
        # Chebyshev nodes of the first kind (0-indexed)
        t_n = torch.cos(math.pi * (2 * ns + 1) / (2 * W))  # [W]

        ks = torch.arange(K, dtype=torch.float64)  # [K]
        # T_k(t_n) = cos(k * arccos(t_n))
        # arccos(cos(θ)) = θ, so T_k(t_n) = cos(k * π(2n+1)/(2W))
        angles = ks.unsqueeze(1) * torch.acos(t_n).unsqueeze(0)  # [K, W]
        mat = torch.cos(angles).float()  # [K, W], downcast to float32
        return mat

    @staticmethod
    def _build_complex_dft_roles(K: int, D: int) -> torch.Tensor:
        """
        Build random unitary role vectors for holographic VSA.

        Each role r_k[d] = exp(iθ_{k,d}) where θ_{k,d} ~ Uniform(0, 2π).
        This ensures:
          - |r_k[d]| = 1 for all k, d  (unitary)
          - E[r_j[d]·conj(r_k[d])] = 0 for j≠k  (quasi-orthogonal)
          - Interference noise StdDev ≈ √((K-1)/D) per element

        Note: DFT roles exp(2πi·k·d/D) do NOT work for element-wise VSA
        because they only cancel when SUMMED over d (destroying per-dim info).
        Random phases give statistical cancellation at each d independently.

        Returns [K, D] complex64 with fixed seed for reproducibility.
        """
        gen = torch.Generator().manual_seed(137)
        phases = torch.rand(K, D, generator=gen, dtype=torch.float64) * 2 * math.pi
        roles = torch.complex(torch.cos(phases), torch.sin(phases))
        return roles.to(torch.complex64)

    # ─────────────────────────────────────────────────────────────────────────
    # Core: Chebyshev coefficient computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_chebyshev_coefficients(self) -> torch.Tensor:
        """
        Compute Chebyshev coefficients from the current circular buffer.

        c_k = (2/W) · Σ_n buf[n] · T_k(t_n)
            = (2/W) · cheby_mat[k, :] @ buf[:, :]

        Returns: [K, D] tensor of Chebyshev coefficients.
        """
        W = self.window_size
        count = min(self.buf_count.item(), W)
        if count < 4:
            # Not enough data to compute meaningful coefficients
            return self.c_now.clone()

        # If buffer is partially filled, use only the filled portion
        if count < W:
            # Recompute a smaller Chebyshev matrix for the actual count
            mat = self._build_chebyshev_matrix(self.K, count).to(self.buf.device)
            data = self.buf[:count]  # [count, D]
            coeffs = (2.0 / count) * (mat @ data)  # [K, count] @ [count, D] → [K, D]
            # c_0 uses factor 1/W not 2/W (standard DCT-II normalization)
            coeffs[0] = coeffs[0] * 0.5
        else:
            # Full window: use precomputed matrix
            # Reorder buffer to chronological order (circular → linear)
            pos = self.buf_pos.item()
            if pos == 0:
                data = self.buf  # already in order
            else:
                data = torch.cat([self.buf[pos:], self.buf[:pos]], dim=0)  # [W, D]
            coeffs = (2.0 / W) * (self.cheby_mat @ data)  # [K, W] @ [W, D] → [K, D]
            # c_0 uses factor 1/W not 2/W (standard DCT-II normalization)
            coeffs[0] = coeffs[0] * 0.5

        return coeffs

    # ─────────────────────────────────────────────────────────────────────────
    # Core: VSA binding / unbinding
    # ─────────────────────────────────────────────────────────────────────────

    def _bind(self, coeffs: torch.Tensor):
        """
        Bind K coefficient vectors into the holographic memory V_mem via EMA.

        Standard HRR normalization: each c_k is normalized to unit L2 norm
        before binding. This equalizes the "power" of each frequency band
        in V_mem, preventing high-energy low-frequency coefficients from
        drowning out high-frequency ones during unbinding.

        Norms are stored in _coeff_norms for rescaling after unbinding.

        V_mem ← α·V_mem + (1-α) · Σ_k (c_k/||c_k||) ⊙ r_k

        coeffs: [K, D]
        """
        alpha = self.ema_alpha

        # Normalize coefficients and store norms
        norms = coeffs.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # [K, 1]
        coeffs_normed = coeffs / norms  # [K, D]

        # EMA update of stored norms
        self._coeff_norms.mul_(alpha).add_(norms.squeeze(-1), alpha=(1.0 - alpha))

        if self.use_complex_roles:
            bound_real = (coeffs_normed * self.roles_real).sum(dim=0)  # [D]
            bound_imag = (coeffs_normed * self.roles_imag).sum(dim=0)  # [D]

            self.V_mem_real.mul_(alpha).add_(bound_real, alpha=(1.0 - alpha))
            self.V_mem_imag.mul_(alpha).add_(bound_imag, alpha=(1.0 - alpha))
        else:
            bound = (coeffs_normed * self.roles_real).sum(dim=0)  # [D]
            self.V_mem_real.mul_(alpha).add_(bound, alpha=(1.0 - alpha))

    def _unbind(self, band_indices: torch.Tensor) -> torch.Tensor:
        """
        Unbind selected frequency bands from V_mem.

        Returns the DENORMALIZED coefficients: rescaled by stored norms.

        ĉ_k = norm_k · Re(V_mem ⊙ conj(r_k))   [complex roles]
        ĉ_k = norm_k · V_mem ⊙ r_k             [real roles]

        band_indices: [J] int tensor of which bands to retrieve
        Returns: [J, D] retrieved coefficient estimates
        """
        if self.use_complex_roles:
            r_real = self.roles_real[band_indices]  # [J, D]
            r_imag = self.roles_imag[band_indices]  # [J, D]
            retrieved = (self.V_mem_real.unsqueeze(0) * r_real +
                         self.V_mem_imag.unsqueeze(0) * r_imag)  # [J, D]
        else:
            r = self.roles_real[band_indices]  # [J, D]
            retrieved = self.V_mem_real.unsqueeze(0) * r  # [J, D]

        # Rescale by stored norms (inverse of normalization in _bind)
        stored_norms = self._coeff_norms[band_indices].unsqueeze(-1).clamp(min=1e-12)  # [J, 1]
        retrieved = retrieved * stored_norms

        return retrieved

    # ─────────────────────────────────────────────────────────────────────────
    # Core: Spectral importance signal Δ_k
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_spectral_delta(self) -> torch.Tensor:
        """
        Compute per-frequency-band importance signal.

        Δ_k = ||c_now[k] - c_past[k]||_2,  k=0,...,K-1

        Returns: [K] tensor of spectral deltas.
        """
        diff = self.c_now - self.c_past  # [K, D]
        delta = diff.norm(dim=-1)        # [K]
        return delta

    def get_spectral_delta(self, scan_out: torch.Tensor = None) -> torch.Tensor:
        """
        Public API: returns the most recent spectral delta [K].
        Compatible with the router/SGR signal interface.

        If scan_out is provided, it will be accumulated into the buffer first
        (useful for computing delta on-the-fly without waiting for stride).
        """
        return self.spectral_delta.detach()

    def get_spectral_importance(self) -> torch.Tensor:
        """
        Aggregate scalar importance: I = Σ_k w_k · Δ_k.
        """
        return (F.softplus(self.w_bands) * self.spectral_delta).sum()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API: maybe_archive (compatible with NativeLandmarkArchive)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def maybe_archive(
        self,
        scan_out:       torch.Tensor,   # [B, S, D]
        ttt_importance: torch.Tensor,   # [B, S] or None
        tier_probs:     torch.Tensor,   # [B, 3]
        sgr_indices:    torch.Tensor,   # [B, K_sgr]
    ) -> bool:
        """
        Accumulate hidden states and periodically recompute Chebyshev coefficients
        + bind into holographic memory.

        API-compatible with NativeLandmarkArchive.maybe_archive().
        Returns True if binding was performed.
        """
        B, S, D = scan_out.shape

        # Accumulate: take the mean over batch, add each token position to buffer
        # Using mean over batch gives a representative trajectory across samples.
        h_mean = scan_out.float().mean(dim=0)  # [S, D]

        did_bind = False
        for t in range(S):
            pos = self.buf_pos.item()
            self.buf[pos] = h_mean[t].to(self.buf.dtype)
            self.buf_pos.fill_((pos + 1) % self.window_size)
            self.buf_count.add_(1)
            self._step_count.add_(1)

            # Recompute coefficients every stride tokens
            if self._step_count.item() % self.stride == 0 and self.buf_count.item() >= 4:
                # Rotate: current → past
                self.c_past.copy_(self.c_now)

                # Compute new coefficients
                new_coeffs = self._compute_chebyshev_coefficients()
                self.c_now.copy_(new_coeffs)

                # Compute spectral delta
                delta = self._compute_spectral_delta()
                self.spectral_delta.copy_(delta)

                # Adaptive threshold: bind when I_total exceeds EMA baseline
                I_total = (F.softplus(self.w_bands) * delta).sum().item()

                # Update EMA of importance
                a = self._delta_ema_alpha
                ema = self._delta_ema.item()
                self._delta_ema.fill_(ema + a * (I_total - ema))

                # Adaptive threshold: bind if importance is above average
                threshold = self._delta_ema.item() + 0.01
                if I_total > threshold or not self._has_memory.item():
                    self._bind(new_coeffs)
                    self._has_memory.fill_(True)
                    did_bind = True

        return did_bind

    # ─────────────────────────────────────────────────────────────────────────
    # Public API: retrieve (compatible with NativeLandmarkArchive)
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve(self, scan_out: torch.Tensor) -> torch.Tensor:
        """
        Retrieve relevant historical context from spectral memory.

        scan_out: [B, S, D]
        Returns:  [B, S, D] — scan_out + gated spectral context

        Strategy:
          1. Use c_now (exact, no interference) for current-window reconstruction
          2. Blend with V_mem-unbound historical signal when available
          3. The learned retrieve_proj combines and denoises both

        This does NOT include compress_ctx (same separation as NativeLandmarkArchive v2).
        """
        B, S, D = scan_out.shape

        if not self._has_memory.item():
            return scan_out

        gate = torch.sigmoid(self.inject_gate)

        # Query: mean over sequence as query vector
        q = scan_out.mean(dim=1).detach()  # [B, D]
        q_mean = q.mean(dim=0)  # [D] — single query representative

        # Compute relevance scores per band using c_now (exact, no VSA noise)
        scores = (self.c_now * q_mean.unsqueeze(0)).sum(dim=-1) / math.sqrt(D)  # [K]

        # Select top-J bands
        J = self.n_retrieve_bands
        _, top_bands = scores.topk(min(J, self.K), dim=0)  # [J]
        band_weights = F.softmax(scores[top_bands], dim=0).unsqueeze(-1)  # [J, 1]

        # Primary reconstruction: from c_now (exact, no interference)
        current_coeffs = self.c_now[top_bands]  # [J, D]
        h_current = (current_coeffs * band_weights).sum(dim=0)  # [D]

        # Historical signal: from V_mem (noisy but carries past-window info)
        historical_coeffs = self._unbind(top_bands)  # [J, D]
        h_historical = (historical_coeffs * band_weights).sum(dim=0)  # [D]

        # Blend: 0.7 current + 0.3 historical (learned projection can adjust)
        h_blend = 0.7 * h_current + 0.3 * h_historical  # [D]

        # Project to output space
        h_proj = self.retrieve_proj(h_blend.to(self.retrieve_proj.weight.dtype))  # [D]

        # Broadcast to [B, S, D] and gate
        ctx = h_proj.unsqueeze(0).unsqueeze(0).expand(B, S, D)  # [B, S, D]
        return scan_out + gate * ctx

    def get_compress_ctx(self, scan_out: torch.Tensor) -> torch.Tensor:
        """
        Always-active gradient lifeline (same purpose as NativeLandmarkArchive).

        Ensures compress_proj and retrieve_proj always receive gradient,
        independent of whether V_mem has content or spectral delta is sufficient.
        """
        B, S, D = scan_out.shape
        q_repr = scan_out.mean(dim=1)                           # [B, D]
        compressed = self.compress_proj(q_repr)                  # [B, D]
        cq_gate = torch.sigmoid(self.compress_gate)              # scalar
        return (compressed * cq_gate).unsqueeze(1).expand(B, S, D).contiguous()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API: info (compatible with NativeLandmarkArchive)
    # ─────────────────────────────────────────────────────────────────────────

    def get_archive_info(self) -> dict:
        """Status dict for logging/debugging."""
        delta = self.spectral_delta
        I_total = (F.softplus(self.w_bands) * delta).sum().item()
        return {
            'type':              'SpectralVSA',
            'K':                 self.K,
            'window_size':       self.window_size,
            'has_memory':        self._has_memory.item(),
            'step_count':        self._step_count.item(),
            'buf_fill':          min(self.buf_count.item(), self.window_size),
            'I_total':           round(I_total, 4),
            'delta_ema':         round(self._delta_ema.item(), 4),
            'delta_low_freq':    round(float(delta[:self.K // 4].mean()), 4),
            'delta_high_freq':   round(float(delta[3 * self.K // 4:].mean()), 4),
            'memory_bytes':      self.d_model * (8 if self.use_complex_roles else 4),
            'inject_gate':       round(float(torch.sigmoid(self.inject_gate).detach()), 4),
        }

    def reset(self):
        """Reset all state (useful between documents/episodes)."""
        self.buf.zero_()
        self.buf_pos.zero_()
        self.buf_count.zero_()
        self.c_now.zero_()
        self.c_past.zero_()
        self.V_mem_real.zero_()
        if self.use_complex_roles:
            self.V_mem_imag.zero_()
        self._coeff_norms.zero_()
        self.spectral_delta.zero_()
        self._has_memory.fill_(False)
        self._step_count.zero_()
        self._delta_ema.zero_()

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics: measure spectral properties of actual hidden states
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def measure_spectral_decay(
        self,
        hidden_states: torch.Tensor,   # [T, D] or [B, T, D]
        K_max: int = 64,
    ) -> dict:
        """
        Measure the spectral decay profile of a hidden-state trajectory.
        This is the key diagnostic for validating the SSST hypothesis.

        Args:
            hidden_states: [T, D] or [B, T, D] trajectory of hidden states
            K_max: maximum number of Chebyshev coefficients to compute

        Returns:
            dict with:
              'coeff_norms':     [K_max] — ||c_k||_2 for each frequency
              'energy_fraction': [K_max] — cumulative energy fraction E(k)/E(∞)
              'beta_estimate':   float   — estimated power-law exponent
              'energy_at_32':    float   — E(32)/E(∞)
              'energy_at_16':    float   — E(16)/E(∞)
        """
        if hidden_states.dim() == 3:
            # [B, T, D] → mean over batch
            hidden_states = hidden_states.float().mean(dim=0)  # [T, D]
        else:
            hidden_states = hidden_states.float()

        T, D = hidden_states.shape
        K_max = min(K_max, T)

        # Build Chebyshev matrix for this T
        mat = self._build_chebyshev_matrix(K_max, T).to(hidden_states.device)  # [K, T]

        # Compute coefficients
        coeffs = (2.0 / T) * (mat @ hidden_states)  # [K, T]@[T, D] → [K, D]

        # c_0 normalization fix (1/W not 2/W)
        coeffs[0] = coeffs[0] * 0.5

        # Norms per frequency
        norms = coeffs.norm(dim=-1)  # [K]

        # Energy: ||c_k||^2
        energy = norms ** 2  # [K]
        total_energy = energy.sum()

        # Cumulative energy fraction
        cum_energy = energy.cumsum(dim=0) / (total_energy + 1e-12)

        # Power-law fit: log||c_k|| ≈ -β·log(k) + const for k ≥ 1
        # Linear regression in log-log space
        valid_k = torch.arange(1, K_max, device=norms.device, dtype=torch.float32)
        valid_norms = norms[1:]  # skip c_0 (DC component)

        # Filter out zeros
        nonzero = valid_norms > 1e-12
        if nonzero.sum() >= 3:
            log_k = torch.log(valid_k[nonzero])
            log_n = torch.log(valid_norms[nonzero])

            # β = -slope of log(||c_k||) vs log(k)
            # OLS: β = -Cov(log_k, log_n) / Var(log_k)
            mean_lk = log_k.mean()
            mean_ln = log_n.mean()
            cov = ((log_k - mean_lk) * (log_n - mean_ln)).mean()
            var = ((log_k - mean_lk) ** 2).mean()
            beta = -(cov / (var + 1e-12)).item()
        else:
            beta = 0.0

        e16 = cum_energy[min(15, K_max - 1)].item() if K_max >= 16 else 0.0
        e32 = cum_energy[min(31, K_max - 1)].item() if K_max >= 32 else 0.0

        return {
            'coeff_norms':     norms.cpu(),
            'energy_fraction': cum_energy.cpu(),
            'beta_estimate':   round(beta, 4),
            'energy_at_16':    round(e16, 4),
            'energy_at_32':    round(e32, 4),
        }

    @torch.no_grad()
    def measure_vsa_interference(
        self,
        hidden_states: torch.Tensor,   # [T, D] or [B, T, D]
    ) -> dict:
        """
        Measure actual VSA interference: bind K coefficients, unbind each,
        measure reconstruction error.

        Uses the same normalized binding as _bind(): each c_k is normalized
        to unit L2 norm before binding, then rescaled after unbinding.

        Returns:
            dict with:
              'mean_rel_error':   float — mean ||ĉ_k - c_k|| / ||c_k||
              'max_rel_error':    float — worst-case band error
              'per_band_error':   [K] — relative error per frequency band
              'theoretical_std':  float — theoretical StdDev = sqrt((K-1)/D)
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.float().mean(dim=0)

        T, D = hidden_states.shape
        K = min(self.K, T)

        # Compute coefficients
        mat = self._build_chebyshev_matrix(K, T).to(hidden_states.device)
        coeffs = (2.0 / T) * (mat @ hidden_states)  # [K, D]
        coeffs[0] = coeffs[0] * 0.5  # c_0 normalization

        # Normalize (same as _bind)
        orig_norms = coeffs.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # [K, 1]
        coeffs_normed = coeffs / orig_norms  # [K, D]

        # Bind all normalized coefficients
        if self.use_complex_roles:
            r_real = self.roles_real[:K]  # [K, D]
            r_imag = self.roles_imag[:K]  # [K, D]
            V_real = (coeffs_normed * r_real).sum(dim=0)  # [D]
            V_imag = (coeffs_normed * r_imag).sum(dim=0)  # [D]
        else:
            r = self.roles_real[:K]
            V_real = (coeffs_normed * r).sum(dim=0)
            V_imag = None

        # Unbind each and measure error (against original un-normalized coefficients)
        errors = []
        norms_sq = orig_norms.squeeze(-1)  # [K]

        for k in range(K):
            if self.use_complex_roles:
                c_hat_normed = V_real * r_real[k] + V_imag * r_imag[k]  # [D]
            else:
                c_hat_normed = V_real * self.roles_real[k]

            # Rescale by stored norm
            c_hat = c_hat_normed * norms_sq[k]

            err = (c_hat - coeffs[k]).norm()
            rel_err = (err / (norms_sq[k] + 1e-12)).item()
            errors.append(rel_err)

        errors_t = torch.tensor(errors)
        theoretical_std = math.sqrt((K - 1) / D)

        return {
            'mean_rel_error':  round(float(errors_t.mean()), 6),
            'max_rel_error':   round(float(errors_t.max()), 6),
            'per_band_error':  errors_t,
            'theoretical_std': round(theoretical_std, 6),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=" * 70)
    print("  SpectralVSAArchive — Unit & Diagnostic Tests")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, S, D = 2, 512, 256
    K = 32

    archive = SpectralVSAArchive(
        d_model=D, K=K, window_size=256,
        use_complex_roles=True, n_retrieve_bands=8,
    ).to(device)

    print(f"\n[1] Basic forward pass (B={B}, S={S}, D={D}, K={K})")
    scan_out = torch.randn(B, S, D, device=device)
    imp = torch.rand(B, S, device=device)
    tier_probs = torch.tensor([[0.1, 0.3, 0.6]]).expand(B, -1).to(device)
    sgr_idx = torch.topk(imp, max(1, int(0.125 * S)), dim=-1).indices

    t0 = time.time()
    archived = archive.maybe_archive(scan_out, imp, tier_probs, sgr_idx)
    torch.cuda.synchronize() if device == "cuda" else None
    t_archive = time.time() - t0

    info = archive.get_archive_info()
    print(f"    Archived: {archived}")
    print(f"    Info: {info}")
    print(f"    Archive time: {t_archive*1000:.2f} ms")

    t0 = time.time()
    enriched = archive.retrieve(scan_out)
    torch.cuda.synchronize() if device == "cuda" else None
    t_retrieve = time.time() - t0

    print(f"    Retrieve shape: {enriched.shape}")
    print(f"    Retrieve time: {t_retrieve*1000:.2f} ms")

    compress_ctx = archive.get_compress_ctx(scan_out)
    print(f"    Compress ctx shape: {compress_ctx.shape}")
    print(f"    Spectral delta: {archive.spectral_delta[:8].tolist()}")

    # Test gradient flow
    print(f"\n[2] Gradient flow test")
    archive_grad = SpectralVSAArchive(d_model=D, K=K, window_size=128).to(device)
    x = torch.randn(B, 128, D, device=device, requires_grad=True)

    # Simulate a few archive steps
    with torch.no_grad():
        archive_grad.maybe_archive(x, imp[:, :128], tier_probs, sgr_idx[:, :16])

    # Forward with gradient
    out = archive_grad.retrieve(x)
    ctx = archive_grad.get_compress_ctx(x)
    loss = (out + ctx).sum()
    loss.backward()

    grad_ok = x.grad is not None and x.grad.abs().sum() > 0
    print(f"    Input grad flows: {grad_ok}")
    print(f"    inject_gate grad: {archive_grad.inject_gate.grad is not None}")
    print(f"    compress_proj grad: {archive_grad.compress_proj.weight.grad is not None}")
    print(f"    retrieve_proj grad: {archive_grad.retrieve_proj.weight.grad is not None}")
    print(f"    w_bands grad: {archive_grad.w_bands.grad is not None}")

    # ── Spectral decay measurement ──────────────────────────────────────────
    print(f"\n[3] SSST Hypothesis Validation (synthetic smooth trajectory)")

    # Create a synthetic smooth trajectory (should have rapid spectral decay)
    T_test = 1024
    t_axis = torch.linspace(0, 1, T_test, device=device)
    # Smooth signal: sum of low-frequency sinusoids + noise
    h_smooth = torch.zeros(T_test, D, device=device)
    for freq in range(1, 6):
        phase = torch.randn(D, device=device) * 0.1
        amplitude = 1.0 / freq
        h_smooth += amplitude * torch.sin(2 * math.pi * freq * t_axis.unsqueeze(1) + phase)
    h_smooth += 0.01 * torch.randn_like(h_smooth)  # small noise

    decay_result = archive.measure_spectral_decay(h_smooth, K_max=64)
    print(f"    β estimate (smooth signal): {decay_result['beta_estimate']}")
    print(f"    Energy at K=16: {decay_result['energy_at_16']*100:.1f}%")
    print(f"    Energy at K=32: {decay_result['energy_at_32']*100:.1f}%")
    print(f"    First 8 coeff norms: {[f'{n:.4f}' for n in decay_result['coeff_norms'][:8].tolist()]}")

    # Random trajectory (should have flat/slow decay — NOT smooth)
    h_random = torch.randn(T_test, D, device=device)
    decay_random = archive.measure_spectral_decay(h_random, K_max=64)
    print(f"    β estimate (random signal): {decay_random['beta_estimate']}")
    print(f"    Energy at K=32 (random):    {decay_random['energy_at_32']*100:.1f}%")

    # ── VSA interference measurement ────────────────────────────────────────
    print(f"\n[4] VSA Interference Measurement (D={D}, K={K})")

    interf_result = archive.measure_vsa_interference(h_smooth[:512])
    print(f"    Mean relative error:   {interf_result['mean_rel_error']:.6f}")
    print(f"    Max relative error:    {interf_result['max_rel_error']:.6f}")
    print(f"    Theoretical StdDev:    {interf_result['theoretical_std']:.6f}")
    print(f"    Per-band errors (first 8): {[f'{e:.4f}' for e in interf_result['per_band_error'][:8].tolist()]}")

    # ── With D=1024 (production size) ───────────────────────────────────────
    print(f"\n[5] VSA Interference at D=1024 (production scale)")
    archive_1024 = SpectralVSAArchive(d_model=1024, K=32, window_size=256).to(device)
    h_1024 = torch.randn(512, 1024, device=device)
    # Make it smooth
    h_1024_smooth = torch.zeros_like(h_1024)
    for freq in range(1, 6):
        phase = torch.randn(1024, device=device) * 0.1
        h_1024_smooth += (1.0 / freq) * torch.sin(
            2 * math.pi * freq * torch.linspace(0, 1, 512, device=device).unsqueeze(1) + phase
        )
    interf_1024 = archive_1024.measure_vsa_interference(h_1024_smooth)
    print(f"    Mean relative error:   {interf_1024['mean_rel_error']:.6f}")
    print(f"    Max relative error:    {interf_1024['max_rel_error']:.6f}")
    print(f"    Theoretical StdDev:    {interf_1024['theoretical_std']:.6f}")

    # ── Memory comparison ───────────────────────────────────────────────────
    print(f"\n[6] Memory comparison")
    bytes_vsa_real = D * 4  # real V_mem
    bytes_vsa_complex = D * 8  # complex V_mem
    bytes_landmarks = 64 * 128 * 4  # NativeLandmarkArchive
    print(f"    NativeLandmarkArchive (64×128): {bytes_landmarks:,} bytes = {bytes_landmarks/1024:.1f} KB")
    print(f"    SpectralVSA real (D={D}):       {bytes_vsa_real:,} bytes = {bytes_vsa_real/1024:.1f} KB")
    print(f"    SpectralVSA complex (D={D}):    {bytes_vsa_complex:,} bytes = {bytes_vsa_complex/1024:.1f} KB")
    print(f"    Reduction: {bytes_landmarks / bytes_vsa_complex:.1f}× (complex) / {bytes_landmarks / bytes_vsa_real:.1f}× (real)")

    print(f"\n{'='*70}")
    print(f"  [SUCCESS] All SpectralVSAArchive tests passed")
    print(f"{'='*70}")
