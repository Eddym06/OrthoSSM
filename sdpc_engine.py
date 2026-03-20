"""
OrthoSSM V10 "Lightning" — Spectral Dual-Path Context Engine
=============================================================
Main engine module with length-based routing and AsyncLightBus.

V10 changes from V9:
  ---------------------------------------------------------
  LION OPTIMIZER: Single momentum buffer (no m2).
    - State per layer: (coeffs, momentum) instead of (coeffs, m1, m2)
    - 33% less memory for recurrent state

  LENGTH ROUTING: Three-tier adaptive path selection.
    - seq < 384:  Fast path  (no TTT, no SLR, no landmarks)
    - seq < 1024: Hybrid path (TTT chunk=32, SLR on all tokens)
    - seq >= 1024: Full Lightning path (everything enabled)

  ASYNC LIGHTBUS: Replaces CrossLayerMemoryBus.
    - 64-dim summary vector instead of full landmark cross-attention
    - Asynchronous residual update (no global dependency)
    - Layers execute nearly independently

  CONFIGURABLE DEGREE: Default 4 (was 8).
    - 50% less register pressure in Triton kernels
    - Sufficient for >99% of real-world signals
    - Degree 8 available via ultra_long_mode=True

  SLR (Spectral Local Refiner): Replaces NSA.
    - Spectral-routed differential local attention
    - Only top-12.5% tokens (by TTT error) enter attention
    - ~64% parameter reduction vs NSA
    - SLR overhead eliminated for seq < 384
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from sdpc_kernel import (
    apply_cheby_rkv_v10,
    GatedComplexityPredictor,
    init_chebyshev_coefficients,
)
from landmark_archive import AsyncLightBus, LandmarkArchive
from slr_module import SpectralLocalRefiner
from ortho_diagnostics import DIAG


# ============================================================================
# LENGTH ROUTING THRESHOLDS
# ============================================================================
FAST_PATH_THRESHOLD = 384      # seq < 384: pure Mamba-style (no TTT, no SLR)
HYBRID_PATH_THRESHOLD = 1024   # seq < 1024: SLR on all tokens, chunked TTT
# seq >= 1024: full Lightning path (SLR with spectral routing)


class SpectralDualPathContextEngine(nn.Module):
    """
    V10 "Lightning" Engine: Length-routed spectral dual-path.

    Three execution tiers based on sequence length:
      Fast (<384):    Chebyshev + EMA only. No TTT, no SLR, no landmarks.
      Hybrid (<1024): Chebyshev + TTT + SLR (all tokens, no routing).
      Full (>=1024):  Everything: TTT + SLR (spectral routing) + landmarks + LightBus.
    """
    def __init__(self, d_model, n_attn_heads, n_cheby_heads=8,
                 window_size=512, max_degree=4, chunk_size=256,
                 max_landmarks=64, archive_interval=131072,
                 use_bf16=False, layer_idx=0, light_bus=None,
                 use_lut=True, gradient_ckpt=False):
        super().__init__()
        self.d_model = d_model
        self.n_cheby_heads = n_cheby_heads
        self.head_dim = d_model // n_cheby_heads
        self.max_degree = max_degree
        self.chunk_size = chunk_size
        self.use_bf16 = use_bf16
        self.layer_idx = layer_idx
        self.light_bus = light_bus
        self.use_lut = use_lut
        self.gradient_ckpt = gradient_ckpt  # C3: granular checkpointing

        # ── Complexity predictor ──
        self.complexity_gate = GatedComplexityPredictor(d_model)

        # ── Input-dependent coefficient initialization ──
        # Instead of random Chebyshev coefficients each forward pass,
        # derive initial coefficients from input statistics.
        # This eliminates high-variance gradients (same input → same coefficients)
        # and lets the outer optimizer learn good spectral initializations.
        # Architecture: bottleneck MLP (d → d/4 → n_h*deg*hd) with zero-init output
        # → at init produces zeros, learned to produce meaningful coefficients.
        coeff_out_dim = n_cheby_heads * max_degree * self.head_dim
        self.coeff_init_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, coeff_out_dim),
        )
        # Zero-init output layer → initial coefficients are zero (safe startup)
        nn.init.zeros_(self.coeff_init_proj[-1].weight)
        nn.init.zeros_(self.coeff_init_proj[-1].bias)

        # Learnable spectral decay per degree (initialized to match spectral init scale)
        k_idx = torch.arange(max_degree, dtype=torch.float32)
        self.register_buffer(
            'coeff_spectral_scale',
            (0.1 / (k_idx + 1.0)).view(1, 1, max_degree, 1)
        )

        # ── SLR (replaces NSA: spectral-routed differential local attention) ──
        self.slr = SpectralLocalRefiner(d_model, n_attn_heads, window_size)

        # ── Landmark archive ──
        self.archive = LandmarkArchive(
            d_model, n_cheby_heads, self.head_dim,
            max_degree, max_landmarks, archive_interval
        )

        # ── Gated State Refresh every 16K tokens ──
        self.refresh_interval = 16384
        self.refresh_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_cheby_heads * self.head_dim)
        )
        self.refresh_gate = nn.Sequential(
            nn.Linear(d_model, n_cheby_heads * self.head_dim),
            nn.Sigmoid()
        )

        # ── Recall Residual ──
        self.recall_proj = nn.Linear(d_model, n_cheby_heads * max_degree * self.head_dim)
        self.recall_gate = nn.Sequential(
            nn.Linear(d_model, n_cheby_heads),
            nn.Sigmoid()
        )
        self.register_buffer('recall_cooldown', torch.tensor(0, dtype=torch.long))

        # ── AsyncLightBus integration ──
        if light_bus is not None:
            # Lightweight 64-dim projection instead of full d_model cross-attention
            self.bus_summary_proj = nn.Linear(d_model, 64)
            self.bus_inject_proj = nn.Linear(64, d_model)

        # Q4: Archive embedding cache (version-keyed by n_archived)
        self._archive_emb_cache: torch.Tensor | None = None
        self._archive_emb_cache_n: int = -1

        # ── Pre-norm ──
        self.input_norm = nn.RMSNorm(d_model)

        # ── SwiGLU output fusion (C2: separate norms for spectral vs local) ──
        self.pre_norm_cheby = nn.RMSNorm(d_model)
        self.pre_norm_slr = nn.RMSNorm(d_model)
        self.fused_gate_value = nn.Linear(2 * d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.post_norm = nn.RMSNorm(d_model)

        # ── Adaptive EMA momentum ──
        self.register_buffer('ema_base', torch.tensor(0.9))
        self.register_buffer('total_tokens_seen', torch.tensor(0, dtype=torch.long))

    def forward(self, x, cheby_state=None, return_state=False):
        B, S, D = x.shape
        device = x.device
        hd = self.head_dim
        n_h = self.n_cheby_heads
        deg = self.max_degree

        # ── Pre-norm (moved before coefficient init to enable input-dependent init) ──
        x_normed = self.input_norm(x)

        # ── Initialize Chebyshev state (V11: input-dependent init) ──
        if cheby_state is None:
            # Input-dependent coefficient initialization:
            # Derive initial coefficients from input statistics instead of random.
            # Same input → same coefficients → low-variance gradients.
            x_mean = x_normed.mean(dim=1)  # [B, D]
            coeff_raw = self.coeff_init_proj(x_mean)  # [B, n_h*deg*hd]
            coeffs = coeff_raw.view(B, n_h, deg, hd) * self.coeff_spectral_scale
            momentum = torch.zeros(B, n_h, deg, hd, device=device)
        elif isinstance(cheby_state, tuple) and len(cheby_state) == 2:
            coeffs, momentum = cheby_state
        elif isinstance(cheby_state, tuple) and len(cheby_state) == 3:
            # V9 backward compat: (coeffs, m1, m2) → use m1 as momentum
            coeffs, momentum, _m2 = cheby_state
        else:
            coeffs = cheby_state
            if coeffs.dim() == 3:
                coeffs = coeffs.unsqueeze(1)
            if coeffs.shape[2] < deg:
                coeffs = F.pad(coeffs, (0, 0, 0, deg - coeffs.shape[2]))
            momentum = torch.zeros_like(coeffs)

        # ── Complexity gate ──
        x_mean = x_normed.mean(dim=1)
        gate_global = self.complexity_gate(x_mean)
        complexity = gate_global.mean()

        # ── Adaptive EMA momentum ──
        with torch.no_grad():
            self.total_tokens_seen.add_(S)
        total = self.total_tokens_seen
        # E4: Decoupled cosine annealing schedule (smooth 0.9 → 0.7)
        total_f = total.float()
        projected_tokens = 1e8  # Total projected training tokens
        progress = (total_f / projected_tokens).clamp(0.0, 1.0)
        ema_momentum_t = 0.7 + 0.5 * (self.ema_base - 0.7) * (1.0 + torch.cos(progress * math.pi))
        ema_momentum = ema_momentum_t.item()

        # ── Diagnostic: sequence length / padding waste ──
        DIAG.record_sequence_length(S)
        DIAG.record_ema_momentum(ema_momentum)
        DIAG.step()

        # ── LENGTH ROUTING ──
        if S < FAST_PATH_THRESHOLD:
            return self._fast_path(x, x_normed, coeffs, momentum, n_h, hd, deg,
                                   ema_momentum, gate_global, return_state)
        elif S < HYBRID_PATH_THRESHOLD:
            return self._hybrid_path(x, x_normed, coeffs, momentum, n_h, hd, deg,
                                     ema_momentum, gate_global, complexity, return_state)
        else:
            return self._full_path(x, x_normed, coeffs, momentum, n_h, hd, deg,
                                   ema_momentum, gate_global, complexity, return_state)

    def _fast_path(self, x, x_normed, coeffs, momentum, n_h, hd, deg,
                   ema_momentum, gate_global, return_state):
        """
        Fast path for seq < 384: No TTT overhead.
        V11: Now includes SLR + SwiGLU for consistent behavior across all paths.
        TTT is still skipped (kernel fast path), but attention is active.
        This ensures the model can learn positional dependencies even for
        short sequences, and eliminates training/inference mismatch.
        """
        B, S, D = x.shape

        # Chebyshev-RKV without TTT (fast path in kernel handles this)
        cheby_out, coeffs, momentum = apply_cheby_rkv_v10(
            x_normed, coeffs, momentum,
            n_heads=n_h, base_lr=0.005,
            ema_momentum=ema_momentum,
            use_bf16=self.use_bf16,
            gate_global=gate_global,
            dynamic_lambda=gate_global,
            use_lut=self.use_lut,
            seq_threshold=FAST_PATH_THRESHOLD,
        )

        # SLR: process all tokens (no routing overhead for short sequences)
        slr_out = self.slr(x_normed, cheby_out=cheby_out)

        # SwiGLU fusion (same as hybrid/full — consistent architecture)
        h1 = self.pre_norm_cheby(cheby_out)
        h2 = self.pre_norm_slr(slr_out)
        combo = torch.cat([h1, h2], dim=-1)
        gate_value = self.fused_gate_value(combo)
        gate, value = gate_value.split(D, dim=-1)
        fused = self.post_norm(F.silu(gate) * value)
        out = x + self.out_proj(fused)

        new_state = (coeffs.detach(), momentum.detach())
        return (out, new_state) if return_state else out

    def _hybrid_path(self, x, x_normed, coeffs, momentum, n_h, hd, deg,
                     ema_momentum, gate_global, complexity, return_state):
        """
        Hybrid path for 384 <= seq < 1024: TTT enabled, SLR on all tokens.
        SLR processes all tokens (no routing) for short-medium sequences.
        No landmark archiving overhead.
        """
        B, S, D = x.shape
        device = x.device

        # Chebyshev-RKV with TTT (full kernel path)
        cheby_out, coeffs, momentum = apply_cheby_rkv_v10(
            x_normed, coeffs, momentum,
            n_heads=n_h, base_lr=0.005,
            ema_momentum=ema_momentum,
            use_bf16=self.use_bf16,
            gate_global=gate_global,
            dynamic_lambda=gate_global,
            use_lut=self.use_lut,
            seq_threshold=0,  # Force full path (we've already routing here)
        )

        # SLR: spectral-routed local refinement (no landmarks needed)
        slr_out = self.slr(x_normed, cheby_out=cheby_out)

        # SwiGLU fusion (C2: separate norms)
        h1 = self.pre_norm_cheby(cheby_out)
        h2 = self.pre_norm_slr(slr_out)
        combo = torch.cat([h1, h2], dim=-1)
        gate_value = self.fused_gate_value(combo)
        gate, value = gate_value.split(D, dim=-1)
        fused = self.post_norm(F.silu(gate) * value)
        out = x + self.out_proj(fused)

        new_state = (coeffs.detach(), momentum.detach())
        return (out, new_state) if return_state else out

    # ---- C3: Checkpointable heavy-compute subfunctions ----

    def _cheby_compute(self, x_normed, coeffs, momentum, n_h,
                       ema_momentum, gate_global, seq_threshold):
        """Checkpointable Chebyshev + EMA + TTT step."""
        return apply_cheby_rkv_v10(
            x_normed, coeffs, momentum,
            n_heads=n_h, base_lr=0.005,
            ema_momentum=ema_momentum,
            use_bf16=self.use_bf16,
            gate_global=gate_global,
            dynamic_lambda=gate_global,
            use_lut=self.use_lut,
            seq_threshold=seq_threshold,
        )

    def _slr_compute(self, x_normed, cheby_out):
        """Checkpointable SLR step."""
        return self.slr(x_normed, cheby_out=cheby_out)

    def _full_path(self, x, x_normed, coeffs, momentum, n_h, hd, deg,
                   ema_momentum, gate_global, complexity, return_state):
        """
        Full Lightning path for seq >= 1024: Everything enabled.
        """
        B, S, D = x.shape
        device = x.device

        # Batch sync for control flow (V9 C9 optimization)
        _sync_vals = torch.stack([
            self.total_tokens_seen.float(),
            complexity.detach(),
            self.recall_cooldown.float()
        ])
        _sv = _sync_vals.tolist()
        total_val = int(_sv[0])
        complexity_scalar = _sv[1]
        cooldown_val = int(_sv[2])

        # Path 1: Chebyshev-RKV with full TTT (C3: checkpointable)
        if self.gradient_ckpt and self.training:
            cheby_out, coeffs, momentum = torch_checkpoint(
                self._cheby_compute,
                x_normed, coeffs, momentum, n_h, ema_momentum, gate_global, 0,
                use_reentrant=False,
            )
        else:
            cheby_out, coeffs, momentum = self._cheby_compute(
                x_normed, coeffs, momentum, n_h, ema_momentum, gate_global, 0,
            )

        # Landmark archiving (kept for recall residual + bus)
        self.archive.maybe_archive(coeffs, S, complexity_score=complexity_scalar)
        landmark_embs, _ = self.archive.get_landmark_embeddings(B, device)

        # Diagnostic: head orthogonality
        DIAG.record_head_orthogonality(coeffs)

        # AsyncLightBus: publish summary
        if self.light_bus is not None and landmark_embs is not None:
            summary = self.bus_summary_proj(landmark_embs.mean(dim=1))  # [B, 64]
            self.light_bus.publish(self.layer_idx, summary)

        # Path 2: SLR — spectral-routed differential local attention
        if self.gradient_ckpt and self.training:
            slr_out = torch_checkpoint(
                self._slr_compute,
                x_normed, cheby_out,
                use_reentrant=False,
            )
        else:
            slr_out = self._slr_compute(x_normed, cheby_out)

        # AsyncLightBus: inject from lower layers
        if self.light_bus is not None and self.layer_idx > 0:
            bus_ctx = self.light_bus.gather(self.layer_idx, B, device)
            if bus_ctx is not None:
                slr_out = slr_out + self.bus_inject_proj(bus_ctx).unsqueeze(1) * 0.1

        # Gated Semantic State Refresh
        if total_val > 0 and (total_val % self.refresh_interval) < S and landmark_embs is not None:
            summary = landmark_embs.mean(dim=1)
            refresh_vec = self.refresh_proj(summary).view(B, n_h, 1, hd)
            gate = self.refresh_gate(summary).view(B, n_h, 1, hd)
            coeffs = coeffs + 0.1 * gate * refresh_vec

        # Recall Residual (uses archived_embs from LandmarkArchive)
        archived_embs = self._get_archive_embs(B, device)
        with torch.no_grad():
            self.recall_cooldown.sub_(S).clamp_(min=0)
        if (archived_embs is not None
                and cooldown_val - S <= 0
                and complexity_scalar > 0.5):
            with torch.no_grad():
                x_mean = x_normed.mean(dim=1)  # [B, D]
                # E3: L2-normalized dot product (bmm on Tensor Cores)
                x_mean_n = F.normalize(x_mean, dim=-1)  # [B, D]
                arch_n = F.normalize(archived_embs, dim=-1)  # [B, n, D]
                arch_sim = torch.bmm(
                    arch_n, x_mean_n.unsqueeze(-1)
                ).squeeze(-1)  # [B, n]
                max_sim, max_idx = arch_sim.max(dim=-1)

            # Diagnostic: recall similarity distribution
            DIAG.record_recall_similarity(max_sim, inject_count=int((max_sim > 0.15).sum().item()))

            inject_mask = (max_sim > 0.15).float().view(B, 1)
            if inject_mask.sum() > 0:
                best_lm = torch.gather(
                    archived_embs, 1,
                    max_idx.view(B, 1, 1).expand(B, 1, D)
                ).squeeze(1)
                recall_vec = self.recall_proj(best_lm).view(B, n_h, deg, hd)
                recall_gate = self.recall_gate(best_lm).view(B, n_h, 1, 1)
                coeffs = coeffs + inject_mask.view(B, 1, 1, 1) * 0.08 * recall_gate * recall_vec
                self.recall_cooldown.fill_(4096)

        # SwiGLU fusion (C2: separate pre_norms for spectral vs local)
        h1 = self.pre_norm_cheby(cheby_out)
        h2 = self.pre_norm_slr(slr_out)
        combo = torch.cat([h1, h2], dim=-1)
        gate_value = self.fused_gate_value(combo)
        gate, value = gate_value.split(D, dim=-1)
        fused = self.post_norm(F.silu(gate) * value)
        out = x + self.out_proj(fused)

        new_state = (coeffs.detach(), momentum.detach())
        return (out, new_state) if return_state else out

    def _get_archive_embs(self, B, device):
        """Q4: Cached archive embeddings — recomputes MLP only when archive changes."""
        n = self.archive.n_archived.item()
        if n <= 0:
            return None
        if (self._archive_emb_cache is not None
                and self._archive_emb_cache_n == n):
            # Cache hit: O(1) — just move to device and expand
            return self._archive_emb_cache.to(device).unsqueeze(0).expand(B, -1, -1)
        # Cache miss: recompute MLP (only when n_archived changes)
        active = self.archive.archived_states[:n].to(device)
        flat   = active.reshape(n, -1)
        with torch.no_grad():
            embs = self.archive.state_to_embedding(flat).detach()  # [n, D]
        self._archive_emb_cache   = embs.cpu()  # store on CPU to save VRAM
        self._archive_emb_cache_n = n
        return embs.unsqueeze(0).expand(B, -1, -1)


def build_ortho_stack(d_model, n_attn_heads, n_cheby_heads=8,
                      n_layers=2, max_degree=4, use_lut=True,
                      gradient_ckpt=False, **kwargs):
    """
    Build a stack of V10 OrthoSSM layers with AsyncLightBus.
    C3: gradient_ckpt enables granular per-op checkpointing in each layer.
    """
    bus = AsyncLightBus(summary_dim=64, n_layers=n_layers)
    layers = nn.ModuleList()
    for i in range(n_layers):
        layer = SpectralDualPathContextEngine(
            d_model, n_attn_heads, n_cheby_heads,
            max_degree=max_degree, layer_idx=i, light_bus=bus,
            use_lut=use_lut, gradient_ckpt=gradient_ckpt,
            **kwargs,
        )
        layers.append(layer)
    return layers, bus
