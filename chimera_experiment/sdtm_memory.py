"""
sdtm_memory.py — Surprise-Driven Dual-Timescale Memory (SDTM)
==============================================================
Memoria asociativa dinámica O(1) VRAM que aprende relaciones semánticas online
durante inferencia, impulsada por la señal de sorpresa del router CHIMERA.

Arquitectura:
  W_enc:  R^{d_model × d_mem}  — encoder (FROZEN online, trained offline)
  W_dec:  R^{d_mem × d_model}  — decoder (FROZEN online, trained offline)
  M_fast: R^{d_mem × d_mem}    — working memory (online-updated via Lion+Kahan)
  M_slow: R^{d_mem × d_mem}    — long-term memory (consolidated from M_fast)

Ciclo de vida:
  READ:   z = GeLU(x @ W_enc);  r = α * z@M_fast + (1-α) * z@M_slow;  out = r @ W_dec
  WRITE:  grad_M = z_q^T @ (z_q @ M_fast - z_v)  [forma cerrada, sin autograd]
          Lion+Kahan update in-place sobre M_fast flatten
  CONSOLIDATE:  SVD(M_fast) → top-r componentes → blend into M_slow → shrink M_fast
  FORGET: usage-weighted decay per-dimension (EMA de activaciones × norma de filas)

Propiedades:
  - O(1) VRAM: M_fast + M_slow tienen tamaño fijo independiente de seq_len
  - CUDA Graph safe: read y write son matmuls puros, sin autograd
  - Reutiliza Lion+Kahan kernel existente (ttt_kernel.py)
  - Integrado con señal de sorpresa (ttt_importance / compute_token_errors_triton)

Referencia: TITANS (ICLR 2025), Fast Weight Programmers (Schlag et al. 2021)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SDTMMemory(nn.Module):
    """
    Surprise-Driven Dual-Timescale Memory.

    Memoria asociativa no-lineal con dos escalas temporales:
      M_fast: working memory, actualizada online vía Lion+Kahan
      M_slow: long-term memory, consolidada periódicamente desde M_fast via SVD

    Args:
        d_model:  dimensión del modelo (entrada/salida)
        d_mem:    dimensión del espacio de memoria POR CABEZA (default: max(64, d_model//4))
        n_heads:  número de cabezas de memoria independientes (default: 1)
                  Cada cabeza se especializa en patrones distintos (funciones, variables, etc.)
                  Multi-head: capacidad = n_heads × d_mem² asociaciones.
        sdtm_lr:  learning rate base para Lion updates
        sdtm_beta: momentum factor para Lion EMA
        consolidation_interval: cada cuántos tokens consolidar M_fast → M_slow
        consolidation_rate:     fracción de shrink de M_fast tras consolidar
        consolidation_rank_frac: fracción de d_mem para top-r SVD
        max_constraint_frac:    γ para constraint: max_δ = γ * ||M||_F / d_mem
        usage_decay_base:       λ_base para usage-weighted decay
        surprise_top_k:         cuántos tokens sorprendentes por chunk para write
    """

    def __init__(
        self,
        d_model: int,
        d_mem: int = None,
        n_heads: int = 1,
        sdtm_lr: float = 5e-4,
        sdtm_beta: float = 0.9,
        consolidation_interval: int = 2048,
        consolidation_rate: float = 0.3,
        consolidation_rank_frac: float = 0.25,
        max_constraint_frac: float = 0.1,
        usage_decay_base: float = 0.01,
        surprise_top_k: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem or max(64, d_model // 4)
        self.n_heads = n_heads
        self.d_mem_total = self.d_mem * n_heads  # total memory dimension across all heads
        self.sdtm_lr = sdtm_lr
        self.sdtm_beta = sdtm_beta
        self.consolidation_interval = consolidation_interval
        self.consolidation_rate = consolidation_rate
        self.consolidation_rank_frac = consolidation_rank_frac
        self.max_constraint_frac = max_constraint_frac
        self.usage_decay_base = usage_decay_base
        self.surprise_top_k = surprise_top_k

        # ── Encoder / Decoder (trained offline, FROZEN during online updates) ──
        # Encoder projects to n_heads × d_mem; multi-head split happens in read/write
        self.W_enc = nn.Linear(d_model, self.d_mem_total, bias=False)
        self.W_dec = nn.Linear(self.d_mem_total, d_model, bias=False)
        nn.init.xavier_uniform_(self.W_enc.weight)
        nn.init.xavier_uniform_(self.W_dec.weight)

        # ── Blend gate: per-head fast/slow mixing ──────────────────────────────
        # Each head can independently blend between fast and slow memory
        self.gate_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 0.0)  # sigmoid(0)=0.5 → equal mix

        # ── Output injection scale (like archive inject_gate) ──────────────────
        # Init -4 → sigmoid(-4)≈0.018 → minimal initial contribution
        self.inject_scale = nn.Parameter(torch.tensor(-4.0))

        # ── Online state: memory matrices per head ─────────────────────────────
        # Shape: [n_heads, d_mem, d_mem] — batched matmul via einsum
        self.register_buffer('M_fast', torch.zeros(n_heads, self.d_mem, self.d_mem))
        self.register_buffer('M_slow', torch.zeros(n_heads, self.d_mem, self.d_mem))

        # ── Lion optimizer state for M_fast (flattened across all heads) ───────
        n_elem = n_heads * self.d_mem * self.d_mem
        self.register_buffer(
            'mom_M_fast',
            torch.zeros(n_elem, dtype=torch.float32)
        )
        self.register_buffer(
            'kahan_M_fast',
            torch.zeros(n_elem, dtype=torch.float32)
        )
        self.register_buffer(
            'kahan_mom_M_fast',
            torch.zeros(n_elem, dtype=torch.float32)
        )

        # ── Usage tracking per head×dimension ──────────────────────────────────
        self.register_buffer('usage_ema', torch.zeros(n_heads, self.d_mem))

        # ── Token counter for consolidation scheduling ─────────────────────────
        self.register_buffer('_token_counter', torch.tensor(0, dtype=torch.long))

        # ── Pending write gradient (applied post-forward, like TTT-Lite) ───────
        self._pending_sdtm_grad: torch.Tensor | None = None
        self._pending_lr_mod: float = 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # READ: associative retrieval through memory matrices
    # ─────────────────────────────────────────────────────────────────────────

    def read(self, x: torch.Tensor) -> torch.Tensor:
        """
        Non-linear associative read from multi-head dual-timescale memory.

        Args:
            x: [B, S, d_model] — input representations

        Returns:
            mem_out: [B, S, d_model] — memory contribution (to be added to residual)

        Cost: O(S × d_model × d_mem_total + S × H × d_mem²)
        """
        B, S = x.shape[0], x.shape[1]

        # Encode with non-linearity → exponential capacity vs linear rank
        z = F.gelu(self.W_enc(x))                                # [B, S, H*d_mem]
        z = z.view(B, S, self.n_heads, self.d_mem)                # [B, S, H, d_mem]

        # Per-head batched retrieval from both timescales
        r_fast = torch.einsum('bshd,hde->bshe', z, self.M_fast)  # [B, S, H, d_mem]
        r_slow = torch.einsum('bshd,hde->bshe', z, self.M_slow)  # [B, S, H, d_mem]

        # Per-head context-dependent blend (learned gate)
        alpha = torch.sigmoid(self.gate_proj(x))                  # [B, S, H]
        alpha = alpha.unsqueeze(-1)                                # [B, S, H, 1]
        r = alpha * r_fast + (1.0 - alpha) * r_slow               # [B, S, H, d_mem]

        # Flatten heads and decode back to model space
        r = r.reshape(B, S, self.d_mem_total)                     # [B, S, H*d_mem]
        mem_out = self.W_dec(r)                                   # [B, S, d_model]

        # Scale output (starts near-zero, model learns to amplify)
        scale = torch.sigmoid(self.inject_scale)
        return scale * mem_out

    # ─────────────────────────────────────────────────────────────────────────
    # WRITE: compute closed-form gradient and queue for post-forward application
    # ─────────────────────────────────────────────────────────────────────────

    def compute_write(
        self,
        x: torch.Tensor,
        per_token_err: torch.Tensor,
        prob_full_mean = 1.0,
    ):
        """
        Compute per-head write gradient in closed form (no autograd needed).

        Selects top-k surprised tokens, computes auto-associative gradient per head:
            ∇M_h = (1/k) * Z_q_h^T @ (Z_q_h @ M_fast_h - Z_v_h)

        Stores gradient [H, d_mem, d_mem] in self._pending_sdtm_grad.

        Args:
            x: [B, S, d_model] — input representations
            per_token_err: [B, S-1] — per-token prediction error (surprise signal)
            prob_full_mean: float — mean prob_full from router (modulates lr)
        """
        B, S, D = x.shape
        if S < 2 or per_token_err is None:
            return

        with torch.no_grad():
            # Select top-k surprised tokens across the batch
            k = min(self.surprise_top_k, per_token_err.shape[1])
            err_mean = per_token_err.mean(dim=0)  # [S-1]
            _, top_indices = torch.topk(err_mean, k)  # [k]

            # Query tokens (current) and value tokens (next)
            x_q = x[:, :-1][:, top_indices]    # [B, k, D]
            x_v = x[:, 1:][:, top_indices]     # [B, k, D]

            # Encode both through frozen W_enc + GeLU, split into heads
            z_q = F.gelu(self.W_enc(x_q)).view(B, k, self.n_heads, self.d_mem)  # [B, k, H, d_mem]
            z_v = F.gelu(self.W_enc(x_v)).view(B, k, self.n_heads, self.d_mem)  # [B, k, H, d_mem]

            # Per-head closed-form gradient via einsum
            residual = torch.einsum('bkhd,hde->bkhe', z_q, self.M_fast) - z_v  # [B, k, H, d_mem]

            # Mean across batch
            z_q_mean = z_q.mean(dim=0)          # [k, H, d_mem]
            res_mean = residual.mean(dim=0)     # [k, H, d_mem]

            # Per-head gradient: [H, d_mem, d_mem]
            grad_M = torch.einsum('khd,khe->hde', z_q_mean, res_mean) / k

            self._pending_sdtm_grad = grad_M
            # Accept both tensor and float for prob_full_mean
            self._pending_lr_mod = prob_full_mean.item() if isinstance(prob_full_mean, torch.Tensor) else prob_full_mean

    # ─────────────────────────────────────────────────────────────────────────
    # APPLY WRITE: Lion+Kahan update on M_fast (called post-forward by trainer)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_memory_inplace(self):
        """
        Apply pending write gradient to M_fast via Lion+Kahan.

        Called by the trainer AFTER loss.backward() and optimizer.step(),
        same pattern as AdvancedChimeraLayer.update_ttt_inplace().

        Uses the existing Lion+Kahan kernel from ttt_kernel.py,
        with matrix-norm-based constraint instead of |A|-based.
        """
        grad = self._pending_sdtm_grad
        if grad is None:
            return
        self._pending_sdtm_grad = None

        from ttt_kernel import lion_constrained_update_inplace

        # Flatten all heads for element-wise Lion update
        M_flat = self.M_fast.view(-1).float().clone()  # [H * d_mem * d_mem]
        grad_flat = grad.view(-1).float()

        # Constraint reference: ||M||_F / d_mem (across all heads)
        M_frobenius = self.M_fast.float().norm()
        constraint_val = max(
            self.max_constraint_frac * M_frobenius.item() / self.d_mem,
            1e-4  # floor to prevent dead memory at init (M=0)
        )
        A_abs_proxy = torch.full_like(
            M_flat,
            constraint_val / 0.1  # Lion constraint = 0.1 * A_abs → we want γ*||M||/d_mem
        )

        # Effective LR modulated by router confidence
        lr_effective = self.sdtm_lr * self._pending_lr_mod

        lion_constrained_update_inplace(
            M_flat,
            self.mom_M_fast,
            grad_flat,
            A_abs_proxy.log().clamp(min=-20),  # A_log format: log(|A|)
            beta=self.sdtm_beta,
            lr=lr_effective,
            active_prob=1.0,
            mom_comp=self.kahan_mom_M_fast,
            dt_comp=self.kahan_M_fast,
        )

        # Write back to [H, d_mem, d_mem]
        self.M_fast.copy_(M_flat.view(self.n_heads, self.d_mem, self.d_mem).to(self.M_fast.dtype))

    # ─────────────────────────────────────────────────────────────────────────
    # USAGE TRACKING & DECAY
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_usage(self, z: torch.Tensor):
        """
        Update per-head×dimension usage EMA from recent read activations.

        Args:
            z: [B, S, H*d_mem] or [B, S, H, d_mem] — encoded read activations
        """
        if z.dim() == 3:
            z = z.view(*z.shape[:-1], self.n_heads, self.d_mem)  # [B, S, H, d_mem]

        # Mean activation magnitude per head×dimension
        activation_mag = z.abs().mean(dim=(0, 1))  # [H, d_mem]

        # Row norms of M_fast per head
        row_norms = self.M_fast.float().norm(dim=2)  # [H, d_mem]

        # Usage = activation_magnitude × row_norm
        usage = activation_mag.float() * row_norms
        ema_alpha = 0.05
        self.usage_ema.mul_(1.0 - ema_alpha).add_(usage * ema_alpha)

    @torch.no_grad()
    def apply_usage_decay(self):
        """
        Apply usage-weighted decay to M_fast per head.

        Dimensions with low usage (rarely activated by real queries) decay faster.
        Dimensions with high usage (frequently used patterns) are preserved.

        decay_i = λ_base / (1 + usage_i / mean_usage)
        """
        mean_usage = self.usage_ema.mean().clamp(min=1e-8)
        decay = self.usage_decay_base / (1.0 + self.usage_ema / mean_usage)
        # decay: [H, d_mem] — applied per-row to M_fast [H, d_mem, d_mem]
        self.M_fast.mul_(1.0 - decay.unsqueeze(-1))

    # ─────────────────────────────────────────────────────────────────────────
    # CONSOLIDATION: M_fast → M_slow via SVD
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def maybe_consolidate(self, tokens_processed: int = 0):
        """
        Consolidate M_fast → M_slow per head if enough tokens processed.

        Steps per head:
          1. SVD of M_fast_h → U, Σ, V^T
          2. Top-r components (r = d_mem * rank_frac)
          3. Blend important components into M_slow_h
          4. Shrink M_fast globally by consolidation_rate

        Cost: O(H × d_mem³) — for H=4, d_mem=192: ~28M FLOPs (negligible vs forward)
        """
        self._token_counter += tokens_processed
        if self._token_counter < self.consolidation_interval:
            return

        self._token_counter.zero_()

        r = max(1, int(self.d_mem * self.consolidation_rank_frac))

        for h in range(self.n_heads):
            M_f = self.M_fast[h].float()
            frobenius = M_f.norm()
            if frobenius < 1e-6:
                continue  # Nothing to consolidate in this head

            # SVD per head
            try:
                U, S_vals, Vh = torch.linalg.svd(M_f, full_matrices=False)
            except torch._C._LinAlgError:
                continue  # SVD failed (degenerate matrix), skip this head

            # Top-r components
            M_important = U[:, :r] @ torch.diag(S_vals[:r]) @ Vh[:r, :]

            # Spectral norm clipping: cap singular values to prevent divergence
            max_sv = S_vals[0].item()
            sv_cap = frobenius / math.sqrt(r)  # balanced energy per component
            if max_sv > sv_cap * 3.0:
                clip_factor = (sv_cap * 3.0) / max_sv
                M_important = M_important * clip_factor

            # Blend into M_slow for this head
            eta_consol = 0.1  # conservative blend rate
            self.M_slow[h].add_(
                (eta_consol * (M_important - self.M_slow[h] * eta_consol)).to(self.M_slow.dtype)
            )

        # Shrink M_fast globally (consolidated patterns now live in M_slow)
        self.M_fast.mul_(1.0 - self.consolidation_rate)

    @torch.no_grad()
    def absorb_landmarks(self, landmark_embeddings: torch.Tensor):
        """
        Absorb landmark embeddings into M_slow per head for symbiotic
        interaction with NativeLandmarkArchive.

        Each head independently absorbs the landmark patterns, allowing
        different heads to extract different aspects of the landmark.

        Args:
            landmark_embeddings: [n, d_model] — landmark vectors from archive
        """
        if landmark_embeddings.shape[0] == 0:
            return

        z = F.gelu(self.W_enc(landmark_embeddings.to(self.W_enc.weight.device)))  # [n, H*d_mem]
        z = z.view(z.shape[0], self.n_heads, self.d_mem)  # [n, H, d_mem]

        eta = 0.05  # conservative for landmark absorption
        for h in range(self.n_heads):
            z_h = z[:, h, :]  # [n, d_mem]
            M_s = self.M_slow[h].to(z_h.dtype)
            residual = z_h - z_h @ M_s
            grad = z_h.T @ residual / z_h.shape[0]
            self.M_slow[h].add_((eta * grad).to(self.M_slow.dtype))

    # ─────────────────────────────────────────────────────────────────────────
    # FULL UPDATE CYCLE (convenience for trainer)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def post_forward_update(self, seq_len: int):
        """
        Complete post-forward update cycle.
        Called by trainer after loss.backward() + optimizer.step().

        Steps:
          1. Apply pending write (Lion+Kahan on M_fast)
          2. Apply usage decay
          3. Maybe consolidate
        """
        self.update_memory_inplace()
        self.apply_usage_decay()
        self.maybe_consolidate(tokens_processed=seq_len)

    # ─────────────────────────────────────────────────────────────────────────
    # STATE MANAGEMENT (for PagedStateManager)
    # ─────────────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Extract all mutable state to CPU dict for pause/resume."""
        return {
            'M_fast': self.M_fast.cpu().clone(),
            'M_slow': self.M_slow.cpu().clone(),
            'mom_M_fast': self.mom_M_fast.cpu().clone(),
            'kahan_M_fast': self.kahan_M_fast.cpu().clone(),
            'kahan_mom_M_fast': self.kahan_mom_M_fast.cpu().clone(),
            'usage_ema': self.usage_ema.cpu().clone(),
            '_token_counter': self._token_counter.item(),
        }

    def set_state(self, state: dict, device: torch.device):
        """Restore mutable state from CPU dict."""
        self.M_fast.copy_(state['M_fast'].to(device))
        self.M_slow.copy_(state['M_slow'].to(device))
        self.mom_M_fast.copy_(state['mom_M_fast'].to(device))
        self.kahan_M_fast.copy_(state['kahan_M_fast'].to(device))
        self.kahan_mom_M_fast.copy_(state['kahan_mom_M_fast'].to(device))
        self.usage_ema.copy_(state['usage_ema'].to(device))
        self._token_counter.fill_(state['_token_counter'])

    def reset_online_state(self):
        """Reset all online state (for new session or test)."""
        self.M_fast.zero_()
        self.M_slow.zero_()
        self.mom_M_fast.zero_()
        self.kahan_M_fast.zero_()
        self.kahan_mom_M_fast.zero_()
        self.usage_ema.zero_()
        self._token_counter.zero_()
        self._pending_sdtm_grad = None
        self._pending_lr_mod = 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # DIAGNOSTICS
    # ─────────────────────────────────────────────────────────────────────────

    def memory_stats(self) -> dict:
        """Return diagnostic statistics about memory state."""
        with torch.no_grad():
            M_f = self.M_fast.float()
            M_s = self.M_slow.float()
            # Per-head max singular value (take the max across heads)
            max_sv_fast = 0.0
            max_sv_slow = 0.0
            for h in range(self.n_heads):
                if M_f[h].norm() > 1e-8:
                    max_sv_fast = max(max_sv_fast, torch.linalg.svdvals(M_f[h])[0].item())
                if M_s[h].norm() > 1e-8:
                    max_sv_slow = max(max_sv_slow, torch.linalg.svdvals(M_s[h])[0].item())
            return {
                'M_fast_frobenius': M_f.norm().item(),
                'M_slow_frobenius': M_s.norm().item(),
                'M_fast_max_sv': max_sv_fast,
                'M_slow_max_sv': max_sv_slow,
                'n_heads': self.n_heads,
                'usage_ema_mean': self.usage_ema.mean().item(),
                'usage_ema_std': self.usage_ema.std().item(),
                'tokens_since_consolidation': self._token_counter.item(),
                'inject_scale': torch.sigmoid(self.inject_scale).item(),
                'd_mem': self.d_mem,
                'd_mem_total': self.d_mem_total,
                'vram_bytes': (
                    self.M_fast.nelement() * self.M_fast.element_size() +
                    self.M_slow.nelement() * self.M_slow.element_size() +
                    self.mom_M_fast.nelement() * 4 +
                    self.kahan_M_fast.nelement() * 4 +
                    self.kahan_mom_M_fast.nelement() * 4 +
                    self.usage_ema.nelement() * 4
                ),
            }

    def extra_repr(self) -> str:
        vram_kb = self.memory_stats()['vram_bytes'] / 1024
        return (f"d_model={self.d_model}, d_mem={self.d_mem}, n_heads={self.n_heads}, "
                f"d_mem_total={self.d_mem_total}, "
                f"lr={self.sdtm_lr}, beta={self.sdtm_beta}, "
                f"consol_interval={self.consolidation_interval}, "
                f"vram_state={vram_kb:.1f}KB")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time

    print("=" * 60)
    print("SDTM Memory — Multi-Head Smoke Test")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d_model = 256
    d_mem = 64
    n_heads = 4
    B, S = 2, 512

    # 1. Instantiate (multi-head)
    mem = SDTMMemory(d_model=d_model, d_mem=d_mem, n_heads=n_heads, surprise_top_k=8).to(device)
    print(f"\n[1] Created: {mem}")
    print(f"    Parameters: {sum(p.numel() for p in mem.parameters()):,}")
    print(f"    Buffers: {sum(b.numel() for b in mem.buffers()):,}")
    stats = mem.memory_stats()
    print(f"    VRAM state: {stats['vram_bytes']/1024:.1f} KB")
    print(f"    n_heads={n_heads}, d_mem={d_mem}, d_mem_total={mem.d_mem_total}")
    print(f"    M_fast shape: {mem.M_fast.shape}")
    assert mem.M_fast.shape == (n_heads, d_mem, d_mem), f"Expected ({n_heads},{d_mem},{d_mem})"

    # 2. Read (empty memory → near-zero output)
    x = torch.randn(B, S, d_model, device=device)
    t0 = time.perf_counter()
    out = mem.read(x)
    t1 = time.perf_counter()
    print(f"\n[2] Read: out.shape={out.shape}, out.abs.mean={out.abs().mean():.6f}")
    print(f"    Time: {(t1-t0)*1000:.2f}ms")

    # 3. Write (simulate surprise)
    per_token_err = torch.randn(B, S-1, device=device).abs()
    mem.compute_write(x, per_token_err, prob_full_mean=0.7)
    assert mem._pending_sdtm_grad is not None, "Gradient should be pending"
    print(f"\n[3] Write computed: grad shape={mem._pending_sdtm_grad.shape}, grad_norm={mem._pending_sdtm_grad.norm():.6f}")
    assert mem._pending_sdtm_grad.shape == (n_heads, d_mem, d_mem), "Grad should be per-head"

    # 4. Apply write
    mem.update_memory_inplace()
    assert mem._pending_sdtm_grad is None, "Gradient should be consumed"
    stats2 = mem.memory_stats()
    print(f"\n[4] Write applied: M_fast_frobenius={stats2['M_fast_frobenius']:.6f}")

    # 5. Read again (non-zero now)
    out2 = mem.read(x)
    print(f"\n[5] Read after write: out.abs.mean={out2.abs().mean():.6f}")
    assert out2.abs().mean() > out.abs().mean(), "Output should increase after write"

    # 6. Usage tracking
    z = F.gelu(mem.W_enc(x))
    mem.update_usage(z)
    print(f"\n[6] Usage EMA: mean={mem.usage_ema.mean():.6f}, std={mem.usage_ema.std():.6f}")

    # 7. Usage decay
    M_before = mem.M_fast.clone()
    mem.apply_usage_decay()
    decay_delta = (M_before - mem.M_fast).abs().mean()
    print(f"\n[7] Usage decay: mean_delta={decay_delta:.6f}")

    # 8. Multiple write cycles → consolidation
    print(f"\n[8] Running {mem.consolidation_interval} tokens of writes...")
    n_writes = 0
    for i in range(0, mem.consolidation_interval + 256, 256):
        x_i = torch.randn(B, 256, d_model, device=device)
        err_i = torch.randn(B, 255, device=device).abs()
        mem.compute_write(x_i, err_i, prob_full_mean=0.5)
        mem.post_forward_update(seq_len=256)
        n_writes += 1

    stats3 = mem.memory_stats()
    print(f"    After {n_writes} writes:")
    print(f"    M_fast_frobenius={stats3['M_fast_frobenius']:.6f}")
    print(f"    M_slow_frobenius={stats3['M_slow_frobenius']:.6f}")
    print(f"    M_slow should be non-zero (consolidated): {stats3['M_slow_frobenius'] > 0}")

    # 9. Landmark absorption
    landmarks = torch.randn(4, d_model, device=device)
    M_slow_before = mem.M_slow.clone()
    mem.absorb_landmarks(landmarks)
    lm_delta = (mem.M_slow - M_slow_before).abs().mean()
    print(f"\n[9] Landmark absorption: delta={lm_delta:.6f}")

    # 10. State save/restore
    state = mem.get_state()
    mem.reset_online_state()
    assert mem.M_fast.abs().sum() == 0, "Should be zero after reset"
    mem.set_state(state, device=torch.device(device))
    assert mem.M_fast.abs().sum() > 0, "Should be restored"
    print(f"\n[10] State save/restore: OK")

    # 11. Backward compatibility (ensure gradients flow through read)
    mem.train()
    x_grad = torch.randn(B, S, d_model, device=device, requires_grad=True)
    out_grad = mem.read(x_grad)
    loss = out_grad.sum()
    loss.backward()
    assert x_grad.grad is not None, "Gradients should flow through read"
    print(f"\n[11] Gradient flow: x.grad.norm={x_grad.grad.norm():.6f}")

    # 12. Single-head backward compatibility
    mem_single = SDTMMemory(d_model=d_model, d_mem=d_mem, n_heads=1, surprise_top_k=8).to(device)
    out_single = mem_single.read(torch.randn(1, 64, d_model, device=device))
    assert out_single.shape == (1, 64, d_model), f"Single-head output shape wrong: {out_single.shape}"
    assert mem_single.M_fast.shape == (1, d_mem, d_mem), f"Single-head M_fast shape wrong"
    print(f"\n[12] Single-head backward compat: OK (M_fast={mem_single.M_fast.shape})")

    print("\n" + "=" * 60)
    print("ALL SDTM MULTI-HEAD SMOKE TESTS PASSED")
    print("=" * 60)
