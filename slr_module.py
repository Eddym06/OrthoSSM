"""
OrthoSSM V11 — Spectral Local Refiner (SLR)
=============================================
Replaces NSA with spectral-routed differential local attention.

Architecture:
  1. SGR (Spectral Gated Routing): TTT error → top-r% token selection
  2. Position-aware causal windowed attention on selected tokens
  3. Differential attention subtract (λ-learnable per-head)
  4. Scatter-back sparse output

Key advantages over NSA:
  - ~64% parameter reduction (328K vs 918K for D=256)
  - ~87.5% compute reduction via MoD routing (only r=12.5% tokens in attention)
  - Zero redundancy (no landmark/archive cross-attention paths)
  - Spectral routing: global path INFORMS local path selection
  - Differential subtract: cancels attention noise via learned per-head λ

Design decisions and error mitigations:
  ---------------------------------------------------------------
  ROUTING SIGNAL: Uses TTT prediction error e_t = x̂[t+1] - s[t]
    (not Chebyshev output norm). This directly measures where the
    global path failed — the mathematically correct routing signal.

  POSITIVE DEFINITENESS: Not needed here. Unlike linearized attention
    (which requires PD kernels for valid normalization), we use standard
    softmax attention on the selected sparse set. No ReLU+ε hack needed.

  GRADIENT FLOW THROUGH ROUTING: torch.topk is non-differentiable
    w.r.t. the selected SET. Routing scores are computed under
    torch.no_grad() to prevent gradient leakage to the Chebyshev path.
    Gradients flow through the attention values at selected positions.

  CAUSAL ORDERING: After topk, indices are sorted to preserve causal
    order. The attention mask enforces both causality AND window bounds
    in original sequence positions.

  MEMORY SAFETY: max_select caps the attention matrix size at 4096×4096
    per head (~256MB total), safe for 6GB GPUs. For S>32K with r=0.125,
    the effective ratio drops below 12.5% to stay within the cap.

  DIFFERENTIAL ATTENTION: λ initialized at sigmoid(0)=0.5 (moderate
    noise cancellation). The model learns to increase λ (more cancel)
    or decrease it (more standard attention) per head independently.

  SHORT SEQUENCE FALLBACK: When S ≤ 2*min_select (≤32), all tokens
    are processed without routing. Uses is_causal=True for FlashAttention.

  ROPE AT SPARSE POSITIONS: RoPE frequencies computed at original
    sequence positions (not compressed indices). This preserves correct
    relative position encoding for windowed causal attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ── Triton (opcional — fallback PyTorch si no disponible) ──────────────────
try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except ImportError:
    _TRITON_OK = False


# ============================================================================
# TRITON KERNEL: Differential Flash Attention con máscara posicional
# ============================================================================
#
# Reemplaza _chunked_sparse_windowed_attn (loop Python, múltiples kernels)
# con un único kernel N-body: O(K × W) por cabeza donde W = window_size.
#
# Estrategia (Flash-Attention 2 style):
#   Grid : (B*nH, ceil(K / BLOCK_Q))
#   Loop : sobre tiles KV causales (kv_start ≤ q_end)
#   Fusión: Q1 y Q2 comparten el mismo tile KV → 1 carga de K/V por tile
#   Fusión: lambda diferencial se aplica en registro, sin tensor intermedio
#   Salida: out = out1 - λ*out2 directamente en Out_ptr
#
if _TRITON_OK:
    @triton.jit
    def _slr_diff_flash_attn_kernel(
        Q1_ptr, Q2_ptr, K_ptr, V_ptr,    # [B*nH, K_seq, HEAD_DIM]
        Pos_ptr,                           # [B, K_seq]  — posiciones originales
        DiffLambda_ptr,                    # [nH]         — logits lambda
        Out_ptr,                           # [B*nH, K_seq, HEAD_DIM]
        # Strides
        stride_bnh_q, stride_k_q, stride_d_q,
        stride_bnh_v, stride_k_v, stride_d_v,
        stride_out_bnh, stride_out_k, stride_out_d,
        stride_pos_b, stride_pos_k,
        # Dimensiones
        K_seq, nH, window_size, scale,
        # Constantes JIT
        BLOCK_Q:   tl.constexpr,
        BLOCK_KV:  tl.constexpr,
        HEAD_DIM:  tl.constexpr,
    ):
        """
        Flash-Attention diferencial con máscara de ventana posicional.

        Cada bloque de programa atiende BLOCK_Q queries contra todos los
        tiles KV causales (kv_start ≤ q_start + BLOCK_Q - 1) y además
        dentro de la ventana posicional (pos_q - pos_k < window_size).

        La λ diferencial es aprendida por cabeza y se aplica inline:
            result = out1 - λ * out2
        eliminando el tensor intermedio `out2` en memoria global.
        """
        # ── IDs de programa ─────────────────────────────────────────────
        pid_bnh = tl.program_id(0)   # índice plano (batch × n_heads)
        pid_q   = tl.program_id(1)   # tile de queries

        bid = pid_bnh // nH           # índice de batch
        hid = pid_bnh % nH            # índice de cabeza

        q_start = pid_q * BLOCK_Q
        q_offs  = q_start + tl.arange(0, BLOCK_Q)   # [BLOCK_Q]
        d_offs  = tl.arange(0, HEAD_DIM)             # [HEAD_DIM]
        q_mask  = q_offs < K_seq                      # [BLOCK_Q]

        # ── Punteros base para (batch, head) ────────────────────────────
        qk_base  = pid_bnh * stride_bnh_q
        v_base   = pid_bnh * stride_bnh_v
        out_base = pid_bnh * stride_out_bnh
        pos_base = bid * stride_pos_b

        # ── Cargar Q1, Q2 [BLOCK_Q, HEAD_DIM] ───────────────────────────
        q1 = tl.load(
            Q1_ptr + qk_base
                   + q_offs[:, None] * stride_k_q
                   + d_offs[None, :] * stride_d_q,
            mask=q_mask[:, None], other=0.0
        )
        q2 = tl.load(
            Q2_ptr + qk_base
                   + q_offs[:, None] * stride_k_q
                   + d_offs[None, :] * stride_d_q,
            mask=q_mask[:, None], other=0.0
        )

        # ── Posiciones de las queries [BLOCK_Q] ─────────────────────────
        pos_q = tl.load(
            Pos_ptr + pos_base + q_offs * stride_pos_k,
            mask=q_mask, other=1 << 30    # posición "infinita" si OOB
        ).to(tl.int32)

        # ── Lambda diferencial por cabeza (escalar) ──────────────────────
        logit_h = tl.load(DiffLambda_ptr + hid)
        lamb = 1.0 / (1.0 + tl.exp(-logit_h.to(tl.float32)))

        # ── Estado online softmax para Q1 y Q2 ──────────────────────────
        m1  = tl.full([BLOCK_Q], float('-inf'), dtype=tl.float32)
        l1  = tl.zeros([BLOCK_Q], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

        m2  = tl.full([BLOCK_Q], float('-inf'), dtype=tl.float32)
        l2  = tl.zeros([BLOCK_Q], dtype=tl.float32)
        acc2 = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

        # ── Loop sobre tiles KV causales ─────────────────────────────────
        # Inclusión causal: solo tiles cuyo inicio ≤ fin del bloque Q.
        # (el token de query más tardío puede ver el primer token del tile)
        kv_end_tile = tl.cdiv(tl.minimum(q_start + BLOCK_Q, K_seq), BLOCK_KV)

        for kv_tile in range(0, kv_end_tile):
            kv_start = kv_tile * BLOCK_KV
            kv_offs  = kv_start + tl.arange(0, BLOCK_KV)   # [BLOCK_KV]
            kv_mask  = kv_offs < K_seq                       # [BLOCK_KV]

            # Cargar K [BLOCK_KV, HEAD_DIM]
            k = tl.load(
                K_ptr + qk_base
                      + kv_offs[:, None] * stride_k_q
                      + d_offs[None, :] * stride_d_q,
                mask=kv_mask[:, None], other=0.0
            )
            # Cargar V [BLOCK_KV, HEAD_DIM]
            v = tl.load(
                V_ptr + v_base
                      + kv_offs[:, None] * stride_k_v
                      + d_offs[None, :] * stride_d_v,
                mask=kv_mask[:, None], other=0.0
            )

            # Posiciones KV [BLOCK_KV]
            pos_k = tl.load(
                Pos_ptr + pos_base + kv_offs * stride_pos_k,
                mask=kv_mask, other=1 << 30
            ).to(tl.int32)

            # ── Máscara: causal Y ventana posicional ─────────────────────
            # causal:  pos_k[j] <= pos_q[i]   para todo (i, j)
            # ventana: pos_q[i] - pos_k[j] < window_size
            causal_ok = pos_k[None, :] <= pos_q[:, None]           # [BLOCK_Q, BLOCK_KV]
            window_ok = (pos_q[:, None] - pos_k[None, :]) < window_size
            valid     = causal_ok & window_ok & kv_mask[None, :] & q_mask[:, None]

            # ── Logits Q1 @ K.T [BLOCK_Q, BLOCK_KV] ─────────────────────
            logits1 = tl.dot(q1, tl.trans(k)) * scale              # [BLOCK_Q, BLOCK_KV]
            logits1 = tl.where(valid, logits1.to(tl.float32), -1e9)

            # ── Online softmax update Q1 ─────────────────────────────────
            m1_new  = tl.maximum(m1, tl.max(logits1, axis=1))      # [BLOCK_Q]
            alpha1  = tl.exp(m1 - m1_new)                          # rescale viejo
            p1      = tl.exp(logits1 - m1_new[:, None])            # [BLOCK_Q, BLOCK_KV]
            l1      = l1 * alpha1 + tl.sum(p1, axis=1)
            acc1    = acc1 * alpha1[:, None] + tl.dot(p1.to(v.dtype), v)
            m1      = m1_new

            # ── Logits Q2 @ K.T (reutiliza k cargado) ───────────────────
            logits2 = tl.dot(q2, tl.trans(k)) * scale
            logits2 = tl.where(valid, logits2.to(tl.float32), -1e9)

            # ── Online softmax update Q2 ─────────────────────────────────
            m2_new  = tl.maximum(m2, tl.max(logits2, axis=1))
            alpha2  = tl.exp(m2 - m2_new)
            p2      = tl.exp(logits2 - m2_new[:, None])
            l2      = l2 * alpha2 + tl.sum(p2, axis=1)
            acc2    = acc2 * alpha2[:, None] + tl.dot(p2.to(v.dtype), v)
            m2      = m2_new

        # ── Normalizar y aplicar lambda diferencial en registro ─────────
        l1_safe = tl.where(l1 > 0, l1, 1.0)[:, None]
        l2_safe = tl.where(l2 > 0, l2, 1.0)[:, None]
        out1    = acc1 / l1_safe                                    # [BLOCK_Q, HEAD_DIM]
        out2    = acc2 / l2_safe

        # Fusionado: result = out1 - λ * out2  (sin tensor out2 en HBM)
        result  = out1 - lamb * out2                                # [BLOCK_Q, HEAD_DIM]

        # ── Almacenar resultado ──────────────────────────────────────────
        tl.store(
            Out_ptr + out_base
                    + q_offs[:, None] * stride_out_k
                    + d_offs[None, :] * stride_out_d,
            result.to(Out_ptr.dtype.element_ty),
            mask=q_mask[:, None]
        )


    # ── Selección automática de bloques por HEAD_DIM ─────────────────────────
    def _slr_get_block_cfg(head_dim: int):
        """Devuelve (BLOCK_Q, BLOCK_KV) óptimos según head_dim."""
        if head_dim <= 16:
            return 128, 64
        elif head_dim <= 32:
            return 64, 64
        elif head_dim <= 64:
            return 64, 32
        else:
            return 32, 32


    def slr_diff_flash_attn(
        q1: torch.Tensor,   # [B, nH, K, hD]
        q2: torch.Tensor,
        k:  torch.Tensor,
        v:  torch.Tensor,
        positions: torch.Tensor,          # [B, K]
        diff_lambda_logit: torch.Tensor,  # [nH]
        window_size: int,
    ) -> torch.Tensor:
        """
        Lanza _slr_diff_flash_attn_kernel para atención diferencial sparse.

        Retorna result [B, nH, K, hD] = softmax(Q1 K.T)V - λ*softmax(Q2 K.T)V
        donde la máscara aplica causalidad + ventana posicional.

        No materializa out2 en HBM. La resta lambda e normalización ocurren
        en registros del kernel.

        Requiere: head_dim ∈ {8, 16, 32, 64, 128}, potencia de 2.
        """
        B, nH, K_seq, hD = q1.shape
        assert q1.is_contiguous() and q2.is_contiguous()
        assert k.is_contiguous()  and v.is_contiguous()

        scale = hD ** -0.5

        # Reshape a [B*nH, K, hD] para el kernel (strides contiguos)
        q1f = q1.reshape(B * nH, K_seq, hD)
        q2f = q2.reshape(B * nH, K_seq, hD)
        kf  = k.reshape(B * nH, K_seq, hD)
        vf  = v.reshape(B * nH, K_seq, hD)

        out = torch.empty_like(q1f)

        BLOCK_Q, BLOCK_KV = _slr_get_block_cfg(hD)

        # HEAD_DIM debe ser potencia de 2 ≥ 8 para tl.dot
        # (si hD < 8, padear; si no es pot-2, subir al siguiente)
        HD_CONST = max(8, 1 << (hD - 1).bit_length())
        if HD_CONST != hD:
            # pad en la dimensión head — raro pero correcto
            pad = HD_CONST - hD
            q1f = torch.nn.functional.pad(q1f, (0, pad))
            q2f = torch.nn.functional.pad(q2f, (0, pad))
            kf  = torch.nn.functional.pad(kf,  (0, pad))
            vf  = torch.nn.functional.pad(vf,  (0, pad))
            out = torch.empty(B * nH, K_seq, HD_CONST,
                              device=q1.device, dtype=q1.dtype)

        grid = (B * nH, triton.cdiv(K_seq, BLOCK_Q))

        _slr_diff_flash_attn_kernel[grid](
            q1f, q2f, kf, vf,
            positions, diff_lambda_logit, out,
            # strides Q/K
            q1f.stride(0), q1f.stride(1), q1f.stride(2),
            # strides V
            vf.stride(0),  vf.stride(1),  vf.stride(2),
            # strides Out
            out.stride(0), out.stride(1), out.stride(2),
            # strides Pos
            positions.stride(0), positions.stride(1),
            # dims
            K_seq, nH, window_size, scale,
            BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, HEAD_DIM=HD_CONST,
        )

        if HD_CONST != hD:
            out = out[..., :hD]

        return out.reshape(B, nH, K_seq, hD)


# ROPE UTILITIES (position-aware for sparse token sets)
# ============================================================================

def _build_rope_cache(positions, head_dim, device):
    """
    Pre-compute RoPE cos/sin at specified positions.

    positions: [B, L] — arbitrary sequence positions (may be sparse)
    Returns: cos_f [B, 1, L, hD//2], sin_f [B, 1, L, hD//2]
    """
    theta = 1.0 / (10000.0 ** (
        torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    ))
    pos = positions.float().unsqueeze(-1)   # [B, L, 1]
    freqs = pos * theta.view(1, 1, -1)      # [B, L, hD//2]
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _apply_rope(x, cos_f, sin_f):
    """
    Apply pre-computed RoPE. x: [B, nH, L, hD].
    Requires even hD (standard for RoPE).
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([
        x1 * cos_f - x2 * sin_f,
        x1 * sin_f + x2 * cos_f,
    ], dim=-1).flatten(-2)


# ============================================================================
# SPECTRAL LOCAL REFINER
# ============================================================================

class SpectralLocalRefiner(nn.Module):
    """
    Spectral Local Refiner — replaces NSA with spectral-routed
    differential local attention.

    Args:
        d_model:       Model dimension
        n_heads:       Number of attention heads
        window_size:   Causal sliding window size for position mask
        select_ratio:  Fraction of tokens to select for refinement (MoD)
        min_select:    Minimum number of tokens to always select
        max_select:    Maximum tokens to select (memory cap for attention matrix)
    """

    def __init__(self, d_model, n_heads, window_size=512,
                 select_ratio=0.125, min_select=16, max_select=4096,
                 attn_chunk_size=256):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model={d_model} must be divisible by n_heads={n_heads}"
        assert d_model // n_heads % 2 == 0, \
            f"head_dim={d_model // n_heads} must be even for RoPE"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.select_ratio = select_ratio
        self.min_select = min_select
        self.max_select = max_select
        # attn_chunk_size: tamaño de bloque de queries en la atención chunked.
        # Cada bloque de Cq queries atiende a ~2*attn_chunk_size keys →
        # O(K × attn_chunk_size) en vez de O(K²). Con K=0.125N y
        # attn_chunk_size=256: O(N) con constante 32.
        self.attn_chunk_size = attn_chunk_size

        # ── Fused projection: Q1, Q2 (differential), K, V in single matmul ──
        # 4D output: [Q1 | Q2 | K | V]
        # Q1 = primary attention query
        # Q2 = differential noise-cancel query
        # K, V = shared keys and values
        self.q1q2kv = nn.Linear(d_model, 4 * d_model, bias=False)

        # ── Output projection ──
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # ── Per-head differential lambda ──
        # sigmoid(logit) ∈ (0, 1); init at 0 → λ=0.5
        # λ close to 1 → strong noise cancellation
        # λ close to 0 → standard attention (no differential)
        self.diff_lambda_logit = nn.Parameter(torch.zeros(n_heads))

        # ── Post-norm ──
        self.post_norm = nn.RMSNorm(d_model)

    # ------------------------------------------------------------------ #
    #  SGR: Spectral Gated Routing                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    @torch.no_grad()
    def compute_routing_scores(x_normed, cheby_out):
        """
        SGR: Per-token importance from Chebyshev prediction error.

        The Chebyshev path's TTT objective predicts x̂[t+1] from EMA state s[t].
        High |e_t| means the global path failed to capture this token —
        it's the token that benefits most from local attention refinement.

        Runs under no_grad to prevent gradient leakage into the Chebyshev path.
        The Chebyshev path trains purely on its spectral TTT objective.

        Args:
            x_normed:  [B, S, D] — pre-normed input
            cheby_out: [B, S, D] — Chebyshev path EMA output (s_t)

        Returns: [B, S] — per-token importance scores (higher = needs refinement)
        """
        B, S, D = x_normed.shape

        # Softsign to match TTT domain: x̂ = x / (1 + |x|)
        abs_x = x_normed.abs().clamp(min=1e-7)
        x_ss = x_normed.sign() * (1.0 - 1.0 / (1.0 + abs_x))

        if S > 1:
            # Autoregressive TTT error: x̂[t+1] - s[t]
            error = x_ss[:, 1:, :] - cheby_out[:, :-1, :]   # [B, S-1, D]
            scores = error.abs().mean(dim=-1)                 # [B, S-1]
            # Last token: always important (most recent, no next to predict)
            last_score = scores.max(dim=-1, keepdim=True).values
            scores = torch.cat([scores, last_score], dim=-1)  # [B, S]
        else:
            # Single-token fallback: score by signal magnitude
            scores = x_normed.abs().mean(dim=-1)              # [B, 1]

        return scores

    # ------------------------------------------------------------------ #
    #  Token selection                                                     #
    # ------------------------------------------------------------------ #

    def _select_tokens(self, scores):
        """
        Select top-K tokens, preserving causal order.

        K = clamp(ceil(S * select_ratio), min_select, min(S, max_select))

        P4: topk with sorted=True returns already-sorted indices in a single
        kernel launch, eliminating the extra .sort() call.

        Returns:
            indices: [B, K] — sorted positions of selected tokens
            K: int — number selected
        """
        B, S = scores.shape
        K = max(self.min_select, int(math.ceil(S * self.select_ratio)))
        K = min(K, S, self.max_select)

        # topk devuelve índices ordenados por valor de score (descendente),
        # NO por posición. Necesitamos ordenar por posición para preservar
        # el orden causal correcto en la atención ventanada.
        _, indices = torch.topk(scores, K, dim=-1, sorted=False)
        indices, _ = torch.sort(indices, dim=-1)   # [B, K] ordenado por posición ↑
        return indices, K

    # ------------------------------------------------------------------ #
    #  O(N) Chunked Sparse Windowed Attention                             #
    # ------------------------------------------------------------------ #

    def _chunked_sparse_windowed_attn(self, q1, q2, k, v, positions):
        """
        Atención diferencial sparse en O(K × attn_chunk_size) en vez de O(K²).

        Estrategia: dividir los K tokens seleccionados en bloques de
        attn_chunk_size queries. Cada bloque de queries solo atiende a
        los keys del propio bloque más el bloque anterior (2 × chunk_size
        keys máx.), filtrando además por causalidad y ventana posicional.

        Complejidad memoria: O(K × 2*chunk_size) << O(K²)
        Para K=0.125N, chunk_size=256: O(N) con constante 32.

        Args:
            q1, q2: [B, nH, K, hD] — queries primaria y diferencial
            k, v:   [B, nH, K, hD] — keys y values compartidos
            positions: [B, K] — posiciones originales de los tokens seleccionados

        Returns:
            out1, out2: [B, nH, K, hD]
        """
        B, nH, K, hD = q1.shape
        C = self.attn_chunk_size
        out1 = torch.zeros(B, nH, K, hD, device=q1.device, dtype=q1.dtype)
        out2 = torch.zeros(B, nH, K, hD, device=q1.device, dtype=q1.dtype)

        for q_start in range(0, K, C):
            q_end = min(q_start + C, K)

            # Keys: bloque previo + bloque actual (ventana deslizante causal)
            k_start = max(0, q_start - C)
            k_end   = q_end   # límite causal: no ver el futuro

            q1_c = q1[:, :, q_start:q_end, :]   # [B, nH, Cq, hD]
            q2_c = q2[:, :, q_start:q_end, :]
            k_c  = k[:, :, k_start:k_end,  :]   # [B, nH, Ck, hD]
            v_c  = v[:, :, k_start:k_end,  :]

            pos_q = positions[:, q_start:q_end].unsqueeze(-1).float()  # [B, Cq, 1]
            pos_k = positions[:, k_start:k_end].unsqueeze(-2).float()  # [B, 1,  Ck]

            # Máscara local [B, 1, Cq, Ck]: causal + ventana posicional
            causal   = pos_k <= pos_q
            in_win   = (pos_q - pos_k) < self.window_size
            local_mask = torch.where(
                causal & in_win, 0.0, float('-inf')
            ).unsqueeze(1)  # [B, 1, Cq, Ck]

            o1 = F.scaled_dot_product_attention(q1_c, k_c, v_c, attn_mask=local_mask)
            o2 = F.scaled_dot_product_attention(q2_c, k_c, v_c, attn_mask=local_mask)

            out1[:, :, q_start:q_end, :] = o1
            out2[:, :, q_start:q_end, :] = o2

        return out1, out2

    # ------------------------------------------------------------------ #
    #  _build_window_mask — OBSOLETO, conservado para compatibilidad       #
    # ------------------------------------------------------------------ #

    def _build_window_mask(self, positions, device):
        """
        OBSOLETO: materializa [B, 1, K, K] — O(K²).
        Reemplazado por _chunked_sparse_windowed_attn.
        Se conserva sólo para compatibilidad con código externo.
        """
        pos_q = positions.unsqueeze(-1).float()
        pos_k = positions.unsqueeze(-2).float()
        causal   = pos_k <= pos_q
        in_window = (pos_q - pos_k) < self.window_size
        mask = torch.where(causal & in_window, 0.0, float('-inf'))
        return mask.unsqueeze(1)

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, x, cheby_out=None, routing_scores=None):
        """
        SLR forward pass.

        Args:
            x:              [B, S, D] — pre-normed input
            cheby_out:      [B, S, D] — Chebyshev path EMA output (for SGR)
            routing_scores: [B, S]    — pre-computed SGR scores (optional)

        Returns:
            output: [B, S, D] — sparse refinement signal
                    (zeros at unselected positions, refined at selected ones)
        """
        B, S, D = x.shape
        device = x.device
        nH = self.n_heads
        hD = self.head_dim

        # ── 1. Spectral Gated Routing ──────────────────────────────────
        do_routing = (S > self.min_select * 2) and (cheby_out is not None)

        if do_routing:
            if routing_scores is None:
                routing_scores = self.compute_routing_scores(x, cheby_out)
            indices, K = self._select_tokens(routing_scores)
        else:
            # Short sequences or no cheby_out: process all tokens
            indices = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
            K = S

        # ── 2. Gather selected tokens ─────────────────────────────────
        idx_d = indices.unsqueeze(-1).expand(B, K, D)   # [B, K, D]
        x_sel = torch.gather(x, 1, idx_d)               # [B, K, D]

        # ── 3. Fused Q1Q2KV projection ────────────────────────────────
        qkv = self.q1q2kv(x_sel)                        # [B, K, 4D]
        q1, q2, k, v = qkv.chunk(4, dim=-1)

        q1 = q1.view(B, K, nH, hD).transpose(1, 2)     # [B, nH, K, hD]
        q2 = q2.view(B, K, nH, hD).transpose(1, 2)
        k  = k.view(B, K, nH, hD).transpose(1, 2)
        v  = v.view(B, K, nH, hD).transpose(1, 2)

        # ── 4. RoPE at original sequence positions ────────────────────
        cos_f, sin_f = _build_rope_cache(indices, hD, device)
        q1 = _apply_rope(q1, cos_f, sin_f)
        q2 = _apply_rope(q2, cos_f, sin_f)
        k  = _apply_rope(k,  cos_f, sin_f)

        # ── 5+6. Differential Attention — Triton fused o fallback PyTorch ──
        #
        # Jerarquía de rutas:
        #   (A) TRITON + routing activo  → _slr_diff_flash_attn_kernel
        #       Un solo kernel: Flash-Attn + máscara posicional + diff-λ.
        #       Elimina el loop Python y los tensores out1/out2 en HBM.
        #   (B) PyTorch + routing activo → _chunked_sparse_windowed_attn
        #       Fallback si Triton no está instalado.
        #   (C) Secuencia corta (sin routing) → FlashAttn causal estándar.
        if do_routing and _TRITON_OK:
            # Ruta A: Triton fusionado  [B, nH, K, hD]
            # slr_diff_flash_attn ya aplica lambda internamente → diff_out directo
            diff_out = slr_diff_flash_attn(
                q1.contiguous(), q2.contiguous(),
                k.contiguous(),  v.contiguous(),
                indices,
                self.diff_lambda_logit,
                self.window_size,
            )
        elif do_routing:
            # Ruta B: PyTorch chunked (fallback sin Triton)
            out1, out2 = self._chunked_sparse_windowed_attn(q1, q2, k, v, indices)
            lamb = torch.sigmoid(self.diff_lambda_logit).view(1, nH, 1, 1)
            diff_out = out1 - lamb * out2              # [B, nH, K, hD]
        else:
            # Ruta C: secuencia corta, todos los tokens, FlashAttn causal
            out1 = F.scaled_dot_product_attention(q1, k, v, is_causal=True)
            out2 = F.scaled_dot_product_attention(q2, k, v, is_causal=True)
            lamb = torch.sigmoid(self.diff_lambda_logit).view(1, nH, 1, 1)
            diff_out = out1 - lamb * out2              # [B, nH, K, hD]

        # ── 7. Reshape and scatter back to full sequence ──────────────
        diff_out = diff_out.transpose(1, 2).contiguous().view(B, K, D)

        output = torch.zeros(B, S, D, device=device, dtype=x.dtype)
        output.scatter_(1, idx_d, diff_out)

        # ── 8. Output projection + norm ───────────────────────────────
        output = self.o_proj(output)
        output = self.post_norm(output)

        return output

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def extra_repr(self):
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"window={self.window_size}, select_ratio={self.select_ratio}, "
            f"max_select={self.max_select}, attn_chunk={self.attn_chunk_size}"
        )
