"""
OrthoSSM V10 — NSA Module (Native Sparse Attention)
====================================================
V10 changes:
  - Dynamic window sizing: min(window_size, seq_len)
  - Short-sequence fast path: use is_causal=True when seq <= window
  - Reduced overhead for hybrid path (no landmarks needed)
  - Same three-path architecture for full path (local + landmark + archive)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _chunked_sliding_window_attn(q, k, v, window_size, scale=None):
    """
    V10: Chunked sliding window causal attention. O(S*W) memory.
    Identical to V9 C11 — already optimal.
    """
    B, nH, S, hD = q.shape
    W = window_size
    if scale is None:
        scale = hD ** -0.5

    n_chunks = (S + W - 1) // W
    pad_s = n_chunks * W - S

    k_pad = F.pad(k, (0, 0, W, pad_s))
    v_pad = F.pad(v, (0, 0, W, pad_s))

    if pad_s > 0:
        q_pad = F.pad(q, (0, 0, 0, pad_s))
    else:
        q_pad = q

    q_chunks = q_pad.view(B, nH, n_chunks, W, hD)

    kv_len = 2 * W
    k_chunks = k_pad.unfold(2, kv_len, W).permute(0, 1, 2, 4, 3)
    v_chunks = v_pad.unfold(2, kv_len, W).permute(0, 1, 2, 4, 3)

    attn = torch.matmul(q_chunks, k_chunks.transpose(-1, -2)) * scale

    qi = torch.arange(W, device=q.device).unsqueeze(1)
    ki = torch.arange(kv_len, device=q.device).unsqueeze(0)
    cw_mask = (ki <= qi) | (ki > qi + W)

    chunk_idx = torch.arange(n_chunks, device=q.device).view(n_chunks, 1)
    ki_1d = torch.arange(kv_len, device=q.device).unsqueeze(0)
    abs_k_pos = chunk_idx * W + ki_1d - W
    pad_mask = (abs_k_pos < 0) | (abs_k_pos >= S)

    full_mask = cw_mask.unsqueeze(0) | pad_mask.unsqueeze(1)

    # E6: Query-side padding mask — prevent softmax probability leak
    q_pos = torch.arange(n_chunks * W, device=q.device).view(n_chunks, W)
    q_invalid = q_pos >= S  # [n_chunks, W] True for padded queries
    full_mask = full_mask | q_invalid.unsqueeze(-1)

    attn = attn.masked_fill(full_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    attn = F.softmax(attn, dim=-1)

    # E6: Fix NaN from all-masked rows (padded query positions)
    attn = attn.nan_to_num(0.0)

    out_chunks = torch.matmul(attn, v_chunks)
    out = out_chunks.reshape(B, nH, n_chunks * W, hD)[:, :, :S, :]
    return out


def _apply_rotary_pos_emb(q, k, seq_len, head_dim, device):
    """Apply RoPE to Q and K. q, k: [B, nH, S, hD]"""
    theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, theta)
    cos_f = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin_f = freqs.sin().unsqueeze(0).unsqueeze(0)

    def rotate(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([
            x1 * cos_f - x2 * sin_f,
            x1 * sin_f + x2 * cos_f,
        ], dim=-1).flatten(-2)

    return rotate(q), rotate(k)


class NSAModule(nn.Module):
    """
    V10 Native Sparse Attention with dynamic window sizing.

    Key V10 optimization:
      - For S <= window_size: use is_causal=True directly (no chunked window)
      - Window size dynamically capped at seq_len for hybrid path
      - Landmark/archive paths only activated when data is provided
    """
    def __init__(self, d_model, n_heads, window_size=512):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        # Path 1: local causal sliding-window
        self.local_qkv = nn.Linear(d_model, 3 * d_model, bias=False)

        # Path 2: cross-attention to landmarks
        self.lm_kv = nn.Linear(d_model, 2 * d_model, bias=False)

        # Path 3: cross-attention to archived landmarks
        self.arch_kv = nn.Linear(d_model, 2 * d_model, bias=False)

        # Output projection
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # SwiGLU fusion (3 paths → gated output)
        self.fused_gate_value = nn.Linear(3 * d_model, 2 * d_model, bias=False)

        # Per-path normalization
        self.norm_local = nn.RMSNorm(d_model)
        self.norm_lm = nn.RMSNorm(d_model)
        self.norm_arch = nn.RMSNorm(d_model)

    def _reshape(self, t, B, L):
        return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x, landmarks=None, archived_landmarks=None):
        B, S, D = x.shape

        # Path 1: Local attention
        qkv = self.local_qkv(x)
        q_loc, k_loc, v_loc = qkv.split(D, dim=-1)
        Q = self._reshape(q_loc, B, S)
        K_loc = self._reshape(k_loc, B, S)
        V_loc = self._reshape(v_loc, B, S)

        Q, K_loc = _apply_rotary_pos_emb(Q, K_loc, S, self.head_dim, x.device)

        # V10: Dynamic window — always use fast path when S <= window_size
        effective_window = min(self.window_size, S)
        if S <= effective_window:
            out_loc = F.scaled_dot_product_attention(Q, K_loc, V_loc, is_causal=True)
        else:
            out_loc = _chunked_sliding_window_attn(
                Q, K_loc, V_loc, effective_window, scale=self.scale
            )

        out_loc = out_loc.transpose(1, 2).contiguous().view(B, S, D)
        out_loc = self.norm_local(out_loc)

        # Path 2: landmarks (skip computation if None)
        if landmarks is not None and landmarks.shape[1] > 0:
            L = landmarks.shape[1]
            kv_lm = self.lm_kv(landmarks)
            K_lm, V_lm = kv_lm.split(D, dim=-1)
            K_lm = self._reshape(K_lm, B, L)
            V_lm = self._reshape(V_lm, B, L)
            out_lm = F.scaled_dot_product_attention(Q, K_lm, V_lm, is_causal=False)
            out_lm = out_lm.transpose(1, 2).contiguous().view(B, S, D)
            out_lm = self.norm_lm(out_lm)
        else:
            out_lm = torch.zeros_like(out_loc)

        # Path 3: archived landmarks (skip if None)
        if archived_landmarks is not None and archived_landmarks.shape[1] > 0:
            A = archived_landmarks.shape[1]
            kv_arch = self.arch_kv(archived_landmarks)
            K_arch, V_arch = kv_arch.split(D, dim=-1)
            K_arch = self._reshape(K_arch, B, A)
            V_arch = self._reshape(V_arch, B, A)
            out_arch = F.scaled_dot_product_attention(Q, K_arch, V_arch, is_causal=False)
            out_arch = out_arch.transpose(1, 2).contiguous().view(B, S, D)
            out_arch = self.norm_arch(out_arch)
        else:
            out_arch = torch.zeros_like(out_loc)

        # SwiGLU gated fusion
        concat = torch.cat([out_loc, out_lm, out_arch], dim=-1)
        gate_value = self.fused_gate_value(concat)
        gate, value = gate_value.split(D, dim=-1)
        out = F.silu(gate) * value

        return self.o_proj(out)
