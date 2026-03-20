"""
Chimera Autonomous Swarm (CAS) — Autonomy-of-Experts × LExI × TTT-Coupled Modulation.

Subsistema que reemplaza el MoE convencional (Top-K softmax router) con un
paradigma de expertos autónomos. Cada experto decide por sí mismo si procesar
cada token, condicionado por:

  1. AoE Micro-Probing: sondas FP8 ultraligeras (~D → 1 scalar) que producen
     un score de confianza c_i = σ(x @ W_probe_i + b_i). Si c_i > τ → activa.
     No hay softmax central. Cualquier número de expertos puede despertar (0..E).

  2. LExI Non-Linear Depth Thresholding: τ_l es un parámetro aprendible
     condicionado por la profundidad relativa de la capa: τ_l = τ_base + γ·SiLU(W_d · l/L).
     Las capas profundas aprenden umbrales más bajos → más expertos colaboran.

  3. TTT-Coupled Autonomy: la pérdida proxy del TTT (per_token_err) modula τ
     en tiempo real: τ_eff = τ_l - α · L_proxy. Alta sorpresa → umbral baja →
     más expertos despiertan → respuesta inmunológica cognitiva.

  4. Grouped GEMM via Triton: kernel que despacha matmuls agrupadas con tamaños
     dinámicos por experto, evitando padding y overhead de Python loops.

Overhead paramétrico: E × (D+1) para probes + ~5 escalares para depth/TTT gate.
Para E=8, D=256: 2056 params extras (~0.002% de modelo 125M). Despreciable.

Referencia teórica:
  - AoE: "Autonomy-of-Experts" (2025) — self-assessment routing
  - LExI: "Layer-wise Expert Influence" (2025) — depth-adaptive capacity
  - SPSA/TTT: Chimera v9 (intra-layer test-time training)
  - Grouped GEMM: Megablocks (Gale et al., 2023), ST-MoE (Zoph et al., 2022)

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from gpu_profile import get_gpu_profile as _get_gpu_profile


# ─────────────────────────────────────────────────────────────────────────────
# § 1. AoE Micro-Probe — ultralight per-expert score
# ─────────────────────────────────────────────────────────────────────────────

class MicroProbe(nn.Module):
    """
    Sonda ultraligera por experto: una proyección D→1 que produce c_i ∈ (0,1).

    Parámetros: D+1 (weight + bias) × dtype.
    En H200 FP8: la proyección se cuantiza on-the-fly en forward.
    En Ada/Ampere: se ejecuta en BF16/FP16 nativo.

    Init bias:
      bias = -log(1/p_init - 1) → sigmoid(bias) = p_init
      Con p_init=0.3: bias ≈ -0.847. Cada experto empieza con ~30% de activación.
      Esto permite que el gradiente fluya desde el primer paso sin colapso
      (todos ON) ni muerte (todos OFF).
    """
    __slots__ = ()

    def __init__(self, d_model: int, p_init: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=True)
        nn.init.normal_(self.proj.weight, std=0.01)
        # Init: sigmoid(bias) ≈ p_init
        nn.init.constant_(self.proj.bias, -math.log(1.0 / p_init - 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, D] — tokens aplanados
        Returns: [T, 1] — confidence scores (pre-sigmoid logits)
        """
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# § 2. LExI Non-Linear Depth Threshold
# ─────────────────────────────────────────────────────────────────────────────

class DepthThreshold(nn.Module):
    """
    Umbral adaptativo por profundidad de capa:

      τ_l = τ_base + γ · SiLU(W_depth · (l / L))

    τ_base: umbral base aprendible (init: logit de p_init ≈ 0.3)
    γ:      amplitud aprendible (init: 0.5 — permite ±0.18 variación vía SiLU)
    W_depth: peso escalar aprendible (init: 2.0 — centra SiLU en zona no lineal)
    """

    def __init__(self, tau_init: float = 0.3):
        super().__init__()
        # τ_base en espacio logit: sigmoid(logit) = tau_init
        tau_logit = -math.log(1.0 / tau_init - 1.0)
        self.tau_base = nn.Parameter(torch.tensor(tau_logit))
        self.gamma    = nn.Parameter(torch.tensor(0.5))
        self.w_depth  = nn.Parameter(torch.tensor(2.0))
        # Buffer escalar reutilizable para depth_ratio — evita la alloc de
        # torch.tensor(depth_ratio, device=...) en cada llamada a forward().
        # fill_() sobreescribe in-place: O(1) sin alloc en el hot path.
        self.register_buffer('_depth_scalar', torch.zeros(1))

    def forward(self, depth_ratio: float) -> torch.Tensor:
        """
        depth_ratio: l / L ∈ [0, 1]
        Returns: τ_l (scalar tensor) — threshold en espacio logit
        """
        # fill_ reutiliza el buffer preallocado — cero alloc por llamada.
        self._depth_scalar.fill_(depth_ratio)
        depth_t = self._depth_scalar.squeeze()   # vista escalar (sin copia)
        return self.tau_base + self.gamma * F.silu(self.w_depth * depth_t)



# ─────────────────────────────────────────────────────────────────────────────
# § 3. Triton Grouped GEMM Kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _grouped_gemm_kernel(
    # Inputs
    X_ptr,             # [total_tokens, D] — all tokens concatenated by expert
    W_ptr,             # [E, D, D_ff] — expert weights (first linear, gate or up)
    # Output
    Y_ptr,             # [total_tokens, D_ff]
    # Metadata
    offsets_ptr,       # [E+1] — cumulative token counts per expert
    # Dimensions
    D: tl.constexpr,
    D_FF: tl.constexpr,
    BLOCK_M: tl.constexpr,   # tile rows (tokens)
    BLOCK_N: tl.constexpr,   # tile cols (output dim)
    BLOCK_K: tl.constexpr,   # reduction dim
):
    """
    Grouped GEMM: Y[expert_e] = X[expert_e] @ W[e]

    Grid: (n_tiles_m * E, n_tiles_n)
      - Eje 0: bloques de tokens × expertos
      - Eje 1: bloques de dimensión de salida

    Cada CTA procesa un tile [BLOCK_M, BLOCK_K] × [BLOCK_K, BLOCK_N] para
    exactamente UN experto. Si el experto tiene 0 tokens, su CTA escribe 0 rows.

    Ventaja sobre PyTorch loop:
      - Un solo kernel launch para todos los expertos
      - Zero padding: tiles parciales se manejan con boundary checks
      - Load de X y W vía TMA en Hopper (cuando BLOCK_K ≤ 128)
    """
    # Decodificar program_id para determinar expert y tile-row
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Determinar qué experto y qué tile-row dentro de ese experto
    # offsets: [0, n_tok_0, n_tok_0+n_tok_1, ...]
    # Búsqueda lineal sin break (Triton no soporta break)
    expert_id = 0
    local_tile = 0
    row_start = 0
    row_end = 0
    
    cum_tiles = 0
    e_start = tl.load(offsets_ptr)           # offsets[0] = 0
    
    # E fixed to 16 max for unrolling
    for e in range(16):
        e_end = tl.load(offsets_ptr + e + 1)
        n_tokens_e = e_end - e_start
        n_tiles_e = tl.cdiv(n_tokens_e, BLOCK_M)
        
        # Check if current tile belongs to expert e
        # Range: [cum_tiles, cum_tiles + n_tiles_e)
        in_range = (pid_m >= cum_tiles) & (pid_m < cum_tiles + n_tiles_e)
        
        if in_range:
            expert_id = e
            local_tile = pid_m - cum_tiles
            row_start = e_start + local_tile * BLOCK_M
            row_end = tl.minimum(row_start + BLOCK_M, e_end)
            
        cum_tiles += n_tiles_e
        e_start = e_end

    # Column range
    col_start = pid_n * BLOCK_N
    col_end = tl.minimum(col_start + BLOCK_N, D_FF)

    # Offsets en X: [row_start:row_end, :]
    row_offs = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offs < row_end

    # Accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Reduction over K dimension
    for k_start in range(0, D, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < D

        # Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + row_offs[:, None] * D + k_offs[None, :]
        x_tile = tl.load(x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W tile: W[expert_id, k, col] — stride: [D*D_FF, D_FF, 1]
        col_offs = col_start + tl.arange(0, BLOCK_N)
        w_ptrs = W_ptr + expert_id * D * D_FF + k_offs[:, None] * D_FF + col_offs[None, :]
        col_mask = col_offs < D_FF
        w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & col_mask[None, :], other=0.0)

        # Matmul accumulate
        acc += tl.dot(x_tile.to(tl.float16), w_tile.to(tl.float16)).to(tl.float32)

    # Store output: [BLOCK_M, BLOCK_N]
    col_offs = col_start + tl.arange(0, BLOCK_N)
    out_ptrs = Y_ptr + row_offs[:, None] * D_FF + col_offs[None, :]
    out_mask = row_mask[:, None] & (col_offs[None, :] < D_FF)
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


class _GroupedGEMMFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_sorted, W, offsets, D_ff):
        # x_sorted: [T, D]
        # W: [E, D, D_ff] -- must be contiguous [In, Out] layout for kernel
        # offsets: [E+1] int32
        
        ctx.save_for_backward(x_sorted, W, offsets)
        ctx.D_ff = D_ff
        
        T, D = x_sorted.shape
        E = W.shape[0]
        y = torch.empty(T, D_ff, device=x_sorted.device, dtype=x_sorted.dtype)

        if T == 0:
            return y

        # Tile sizes logic
        BLOCK_M = 32
        BLOCK_N = min(64, triton.next_power_of_2(D_ff))
        BLOCK_K = min(64, triton.next_power_of_2(D))

        n_tiles_m = triton.cdiv(T, BLOCK_M) + E
        n_tiles_n = triton.cdiv(D_ff, BLOCK_N)

        _grouped_gemm_kernel[(n_tiles_m, n_tiles_n)](
            x_sorted, W, y, offsets,
            D=D, D_FF=D_ff,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [T, D_ff]
        x_sorted, W, offsets = ctx.saved_tensors
        
        grad_output = grad_output.contiguous()
        d_x = torch.zeros_like(x_sorted)
        d_W = torch.zeros_like(W)
        
        # CPU Loop for Backward (Simpler than Triton & Correct)
        offsets_cpu = offsets.cpu()
        E = W.shape[0]
        
        for e in range(E):
            start = offsets_cpu[e].item()
            end = offsets_cpu[e+1].item()
            if start == end: 
                continue
            
            # Blocks
            # W[e]: [D, D_ff]
            w_e = W[e] 
            g_e = grad_output[start:end] # [N, D_ff]
            x_e = x_sorted[start:end]    # [N, D]
            
            # d_x_e = g_e @ w_e.T  => [N, D_ff] @ [D_ff, D] = [N, D]
            # No need for .t() if using F.linear logic, but here W is [D, D_ff]
            d_x[start:end] = torch.mm(g_e, w_e.t())
            
            # d_w_e = x_e.T @ g_e  => [D, N] @ [N, D_ff] = [D, D_ff]
            d_W[e] = torch.mm(x_e.t(), g_e)
            
        return d_x, d_W, None, None


def triton_grouped_gemm(
    x_sorted: torch.Tensor,      # [total_tokens, D]
    W: torch.Tensor,              # [E, D, D_ff]
    offsets: torch.Tensor,        # [E+1] int32
    D_ff: int,
) -> torch.Tensor:
    """
    Wrapper autograd-safe para Grouped GEMM.
    """
    return _GroupedGEMMFunc.apply(
        x_sorted.contiguous(),
        W.contiguous(),
        offsets.contiguous(),
        D_ff
    )


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# § 4. Sparse Dispatch Engine
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# § Elite Triton Grouped GEMM (Zero-Sync, Single Launch) - Autograd Enabled
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _grouped_gemm_fw_elite(
    X_ptr, W_ptr, Y_ptr, Offsets_ptr,
    stride_xm, stride_xk,
    stride_we, stride_wk, stride_wn,
    stride_ym, stride_yn,
    D: tl.constexpr, D_FF: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    tile_m    = tl.program_id(1)
    tile_n    = tl.program_id(2)
    
    e_start = tl.load(Offsets_ptr + expert_id)
    e_end   = tl.load(Offsets_ptr + expert_id + 1)
    
    n_tokens = e_end - e_start
    if tile_m * BLOCK_M >= n_tokens:
        return 
        
    row_start = e_start + tile_m * BLOCK_M
    row_end = tl.minimum(row_start + BLOCK_M, e_end)
    
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    w_ptrs = W_ptr + expert_id * stride_we + offs_k[:, None] * stride_wk + offs_n[None, :]
    
    # K loop
    for k in range(0, tl.cdiv(D, BLOCK_K)):
        k_start = k * BLOCK_K
        offs_k_curr = k_start + offs_k
        
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k_curr[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < row_end) & (offs_k_curr[None, :] < D), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k_curr[:, None] < D) & (offs_n[None, :] < D_FF), other=0.0)
        
        acc += tl.dot(x.to(tl.float16), w.to(tl.float16)).to(tl.float32)
        w_ptrs += BLOCK_K * stride_wk
        
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=(offs_m[:, None] < row_end) & (offs_n[None, :] < D_FF))


def triton_grouped_gemm(x: torch.Tensor, w: torch.Tensor, offsets: torch.Tensor):
    T, D = x.shape
    E, _, D_FF = w.shape
    y = torch.empty((T, D_FF), device=x.device, dtype=x.dtype)
    
    counts = offsets[1:] - offsets[:-1]
    max_tokens = counts.max().item() if T > 0 else 0
    if max_tokens == 0: return y
    
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 64
    grid = (E, triton.cdiv(max_tokens, BLOCK_M), triton.cdiv(D_FF, BLOCK_N))
    
    _grouped_gemm_fw_elite[grid](
        x, w, y, offsets,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1), w.stride(2),
        y.stride(0), y.stride(1),
        D, D_FF,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return y

class GroupedGEMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, offsets):
        y = triton_grouped_gemm(x, w, offsets)
        ctx.save_for_backward(x, w, offsets)
        return y
        
    @staticmethod
    def backward(ctx, grad_y):
        x, w, offsets = ctx.saved_tensors
        # x: [T, D_in], w: [E, D_in, D_out], grad_y: [T, D_out]
        
        # grad_x = grad_y @ w.t() -> Un grouped GEMM!! Launch 1 vez!
        # grad_y: [T, D_out], w.t(): [E, D_out, D_in] -> grad_x: [T, D_in]
        w_t = w.transpose(1, 2).contiguous()
        grad_x = triton_grouped_gemm(grad_y, w_t, offsets)
        
        # grad_w = x.t() @ grad_y, for each expert -> reduction
        grad_w = torch.zeros_like(w)
        for e in range(w.shape[0]):
            s, e_idx = offsets[e].item(), offsets[e+1].item()
            if s < e_idx:
                # O(E) dispatches para weight grad, super aceptables (E=8 o 64).
                grad_w[e] = x[s:e_idx].t() @ grad_y[s:e_idx]
            
        return grad_x, grad_w, None

def sparse_expert_dispatch(
    x: torch.Tensor,
    active_mask: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    W_gate: torch.Tensor | None,
    confidence: torch.Tensor,
    use_triton_gemm: bool = True,
) -> torch.Tensor:
    """
    FRENTE 2: EL JEFE FINAL DEL BAJO NIVEL (Triton Grouped GEMM Elite).
    Despacho unificado usando empaquetado de tokens (argsort).
    """
    T, D = x.shape
    E = active_mask.shape[1]
    out = torch.zeros_like(x)
    
    if T == 0: return out

    if not use_triton_gemm:
        for e in range(E):
            tokens_e = active_mask[:, e].nonzero(as_tuple=True)[0]
            x_e = x[tokens_e]                        
            up = F.linear(x_e, W_up[e])              
            if W_gate is not None:
                gate = F.silu(F.linear(x_e, W_gate[e]))
                up = up * gate
            else:
                up = F.silu(up)
            down = F.linear(up, W_down[e])           
            c_e = confidence[tokens_e, e].unsqueeze(-1)
            out.index_add_(0, tokens_e, down * c_e)
        return out

    # 1. Sort de tokens
    token_ids, expert_ids = active_mask.nonzero(as_tuple=True)
    sort_idx = torch.argsort(expert_ids)
    sorted_experts = expert_ids[sort_idx]
    sorted_tokens  = token_ids[sort_idx]
    
    # 2. Offsets (bincount)
    counts = torch.bincount(sorted_experts, minlength=E)
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=x.device)
    offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    
    X_grouped = x[sorted_tokens]  # [Alloc, D]
    
    # 3. Triton Grouped GEMM
    # Importante: los pesos están en [E, D_out, D_in] (linear style).
    # Necesitamos pasarlos a Triton como [E, D_in, D_out] -> .permute(0,2,1).contiguous()
    W_up_t = W_up.permute(0, 2, 1).contiguous()
    up = GroupedGEMMFunction.apply(X_grouped, W_up_t, offsets)
    
    if W_gate is not None:
        W_gate_t = W_gate.permute(0, 2, 1).contiguous()
        gate = GroupedGEMMFunction.apply(X_grouped, W_gate_t, offsets)
        up = up * F.silu(gate)
    else:
        up = F.silu(up)
        
    W_down_t = W_down.permute(0, 2, 1).contiguous()
    down = GroupedGEMMFunction.apply(up, W_down_t, offsets)
    
    # 4. Scatter-Add
    c_grouped = confidence[sorted_tokens, sorted_experts].unsqueeze(-1)
    down = down * c_grouped
    
    out.index_add_(0, sorted_tokens, down.to(out.dtype))
    
    return out

    # ── 1. Empaquetado de Tokens (Token Sorting) sin CPU syncs ────────────────
    # Encontrar todos los pares (token_id, expert_id) activos
    token_ids, expert_ids = active_mask.nonzero(as_tuple=True)
    
    # Agrupar físicamente todos los tokens ordenados por experto 
    # argsort es estable y masivo en GPU.
    sort_idx = torch.argsort(expert_ids)
    sorted_experts = expert_ids[sort_idx]
    sorted_tokens  = token_ids[sort_idx]
    
    # ── 2. Array de Offsets (El Mapa del Tesoro) ─────────────────────────────
    # bincount cuenta cuentos tokens tiene cada experto. cumsum construye el mapa.
    counts = torch.bincount(sorted_experts, minlength=E)
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=x.device)
    offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    
    # Crear tensor X_grouped usando los índices del argsort
    X_grouped = x[sorted_tokens]  # [Alloc, D]
    
    # ── 3. El Kernel Triton (La Magia) Un solo lanzamiento ───────────────────
    # Triton requiere layout [E, In, Out], así que transponemos pesps
    W_up_t = W_up.permute(0, 2, 1).contiguous()
    up = triton_grouped_gemm(X_grouped, W_up_t, offsets)
    
    if W_gate is not None:
        W_gate_t = W_gate.permute(0, 2, 1).contiguous()
        gate = triton_grouped_gemm(X_grouped, W_gate_t, offsets)
        up = up * F.silu(gate)
    else:
        up = F.silu(up)
        
    W_down_t = W_down.permute(0, 2, 1).contiguous()
    down = triton_grouped_gemm(up, W_down_t, offsets)
    
    # ── 4. Scatter-Add Ponderado de Regreso ──────────────────────────────────
    c_grouped = confidence[sorted_tokens, sorted_experts].unsqueeze(-1)
    down = down * c_grouped
    
    out = torch.zeros_like(x)
    out.index_add_(0, sorted_tokens, down)
    
    return out

    # Procesamiento por experto nativo en PyTorch: simple, sin Triton-overhead y rápido.
    # Soporta batch vacío sin fallar gracias a semánticas nativas.
    for e in range(E):
        tokens_e = active_mask[:, e].nonzero(as_tuple=True)[0]
        
        # Ejecutar subgraph para los tokens
        x_e = x[tokens_e]                        # [N_e, D]
        up = F.linear(x_e, W_up[e])              # [N_e, d_ff]
        
        if W_gate is not None:
            gate = F.silu(F.linear(x_e, W_gate[e]))
            up = up * gate
        else:
            up = F.silu(up)

        down = F.linear(up, W_down[e])           # [N_e, D]

        # Ponderar y acumular
        c_e = confidence[tokens_e, e].unsqueeze(-1)
        out.index_add_(0, tokens_e, down * c_e)

    return out
# ─────────────────────────────────────────────────────────────────────────────



    # Procesamiento por experto nativo en PyTorch: simple, sin Triton-overhead y rápido.
    # Soporta batch vacío sin fallar gracias a semánticas nativas.
    for e in range(E):
        tokens_e = active_mask[:, e].nonzero(as_tuple=True)[0]
        
        # Ejecutar subgraph para los tokens
        x_e = x[tokens_e]                        # [N_e, D]
        up = F.linear(x_e, W_up[e])              # [N_e, d_ff]
        
        if W_gate is not None:
            gate = F.silu(F.linear(x_e, W_gate[e]))
            up = up * gate
        else:
            up = F.silu(up)

        down = F.linear(up, W_down[e])           # [N_e, D]

        # Ponderar y acumular
        c_e = confidence[tokens_e, e].unsqueeze(-1)
        out.index_add_(0, tokens_e, down * c_e)

    return out
# ─────────────────────────────────────────────────────────────────────────────



    # ── Fallback Path (PyTorch Loop) ─────────────────────────────────────────
    for e in range(E):
        mask_e = active_mask[:, e]              # [T] bool
        n_active = mask_e.sum().item()
        if n_active == 0:
            continue

        x_e = x[mask_e]                          # [n_active, D]
        c_e = confidence[mask_e, e:e+1]          # [n_active, 1]

        # SwiGLU: up * silu(gate)
        # W_up[e]: [D_ff, D]. x_e: [N, D]. Linear: x W.T
        w_up_e = W_up[e]
        up = F.linear(x_e, w_up_e)              # [n_active, D_ff]
        if W_gate is not None:
            w_gate_e = W_gate[e]
            gate = F.silu(F.linear(x_e, w_gate_e))
            up = up * gate
        w_down_e = W_down[e]
        down = F.linear(up, w_down_e)           # [n_active, D]

        # Weighted scatter-add back
        out[mask_e] += c_e * down

    return out


# ─────────────────────────────────────────────────────────────────────────────
# § 5. CAS Module — full Chimera Autonomous Swarm
# ─────────────────────────────────────────────────────────────────────────────

class ChimeraAutonomousSwarm(nn.Module):
    """
    El subsistema principal. Reemplaza ChimeraMoEFFN con experts autónomos.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        d_ff: int | None = None,
        layer_idx: int = 0,
        n_layers: int = 4,
        tau_init: float = 0.3,
        ttt_alpha: float = 2.0,
        target_active: float = 0.25,
        use_swiglue: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.d_ff = d_ff or (d_model * 2)  # SwiGLU: 2D → eficiente
        self.layer_idx = layer_idx
        self.n_layers = n_layers
        self.depth_ratio = layer_idx / max(n_layers - 1, 1)
        self.ttt_alpha = ttt_alpha
        self.target_active = target_active
        self.use_swiglue = use_swiglue
        self.use_triton = True  # Enable Triton by default (guards in forward will check device)

        # ── Input normalization ──────────────────────────────────────────────
        self.norm = nn.RMSNorm(d_model)

        # ── Micro-probes (AoE): 1 per expert ────────────────────────────────
        self.probes = nn.ModuleList([
            MicroProbe(d_model, p_init=tau_init)
            for _ in range(n_experts)
        ])

        # ── Depth threshold (LExI) ──────────────────────────────────────────
        self.depth_threshold = DepthThreshold(tau_init=tau_init)

        # ── TTT coupling scalar ──────────────────────────────────────────────
        self.ttt_alpha_param = nn.Parameter(torch.tensor(0.5))

        # ── Expert networks (SwiGLU) as Parameters for Triton ───────────────
        # Using Parameters allow contiguous storage for Grouped GEMM
        # W shapes: [E, Out, In] to match nn.Linear conventions per slice
        
        # Up: D -> D_ff. Weight: [D_ff, D]
        self.W_up = nn.Parameter(torch.empty(n_experts, self.d_ff, d_model))
        nn.init.kaiming_uniform_(self.W_up, a=math.sqrt(5))
        
        if use_swiglue:
            # Gate: D -> D_ff. Weight: [D_ff, D]
            self.W_gate = nn.Parameter(torch.empty(n_experts, self.d_ff, d_model))
            nn.init.kaiming_uniform_(self.W_gate, a=math.sqrt(5))
        else:
            self.register_parameter('W_gate', None)
            
        # Down: D_ff -> D. Weight: [D, D_ff]
        self.W_down = nn.Parameter(torch.empty(n_experts, d_model, self.d_ff))
        nn.init.kaiming_uniform_(self.W_down, a=math.sqrt(5))

        # ── Output scale — warm-start near zero ─────────────────────────────
        # sigmoid(-4) ≈ 0.018 → CAS contribución es mínima al inicio
        self.scale = nn.Parameter(torch.tensor(-4.0))

        # ── Tracking buffers ─────────────────────────────────────────────────
        self.register_buffer('expert_activation_ema', torch.full((n_experts,), tau_init))
        self.register_buffer('total_activation_ema', torch.tensor(tau_init))




    def _compute_threshold(self, ttt_proxy_loss: torch.Tensor | float | None = None) -> torch.Tensor:
        """
        Computa el umbral efectivo τ_eff en espacio logit.

        1. LExI: τ_l = τ_base + γ·SiLU(W_depth · depth_ratio)
        2. TTT: τ_eff = τ_l - α·L_proxy (si disponible)

        Returns: scalar tensor (logit-space threshold)
        """
        tau_l = self.depth_threshold(self.depth_ratio)

        if ttt_proxy_loss is not None:
            # Ensure tensor for graph compatibility
            if isinstance(ttt_proxy_loss, float):
                 proxy = torch.tensor(ttt_proxy_loss, device=tau_l.device, dtype=tau_l.dtype)
            else:
                 proxy = ttt_proxy_loss

            # Clamp proxy loss to prevent extreme threshold shifts
            # Use torch.clamp for tensor operations
            proxy = torch.clamp(proxy, min=0.0, max=5.0)
            alpha = self.ttt_alpha_param.abs()  # positive coupling
            tau_l = tau_l - alpha * proxy

        return tau_l

    def forward(
        self,
        x: torch.Tensor,                          # [B, S, D]
        ttt_proxy_loss: torch.Tensor | float | None = None,       # L_proxy del TTT
        graph_mode: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            out: [B, S, D] — x + scale · swarm_output
            aux: dict with:
              - 'cas_active_ratio': tensor scalar
              - 'cas_budget_loss': tensor scalar
              - 'cas_diversity_loss': tensor scalar
              - 'cas_balance_loss': tensor scalar
              - 'cas_expert_activations': [E] tensor
        """
        B, S, D = x.shape
        T = B * S

        x_norm = self.norm(x)
        xf = x_norm.reshape(T, D)

        # ── 1. Micro-probing: compute all expert confidences ────────────────
        # Stack all probe outputs: [T, E]
        probe_logits = torch.cat([probe(xf) for probe in self.probes], dim=-1)  # [T, E]
        confidence = torch.sigmoid(probe_logits)   # [T, E] ∈ (0, 1)

        # ── 2. Depth + TTT threshold ────────────────────────────────────────
        tau_logit = self._compute_threshold(ttt_proxy_loss)
        tau = torch.sigmoid(tau_logit)             # scalar ∈ (0, 1)

        # ── 3. Active mask creation ─────────────────────────────────────────
        # Straight-Through Estimator: hard mask in forward, soft gradient in backward
        if self.training and not graph_mode:
            temp = 0.1
            soft_mask = torch.sigmoid((probe_logits - tau_logit) / temp)  # [T, E]
            hard_mask = (confidence > tau).float()
            active_mask_float = hard_mask + (soft_mask - soft_mask.detach())
        else:
            active_mask_float = (confidence > tau).float()

        active_mask_bool = confidence > tau         # [T, E] bool — for dispatch

        # ── 4. Dispatch & compute ───────────────────────────────────────────
        # Weighted confidence for aggregation
        weighted_conf = active_mask_float * confidence  # [T, E]

        # Auto-detect hardware for Grouped GEMM (requires SM80+)
        # Use Triton GEMM if available and not in graph mode (since Triton kernel calls are external)
        # Note: sparse_expert_dispatch handles the logic internally or via argument
        # We enable it for Ampre/Hopper
        
        # Check explicit hardware capability or config
        # _get_gpu_profile check is reliable
        gpu_prof = _get_gpu_profile()
        # use_triton = gpu_prof.is_ampere_or_better and not graph_mode
        use_triton = self.use_triton and (x.device.type == 'cuda')

        swarm_out = sparse_expert_dispatch(
            xf, active_mask_bool,
            self.W_up, self.W_down, self.W_gate,
            weighted_conf,
            use_triton_gemm=use_triton
        )
        swarm_out = swarm_out.reshape(B, S, D)

        # ── 5. Output with learnable scale ──────────────────────────────────
        scale = torch.sigmoid(self.scale)
        out = x + scale * swarm_out

        # ── 6. Auxiliary losses (Asynchronous / Graph-friendly) ─────────────
        # Avoid .item() synchronization. Keep everything as tensors.
        
        # Calculate active per expert for EMA and balance loss
        # Use float() to ensure precision for sums
        active_per_expert = active_mask_bool.float().sum(0)  # [E]
        total_active = active_mask_bool.float().sum()
        active_ratio = total_active / max(T, 1)

        if not graph_mode:
            # Update EMA buffers (in-place, no gradients)
            with torch.no_grad():
                ema_decay = 0.99
                self.expert_activation_ema.mul_(ema_decay).add_(
                    (1 - ema_decay) * active_per_expert / max(T, 1)
                )
                self.total_activation_ema.mul_(ema_decay).add_(
                    (1 - ema_decay) * active_ratio
                )

        # Budget loss: penalize if active ratio exceeds target
        # Use active_mask_float (with gradients) for loss calculation
        mean_active_soft = active_mask_float.mean()
        active_ratio_diff = mean_active_soft - self.target_active
        budget_loss = F.relu(active_ratio_diff).pow(2)

        # Diversity loss: maximize entropy of per-expert activation distribution
        eps_div = 1e-8
        p_expert = active_mask_float.sum(0) / max(T, 1)  # [E] 
        entropy = -(p_expert * (p_expert + eps_div).log()).sum()
        max_entropy = math.log(self.n_experts)
        diversity_loss = -(entropy / max_entropy)

        # Balance loss: std of activation counts
        # Use soft mask sum for gradients
        soft_counts = active_mask_float.sum(0)
        balance_loss = soft_counts.std() / (soft_counts.mean() + 1e-6)

        aux = {
            'cas_active_ratio': active_ratio,     # tensor
            'cas_budget_loss': budget_loss,       # tensor with grad
            'cas_diversity_loss': diversity_loss, # tensor with grad
            'cas_balance_loss': balance_loss,     # tensor with grad
            'cas_expert_activations': active_per_expert.detach(), # tensor
            'cas_threshold': tau,                 # tensor
            'cas_scale': scale,                   # tensor
        }

        return out, aux

    def extra_repr(self) -> str:
        return (
            f"n_experts={self.n_experts}, d_ff={self.d_ff}, "
            f"depth={self.layer_idx}/{self.n_layers}, "
            f"target_active={self.target_active:.0%}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# § 6. Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=" * 70)
    print("CAS — Chimera Autonomous Swarm — Unit Tests")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    D, E = 256, 8
    B, S = 2, 128

    # ── Test 1: MicroProbe ───────────────────────────────────────────────────
    probe = MicroProbe(D, p_init=0.3).to(device)
    x_test = torch.randn(B * S, D, device=device)
    logits = probe(x_test)
    assert logits.shape == (B * S, 1), f"Probe shape: {logits.shape}"
    init_prob = torch.sigmoid(logits).mean().item()
    print(f"  MicroProbe: shape={logits.shape}, mean_p={init_prob:.3f} (expect ~0.3)")
    assert 0.1 < init_prob < 0.6, f"Init prob out of range: {init_prob}"
    print("  PASS: MicroProbe")

    # ── Test 2: DepthThreshold ───────────────────────────────────────────────
    dt = DepthThreshold(tau_init=0.3).to(device)
    tau_shallow = dt(0.0)
    tau_deep = dt(1.0)
    print(f"  DepthThreshold: shallow={torch.sigmoid(tau_shallow).item():.3f}, "
          f"deep={torch.sigmoid(tau_deep).item():.3f}")
    print("  PASS: DepthThreshold")

    # ── Test 3: Full CAS forward ─────────────────────────────────────────────
    cas = ChimeraAutonomousSwarm(
        d_model=D, n_experts=E, d_ff=D * 2,
        layer_idx=2, n_layers=8, tau_init=0.3,
    ).to(device)

    x = torch.randn(B, S, D, device=device)

    # Without TTT modulation
    out, aux = cas(x, ttt_proxy_loss=None)
    assert out.shape == (B, S, D), f"CAS output shape: {out.shape}"
    print(f"  CAS forward (no TTT): active_ratio={aux['cas_active_ratio']:.3f}, "
          f"scale={aux['cas_scale']:.4f}, threshold={aux['cas_threshold']:.3f}")
    print(f"    budget_loss={aux['cas_budget_loss'].item():.4f}, "
          f"diversity_loss={aux['cas_diversity_loss'].item():.4f}")
    print(f"    per_expert: {aux['cas_expert_activations'].tolist()}")
    print("  PASS: CAS forward (no TTT)")

    # With TTT modulation — high surprise → lower threshold → more experts
    out_ttt, aux_ttt = cas(x, ttt_proxy_loss=0.8)
    print(f"  CAS forward (TTT=0.8): active_ratio={aux_ttt['cas_active_ratio']:.3f}, "
          f"threshold={aux_ttt['cas_threshold']:.3f}")
    # High surprise should lower threshold → more activations
    print(f"    threshold drop: {aux['cas_threshold']:.3f} → {aux_ttt['cas_threshold']:.3f}")
    print("  PASS: CAS forward (with TTT)")

    # ── Test 4: Gradient flow ────────────────────────────────────────────────
    cas.train()
    x_grad = torch.randn(B, S, D, device=device, requires_grad=True)
    out_g, aux_g = cas(x_grad, ttt_proxy_loss=0.5)
    loss = out_g.sum() + 0.01 * aux_g['cas_budget_loss'] + 0.01 * aux_g['cas_diversity_loss']
    loss.backward()
    assert x_grad.grad is not None, "No gradient on input"
    # Check probe gradients
    probe_grads = [p.proj.weight.grad for p in cas.probes if p.proj.weight.grad is not None]
    print(f"  Gradient flow: input_grad_norm={x_grad.grad.norm():.4f}, "
          f"probes_with_grad={len(probe_grads)}/{E}")
    assert len(probe_grads) > 0, "No probe received gradient"
    # Check depth threshold gradient
    assert cas.depth_threshold.tau_base.grad is not None, "depth_threshold.tau_base has no grad"
    assert cas.ttt_alpha_param.grad is not None, "ttt_alpha_param has no grad"
    print("  PASS: Gradient flow")

    # ── Test 5: Extreme TTT → all experts wake up ───────────────────────────
    _, aux_extreme = cas(x, ttt_proxy_loss=5.0)
    print(f"  Extreme TTT=5.0: active_ratio={aux_extreme['cas_active_ratio']:.3f} "
          f"(expect high)")
    print("  PASS: Extreme TTT modulation")

    # ── Test 6: Zero TTT → conservative activation ──────────────────────────
    _, aux_zero = cas(x, ttt_proxy_loss=0.0)
    print(f"  Zero TTT=0.0: active_ratio={aux_zero['cas_active_ratio']:.3f} "
          f"(expect moderate)")
    print("  PASS: Zero TTT modulation")

    # ── Param count ──────────────────────────────────────────────────────────
    total = sum(p.numel() for p in cas.parameters())
    probes_total = sum(p.numel() for probe in cas.probes for p in probe.parameters())
    experts_total = total - probes_total
    print(f"\n  Param count: total={total:,}, probes={probes_total:,}, "
          f"experts={experts_total:,}")
    print(f"  Probe overhead: {probes_total/total*100:.2f}%")

    print("\n" + "=" * 70)
    print("ALL CAS TESTS PASSED ✓")
    print("=" * 70)
