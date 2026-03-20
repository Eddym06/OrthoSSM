import re

with open('cas_swarm.py', 'r') as f:
    text = f.read()

# I will append the Grouped GEMM to cas_swarm.py
triton_imports = """import triton
import triton.language as tl
"""

if "import triton" not in text:
    text = text.replace("import torch.nn.functional as F", "import torch.nn.functional as F\nimport triton\nimport triton.language as tl")

triton_kernel = """
# ─────────────────────────────────────────────────────────────────────────────
# § Elite Triton Grouped GEMM (Zero-Sync, Single Launch)
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
    # Cuadrícula [E, tiles_m, tiles_n] -> program_id(0) es Experto, program_id(1) es Tile de Tokens
    expert_id = tl.program_id(0)
    tile_m    = tl.program_id(1)
    tile_n    = tl.program_id(2)
    
    e_start = tl.load(Offsets_ptr + expert_id)
    e_end   = tl.load(Offsets_ptr + expert_id + 1)
    
    n_tokens = e_end - e_start
    if tile_m * BLOCK_M >= n_tokens:
        return # ★ Return instantáneo en hardware si el experto está vacío ★
        
    row_start = e_start + tile_m * BLOCK_M
    row_end = tl.minimum(row_start + BLOCK_M, e_end)
    
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # W es [E, D, D_FF], saltos de experto y token
    w_ptrs = W_ptr + expert_id * stride_we + offs_k[:, None] * stride_wk + offs_n[None, :]
    
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
    # x: [T_alloc, D_in], w: [E, D_in, D_out], offsets: [E+1]
    T, D = x.shape
    E, _, D_FF = w.shape
    y = torch.empty((T, D_FF), device=x.device, dtype=x.dtype)
    
    # max_tokens es solo heurística para grid, puede sobreestimar sin impacto
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
"""

# Replace in file using regex or append
# We can replace the existing sparse_expert_dispatch completely to include the argsort logic as requested.
# I'll just rewrite the dispatch to use the user's specific request.

new_dispatch = """def sparse_expert_dispatch(
    x: torch.Tensor,           # [T, D]
    active_mask: torch.Tensor, # [T, E] bool
    W_up: torch.Tensor,        # [E, D_ff, D]  (layout Out,In -> Triton needs In,Out so we permute)
    W_down: torch.Tensor,      # [E, D, D_ff]  -> permute as well
    W_gate: torch.Tensor | None, # [E, D_ff, D]
    confidence: torch.Tensor,  # [T, E]
    use_triton_gemm: bool = True,
) -> torch.Tensor:
    \"\"\"
    FRENTE 2: EL JEFE FINAL DEL BAJO NIVEL (Triton Grouped GEMM Elite).
    Despacho unificado usando empaquetado de tokens (argsort) y 1 solo lanzamiento de Triton.
    \"\"\"
    T, D = x.shape
    E = active_mask.shape[1]
    
    if T == 0:
        return torch.zeros_like(x)

    if not use_triton_gemm:
        # Fallback de PyTorch Vectorizado
        out = torch.zeros_like(x)
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
    
    return out"""

# Replace exact functions or just append and rewrite.
# Let's replace the whole section starting with def sparse_expert_dispatch
pattern = r"def sparse_expert_dispatch\(.*?\)\s*->\s*torch\.Tensor:.*?    return out"
match = re.search(pattern, text, flags=re.DOTALL)
if match:
    text = text[:match.start()] + new_dispatch + text[match.end():]
    
    # Now insert the triton kernel exactly above new_dispatch
    text = text.replace(new_dispatch, triton_kernel + "\n" + new_dispatch)
    
    with open('cas_swarm.py', 'w') as f:
        f.write(text)
    print("Patched cas_swarm.py with Triton Grouped GEMM")
else:
    print("Could not find sparse_expert_dispatch")
