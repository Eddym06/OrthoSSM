import re

with open('/home/OrthoSSM/chimera_h200/cas_swarm.py', 'r') as f:
    text = f.read()

# I want to scrub all definitions of triton_grouped_gemm, _grouped_gemm_fw_elite, GroupedGEMMFunction
# and replace them with a single clean one.

# Let's just match the first class/def and replace up to sparse_expert_dispatch
pattern = r"# ─────────────────────────────────────────────────────────────────────────────\n# § Elite Triton Grouped GEMM.*?(?=def sparse_expert_dispatch)"
match = bool(re.search(pattern, text, flags=re.DOTALL))
print("Match found:", match)

if match:
    new_triton = """# ─────────────────────────────────────────────────────────────────────────────
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

"""
    text = re.sub(pattern, new_triton, text, flags=re.DOTALL)
    with open('/home/OrthoSSM/chimera_h200/cas_swarm.py', 'w') as f:
        f.write(text)

