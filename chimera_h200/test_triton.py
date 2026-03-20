import torch
import triton
import triton.language as tl

@triton.jit
def grouped_gemm_fw_kernel(
    X_ptr, W_ptr, Y_ptr, Offsets_ptr,
    stride_xm, stride_xk,
    stride_we, stride_wk, stride_wn,
    stride_ym, stride_yn,
    D: tl.constexpr, D_FF: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 1. Binary search or linear search for expert_id? 
    # Much simpler: grid is 1D over experts, then loop over M tiles for that expert!
    # No! Grid is (E, cdiv(max_M, BLOCK_M), cdiv(D_FF, BLOCK_N))
    # It minimizes atomics.
    expert_id = tl.program_id(0)
    tile_m    = tl.program_id(1)
    tile_n    = tl.program_id(2)
    
    e_start = tl.load(Offsets_ptr + expert_id)
    e_end   = tl.load(Offsets_ptr + expert_id + 1)
    
    n_tokens = e_end - e_start
    if tile_m * BLOCK_M >= n_tokens:
        return # Instant return! (User's request)
        
    row_start = e_start + tile_m * BLOCK_M
    row_end = tl.minimum(row_start + BLOCK_M, e_end)
    
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # pointers
    # W is [E, D, D_FF], stride_we = D * D_FF, stride_wk = D_FF, stride_wn = 1
    w_ptrs = W_ptr + expert_id * stride_we + offs_k[:, None] * stride_wk + offs_n[None, :]
    
    for k in range(0, tl.cdiv(D, BLOCK_K)):
        k_start = k * BLOCK_K
        offs_k_curr = k_start + offs_k
        
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k_curr[None, :] * stride_xk
        
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < row_end) & (offs_k_curr[None, :] < D), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k_curr[:, None] < D) & (offs_n[None, :] < D_FF), other=0.0)
        
        acc += tl.dot(x, w)
        w_ptrs += BLOCK_K * stride_wk
        
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < row_end) & (offs_n[None, :] < D_FF))


def triton_grouped_gemm_fw(x: torch.Tensor, w: torch.Tensor, offsets: torch.Tensor):
    T, D = x.shape
    E, _, D_FF = w.shape
    y = torch.zeros((T, D_FF), device=x.device, dtype=x.dtype)
    
    counts = offsets[1:] - offsets[:-1]
    max_tokens = counts.max().item()
    if max_tokens == 0: return y
    
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 64
    
    grid = (E, triton.cdiv(max_tokens, BLOCK_M), triton.cdiv(D_FF, BLOCK_N))
    
    grouped_gemm_fw_kernel[grid](
        x, w, y, offsets,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1), w.stride(2),
        y.stride(0), y.stride(1),
        D, D_FF,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return y

# Test
x = torch.randn(10, 32, device='cuda')
w = torch.randn(2, 32, 64, device='cuda')
offsets = torch.tensor([0, 5, 10], dtype=torch.int32, device='cuda')
y = triton_grouped_gemm_fw(x, w, offsets)

y_ref = torch.cat([x[:5] @ w[0], x[5:] @ w[1]], dim=0)
print(f"Diff: {(y - y_ref).abs().max().item()}")
