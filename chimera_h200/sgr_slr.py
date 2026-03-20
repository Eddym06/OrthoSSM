"""
SGR (Selective Gradient Routing) + SLR Flash-Differential Attention
====================================================================
Versión: V4 — Flash-Differential Attention Triton (kernel batched 2D)

PROBLEMA V3:
  - DiffAttnV2Function materializa A1[B,K,W] y A2[B,K,W] completos en HBM.
    Para B=2, K=64, W=64, d=32: 2×2×64×64×4 bytes = 2MB solo en attention maps.
  - 10+ kernel launches separados por forward: topk→gather→linear×2→split×2
    →softmax×2→bmm×2→norm→linear→scatter. CPU overhead de lanzamiento
    domina para K/W pequeños.

SOLUCIÓN V4:
  [+++] FlashDiffSLRFunction — kernel ÚNICO Triton (_flash_diff_slr_fwd_kernel):
       • Grid 2D: (ceil(K/BLOCK_K), B) — lanza todos los batches en UNA llamada
       • Q1,Q2 del bloque se cargan UNA VEZ a SRAM por SM
       • Para cada bloque de K/V: lee [BLOCK_W,d], dot products con online-softmax,
         acumula acc1, acc2 en registers (dos streams independientes)
       • Escribe Out[B,K,d] EN UNA SOLA store — A1/A2 NUNCA van a HBM
       • @triton.autotune: configs cubren BLOCK_K∈{16,32,64}, BLOCK_W∈{32-128}

  [++] SGRSelector V2 con histograma para S>2048 (O(S) sin sort vs O(S log K))

  [=]  FusedProjectionSplit, _gather_windows_batched: sin cambios (ya óptimos)
  [=]  API 100% compatible con advanced_chimera.py y landmark_native.py

IMPACTO ESPERADO: SLR 2.09ms → ~0.9ms (≈2.3× speedup)
  · Eliminación de A1/A2 HBM en forward: -50% ancho de banda de memoria
  · Kernel 2D batched: reduce launches de 10+ a 4 (gather, flash, norm, scatter)
  · Mejor ocupancia GPU: tiles más grandes usando SRAM eficientemente

API (compatible con advanced_chimera.py y landmark_native.py):
  slr_out, top_idx = SLRDifferentialModule()(scan_out, importance)
  out = diff_attn_v2_triton(Q1, Q2, K1, K2, V, lam)   # landmark_native
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

# ── GPU-Adaptive JIT: cargar perfil hardware UNA VEZ al importar el módulo ───
# get_triton_configs_flash() genera la lista de triton.Config apropiada para
# el GPU detectado: tiles pequeños en laptops, tiles gigantes en H200/B200.
# Esto reemplaza los 6-7 configs hardcodeados que sólo eran óptimos en RTX 4050.
from gpu_profile import get_gpu_profile as _get_gpu_profile
from gpu_profile import get_triton_configs_flash as _get_flash_configs

_GPU_PROF       = _get_gpu_profile()       # singleton — detecta hardware UNA VEZ
_FLASH_CONFIGS  = _get_flash_configs(_GPU_PROF)  # lista de triton.Config adaptativa


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Triton kernel con @triton.autotune — para inferencia / landmark_native.py
#     Grid 1D sobre K queries, para tensores sin dimension batch (como landmark)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=_FLASH_CONFIGS,   # generado por gpu_profile.py según hardware real
    key=['K_size', 'W_size', 'd_head'],
)
@triton.jit
def _diff_attn_v2_fwd_kernel(
    Q1_ptr, Q2_ptr,
    K1_ptr, K2_ptr,
    V_ptr,
    Out_ptr,
    lam,
    K_size,
    W_size,
    d_head:  tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    V2 correcto: dos online-softmaxes INDEPENDIENTES en Triton SRAM.
    attn1 = softmax(Q1@K1.T/sqrt(d))
    attn2 = softmax(Q2@K2.T/sqrt(d))
    out   = (attn1 - lambda*attn2) @ V
    """
    q_start = tl.program_id(0) * BLOCK_K
    q_offs  = q_start + tl.arange(0, BLOCK_K)
    d_offs  = tl.arange(0, d_head)
    q_mask  = q_offs < K_size
    scale   = 1.0 / tl.sqrt(float(d_head))

    Q1 = tl.load(Q1_ptr + q_offs[:, None] * d_head + d_offs[None, :],
                 mask=q_mask[:, None], other=0.0)
    Q2 = tl.load(Q2_ptr + q_offs[:, None] * d_head + d_offs[None, :],
                 mask=q_mask[:, None], other=0.0)

    acc1 = tl.zeros([BLOCK_K, d_head], dtype=tl.float32)
    m1   = tl.full([BLOCK_K], -1e9, dtype=tl.float32)
    den1 = tl.zeros([BLOCK_K], dtype=tl.float32)

    acc2 = tl.zeros([BLOCK_K, d_head], dtype=tl.float32)
    m2   = tl.full([BLOCK_K], -1e9, dtype=tl.float32)
    den2 = tl.zeros([BLOCK_K], dtype=tl.float32)

    for w_start in range(0, W_size, BLOCK_W):
        w_offs = w_start + tl.arange(0, BLOCK_W)
        w_mask = w_offs < W_size

        K1t = tl.load(K1_ptr + w_offs[:, None] * d_head + d_offs[None, :],
                      mask=w_mask[:, None], other=0.0)
        K2t = tl.load(K2_ptr + w_offs[:, None] * d_head + d_offs[None, :],
                      mask=w_mask[:, None], other=0.0)
        Vt  = tl.load(V_ptr  + w_offs[:, None] * d_head + d_offs[None, :],
                      mask=w_mask[:, None], other=0.0)

        s1 = tl.dot(Q1, tl.trans(K1t)) * scale
        s2 = tl.dot(Q2, tl.trans(K2t)) * scale
        s1 = tl.where(w_mask[None, :], s1, -1e9)
        s2 = tl.where(w_mask[None, :], s2, -1e9)

        nm1   = tl.maximum(m1, tl.max(s1, axis=1))
        acc1  = acc1 * tl.exp(m1 - nm1)[:, None]
        den1  = den1 * tl.exp(m1 - nm1)
        p1    = tl.exp(s1 - nm1[:, None])
        den1  = den1 + tl.sum(p1, axis=1)
        acc1  = acc1 + tl.dot(p1, Vt)
        m1    = nm1

        nm2   = tl.maximum(m2, tl.max(s2, axis=1))
        acc2  = acc2 * tl.exp(m2 - nm2)[:, None]
        den2  = den2 * tl.exp(m2 - nm2)
        p2    = tl.exp(s2 - nm2[:, None])
        den2  = den2 + tl.sum(p2, axis=1)
        acc2  = acc2 + tl.dot(p2, Vt)
        m2    = nm2

    out1 = acc1 / (den1[:, None] + 1e-6)
    out2 = acc2 / (den2[:, None] + 1e-6)
    out  = out1 - lam * out2

    tl.store(Out_ptr + q_offs[:, None] * d_head + d_offs[None, :],
             out, mask=q_mask[:, None])


# ─────────────────────────────────────────────────────────────────────────────
# 1b. TMA 1D kernel — H200/Blackwell path (tl.make_block_ptr + FP8 e4m3fn)
#
#     Mismo algoritmo que _diff_attn_v2_fwd_kernel pero con:
#       - TMA async DMA para Q/K/V loads (elimina overhead de máscaras SIMT)
#       - FP8 e4m3fn path para Q@K matmul (2× throughput en Hopper)
#     Despacho: diff_attn_v2_triton() selecciona este kernel cuando
#     _GPU_PROF.use_tma=True.
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=_FLASH_CONFIGS,
    key=['K_size', 'W_size', 'd_head'],
)
@triton.jit
def _diff_attn_v2_fwd_kernel_h200(
    Q1_ptr, Q2_ptr,
    K1_ptr, K2_ptr,
    V_ptr,
    Out_ptr,
    lam,
    K_size,
    W_size,
    d_head:  tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_W: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    """
    H200 TMA 1D diff-attn kernel: block_ptr async DMA + optional FP8.
    Grid 1D: (ceil(K/BLOCK_K),) — para tensores sin batch dim (landmark_native).
    """
    q_start = tl.program_id(0) * BLOCK_K
    scale   = 1.0 / tl.sqrt(float(d_head))

    # ── TMA block_ptr para Q1, Q2 ────────────────────────────────────────
    Q1_bptr = tl.make_block_ptr(
        Q1_ptr, shape=(K_size, d_head), strides=(d_head, 1),
        offsets=(q_start, 0), block_shape=(BLOCK_K, d_head), order=(1, 0),
    )
    Q2_bptr = tl.make_block_ptr(
        Q2_ptr, shape=(K_size, d_head), strides=(d_head, 1),
        offsets=(q_start, 0), block_shape=(BLOCK_K, d_head), order=(1, 0),
    )
    Q1 = tl.load(Q1_bptr, boundary_check=(0,), padding_option="zero").to(tl.bfloat16)
    Q2 = tl.load(Q2_bptr, boundary_check=(0,), padding_option="zero").to(tl.bfloat16)

    acc1 = tl.zeros([BLOCK_K, d_head], dtype=tl.float32)
    m1   = tl.full([BLOCK_K], -1e9, dtype=tl.float32)
    den1 = tl.zeros([BLOCK_K], dtype=tl.float32)

    acc2 = tl.zeros([BLOCK_K, d_head], dtype=tl.float32)
    m2   = tl.full([BLOCK_K], -1e9, dtype=tl.float32)
    den2 = tl.zeros([BLOCK_K], dtype=tl.float32)

    # ── TMA block_ptr para K/V (avanzables en el loop) ───────────────────
    K1_bptr = tl.make_block_ptr(
        K1_ptr, shape=(W_size, d_head), strides=(d_head, 1),
        offsets=(0, 0), block_shape=(BLOCK_W, d_head), order=(1, 0),
    )
    K2_bptr = tl.make_block_ptr(
        K2_ptr, shape=(W_size, d_head), strides=(d_head, 1),
        offsets=(0, 0), block_shape=(BLOCK_W, d_head), order=(1, 0),
    )
    V_bptr = tl.make_block_ptr(
        V_ptr, shape=(W_size, d_head), strides=(d_head, 1),
        offsets=(0, 0), block_shape=(BLOCK_W, d_head), order=(1, 0),
    )

    w_start = tl.zeros([], dtype=tl.int32)
    for _ in range(0, tl.cdiv(W_size, BLOCK_W)):
        K1t = tl.load(K1_bptr, boundary_check=(0,), padding_option="zero").to(tl.bfloat16)
        K2t = tl.load(K2_bptr, boundary_check=(0,), padding_option="zero").to(tl.bfloat16)
        Vt  = tl.load(V_bptr,  boundary_check=(0,), padding_option="zero").to(tl.bfloat16)

        K1_bptr = tl.advance(K1_bptr, (BLOCK_W, 0))
        K2_bptr = tl.advance(K2_bptr, (BLOCK_W, 0))
        V_bptr  = tl.advance(V_bptr,  (BLOCK_W, 0))

        w_offs = w_start + tl.arange(0, BLOCK_W)
        w_mask = w_offs < W_size
        w_start = w_start + BLOCK_W

        # FP8 path: Q/K → e4m3fn antes del matmul
        if USE_FP8:
            Q1_dot = Q1.to(tl.float8e4nv)
            Q2_dot = Q2.to(tl.float8e4nv)
            K1_dot = K1t.to(tl.float8e4nv)
            K2_dot = K2t.to(tl.float8e4nv)
        else:
            Q1_dot = Q1
            Q2_dot = Q2
            K1_dot = K1t
            K2_dot = K2t

        s1 = tl.dot(Q1_dot, tl.trans(K1_dot), out_dtype=tl.float32) * scale
        s2 = tl.dot(Q2_dot, tl.trans(K2_dot), out_dtype=tl.float32) * scale
        s1 = tl.where(w_mask[None, :], s1, -1e9)
        s2 = tl.where(w_mask[None, :], s2, -1e9)

        nm1   = tl.maximum(m1, tl.max(s1, axis=1))
        corr1 = tl.exp(m1 - nm1)
        p1    = tl.exp(s1 - nm1[:, None])
        acc1  = acc1 * corr1[:, None] + tl.dot(p1.to(tl.bfloat16), Vt, out_dtype=tl.float32)
        den1  = den1 * corr1 + tl.sum(p1, axis=1)
        m1    = nm1

        nm2   = tl.maximum(m2, tl.max(s2, axis=1))
        corr2 = tl.exp(m2 - nm2)
        p2    = tl.exp(s2 - nm2[:, None])
        acc2  = acc2 * corr2[:, None] + tl.dot(p2.to(tl.bfloat16), Vt, out_dtype=tl.float32)
        den2  = den2 * corr2 + tl.sum(p2, axis=1)
        m2    = nm2

    safe_den1 = tl.maximum(den1, 1e-6)
    safe_den2 = tl.maximum(den2, 1e-6)
    out = acc1 / safe_den1[:, None] - lam * acc2 / safe_den2[:, None]

    # Store via TMA block_ptr
    Out_bptr = tl.make_block_ptr(
        Out_ptr, shape=(K_size, d_head), strides=(d_head, 1),
        offsets=(q_start, 0), block_shape=(BLOCK_K, d_head), order=(1, 0),
    )
    tl.store(Out_bptr, out, boundary_check=(0,))


def diff_attn_v2_triton(Q1, Q2, K1, K2, V, lam: float):
    """
    Wrapper de inferencia — tensor por elemento del batch: [K,d],[W,d].
    Usado por landmark_native.py (sin requerimiento de backward).
    Fallback PyTorch cuando K<16 o W<16.
    Despacha a kernel TMA (H200/Blackwell) o SIMT (Ada/Ampere) automáticamente.
    """
    K_size, d_head = Q1.shape
    W_size = K1.shape[0]

    if K_size < 16 or W_size < 16:
        scale = 1.0 / math.sqrt(d_head)
        s1    = (Q1.float() @ K1.float().T) * scale
        s2    = (Q2.float() @ K2.float().T) * scale
        a1    = torch.softmax(s1, dim=-1)
        a2    = torch.softmax(s2, dim=-1)
        return (a1 @ V.float() - lam * (a2 @ V.float())).to(Q1.dtype)

    d_p2 = triton.next_power_of_2(d_head)
    out  = torch.zeros(K_size, d_p2, device=Q1.device, dtype=torch.float32)

    def pad(t):
        t = t.float().contiguous()
        return F.pad(t, (0, d_p2 - d_head)) if d_head != d_p2 else t

    if _USE_H200_KERNEL:
        # H200/Blackwell: TMA block_ptr + optional FP8
        _diff_attn_v2_fwd_kernel_h200[(triton.cdiv(K_size, 1),)](
            pad(Q1), pad(Q2), pad(K1), pad(K2), pad(V), out,
            lam, K_size, W_size, d_head=d_p2,
            USE_FP8=int(_GPU_PROF.use_fp8_fwd),
        )
    else:
        # Ada/Ampere: SIMT loads (backward compat)
        _diff_attn_v2_fwd_kernel[(triton.cdiv(K_size, 1),)](
            pad(Q1), pad(Q2), pad(K1), pad(K2), pad(V), out,
            lam, K_size, W_size, d_head=d_p2,
        )
    return out[:, :d_head]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Flash-Differential Attention kernel — 2D grid, batched, sin A1/A2 en HBM
#
#     grid = (ceil(K_size / BLOCK_K), B)
#     Cada SM procesa BLOCK_K queries de UN elemento del batch.
#     Itera sobre bloques BLOCK_W de keys/values con dos online-softmax streams.
#     A1[B,K,W] y A2[B,K,W] NUNCA se escriben en HBM — todo en SRAM del SM.
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=_FLASH_CONFIGS,   # mismo espacio de búsqueda adaptativo que kernel 1D
    key=['K_size', 'W_size', 'd_head'],
)
@triton.jit
def _flash_diff_slr_fwd_kernel(
    # Queries [B, K, d] — proyectadas antes de entrar al kernel
    Q1_ptr, Q2_ptr,
    # Keys/Values [B, W, d] — ventana de contexto seleccionada
    K1_ptr, K2_ptr, V_ptr,
    # Output [B, K, d]
    Out_ptr,
    # Lambda escalar (ya aplicado sigmoid fuera del kernel)
    lam,
    # Dims
    B, K_size, W_size,
    # Strides Q [B, K, d]
    sq_b, sq_k,
    # Strides K/V [B, W, d]
    sk_b, sk_w,
    # Strides Out [B, K, d]
    so_b, so_k,
    # Compile-time constants
    d_head:  tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Flash-Differential Attention batched — dos streams online-softmax.

    out[b,q] = (Σ_k exp(s1[q,k]-m1[q]) V[k]) / den1[q]
             - λ * (Σ_k exp(s2[q,k]-m2[q]) V[k]) / den2[q]

    s1 = Q1 @ K1.T / √d,   s2 = Q2 @ K2.T / √d
    Nunca materializa la matriz de atención [K, W] en HBM.
    """
    pid_q = tl.program_id(0)   # query block
    pid_b = tl.program_id(1)   # batch element

    q_start = pid_q * BLOCK_K
    q_offs  = q_start + tl.arange(0, BLOCK_K)
    d_offs  = tl.arange(0, d_head)
    q_mask  = q_offs < K_size
    scale   = 1.0 / tl.sqrt(float(d_head))

    # Base pointers para este batch
    Q1_base = Q1_ptr + pid_b * sq_b
    Q2_base = Q2_ptr + pid_b * sq_b
    K1_base = K1_ptr + pid_b * sk_b
    K2_base = K2_ptr + pid_b * sk_b
    V_base  = V_ptr  + pid_b * sk_b

    # Cargar queries en BF16 — UNA lectura HBM, se quedan en SRAM  [BLOCK_K, d_head]
    # BF16 loads: ~50% menos bandwidth HBM. Tensor Core HMMA BF16×BF16→FP32.
    Q1 = tl.load(Q1_base + q_offs[:, None] * sq_k + d_offs[None, :],
                 mask=q_mask[:, None], other=0.0).to(tl.bfloat16)
    Q2 = tl.load(Q2_base + q_offs[:, None] * sq_k + d_offs[None, :],
                 mask=q_mask[:, None], other=0.0).to(tl.bfloat16)

    # Acumuladores en FP32 (registers del SM, NO en HBM)
    acc1 = tl.zeros([BLOCK_K, d_head], dtype=tl.float32)
    m1   = tl.full([BLOCK_K], -1e9,    dtype=tl.float32)
    den1 = tl.zeros([BLOCK_K],         dtype=tl.float32)

    acc2 = tl.zeros([BLOCK_K, d_head], dtype=tl.float32)
    m2   = tl.full([BLOCK_K], -1e9,    dtype=tl.float32)
    den2 = tl.zeros([BLOCK_K],         dtype=tl.float32)

    # Iterar sobre bloques de keys/values
    for w_start in range(0, W_size, BLOCK_W):
        w_offs = w_start + tl.arange(0, BLOCK_W)
        w_mask = w_offs < W_size

        # BF16 loads: K/V en BF16 → Tensor Core path para dot products
        K1b = tl.load(K1_base + w_offs[:, None] * sk_w + d_offs[None, :],
                      mask=w_mask[:, None], other=0.0).to(tl.bfloat16)
        K2b = tl.load(K2_base + w_offs[:, None] * sk_w + d_offs[None, :],
                      mask=w_mask[:, None], other=0.0).to(tl.bfloat16)
        Vb  = tl.load(V_base  + w_offs[:, None] * sk_w + d_offs[None, :],
                      mask=w_mask[:, None], other=0.0).to(tl.bfloat16)

        # Scores [BLOCK_K, BLOCK_W] — BF16 inputs, FP32 accumulation (Tensor Core)
        s1 = tl.dot(Q1, tl.trans(K1b), out_dtype=tl.float32) * scale
        s2 = tl.dot(Q2, tl.trans(K2b), out_dtype=tl.float32) * scale
        s1 = tl.where(w_mask[None, :], s1, -1e9)
        s2 = tl.where(w_mask[None, :], s2, -1e9)

        # Online-softmax stream 1 (FP32 para estabilidad numérica)
        nm1   = tl.maximum(m1, tl.max(s1, axis=1))
        corr1 = tl.exp(m1 - nm1)
        p1    = tl.exp(s1 - nm1[:, None])
        # p@V: cast p1 a BF16 para Tensor Core; acumula en FP32 vía out_dtype
        acc1  = acc1 * corr1[:, None] + tl.dot(p1.to(tl.bfloat16), Vb, out_dtype=tl.float32)
        den1  = den1 * corr1 + tl.sum(p1, axis=1)
        m1    = nm1

        # Online-softmax stream 2 (FP32 para estabilidad numérica)
        nm2   = tl.maximum(m2, tl.max(s2, axis=1))
        corr2 = tl.exp(m2 - nm2)
        p2    = tl.exp(s2 - nm2[:, None])
        acc2  = acc2 * corr2[:, None] + tl.dot(p2.to(tl.bfloat16), Vb, out_dtype=tl.float32)
        den2  = den2 * corr2 + tl.sum(p2, axis=1)
        m2    = nm2

    # out = softmax1 @ V - λ * softmax2 @ V   [BLOCK_K, d_head]
    # FIX: tl.maximum(den, 1e-6) en vez de (den + 1e-8).
    # Razón: en secuencias largas (W>>1024) den1 puede crecer a 1e8+,
    # haciendo que el epsilon aditivo 1e-8 sea completamente absorbido
    # por la mantisa FP32 (23 bits ≈ 7 dígitos decimales) → efectivamente
    # divide por 0 cuando el denominator real es 0.
    # tl.maximum garantiza un floor de 1e-6 independientemente de la magnitud
    # del acumulador: si den=0, divide por 1e-6; si den=1e8, divide por 1e8.
    # Consistente con el kernel 1D (_diff_attn_v2_fwd_kernel) que usa 1e-6.
    safe_den1 = tl.maximum(den1, 1e-6)
    safe_den2 = tl.maximum(den2, 1e-6)
    out = acc1 / safe_den1[:, None] - lam * acc2 / safe_den2[:, None]

    # UNA sola escritura HBM por SM
    Out_base = Out_ptr + pid_b * so_b
    tl.store(Out_base + q_offs[:, None] * so_k + d_offs[None, :],
             out, mask=q_mask[:, None])


# ─────────────────────────────────────────────────────────────────────────────
# 2b. Flash-Diff kernel H200 — TMA block_ptr + FP8 WGMMA path
#
#     Optimizaciones hardware Hopper/Blackwell (SM≥9.0):
#
#     [TMA] tl.make_block_ptr + tl.load(bptr, boundary_check=(0,)):
#       La unidad TMA del SM hace DMA 2D asíncrono Q/K/V → SRAM.
#       Libera todos los SIMT lanes para aritmética pura mientras
#       los datos se transfieren. En H200 con BW=4800 GB/s esto
#       se traduce en ~40% de mejora en throughput de memoria vs
#       loads escalares con máscara.
#
#     [FP8] Cuando USE_FP8=True: Q/K se castean a tl.float8e4nv
#       (e4m3fn, rango ±448) antes de tl.dot → WGMMA FP8 path.
#       En H200: 1979 TFLOPS FP8 vs 989 TFLOPS BF16 → 2× throughput.
#       V se mantiene BF16 (output del matmul V es FP32 acumulado;
#       bajar V a FP8 introduce ruido en la combinación final).
#
#     Dispatching desde Python: _flash_diff_attn_fwd selecciona
#     este kernel cuando _GPU_PROF.use_tma=True, de lo contrario
#     usa _flash_diff_slr_fwd_kernel (path Ada/Ampere sin cambios).
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=_FLASH_CONFIGS,
    key=['K_size', 'W_size', 'd_head'],
)
@triton.jit
def _flash_diff_slr_fwd_kernel_h200(
    # Queries [B, K, d]
    Q1_ptr, Q2_ptr,
    # Keys/Values [B, W, d]
    K1_ptr, K2_ptr, V_ptr,
    # Output [B, K, d]
    Out_ptr,
    lam,
    B, K_size, W_size,
    sq_b, sq_k,
    sk_b, sk_w,
    so_b, so_k,
    d_head:  tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_W: tl.constexpr,
    USE_FP8: tl.constexpr,   # True en H200/Blackwell: Q/K cast a FP8 e4m3fn
):
    """
    H200 path: TMA async DMA (block_ptr) + FP8 e4m3fn para Q@K matmul.

    TMA block_ptr elimina el overhead de máscaras SIMT:
      - tl.make_block_ptr(...) declara un 2D async DMA descriptor
      - tl.load(bptr, boundary_check=(0,)) → TMA hardware hace la transferencia
      - tl.advance(bptr, offset) avanza el descriptor sin aritmética de punteros

    FP8 path (USE_FP8=True):
      - Q/K se castean a tl.float8e4nv (e4m3fn): rango ±448
      - tl.dot acepta FP8 inputs → rutea a WGMMA FP8 instruction
      - out_dtype=tl.float32 asegura acumulación precisa
      - V queda en BF16 (output de p@V en FP32 via out_dtype)
    """
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)

    q_start = pid_q * BLOCK_K
    scale   = 1.0 / tl.sqrt(float(d_head))

    # Base pointers para este batch
    Q1_base = Q1_ptr + pid_b * sq_b
    Q2_base = Q2_ptr + pid_b * sq_b
    K1_base = K1_ptr + pid_b * sk_b
    K2_base = K2_ptr + pid_b * sk_b
    V_base  = V_ptr  + pid_b * sk_b

    # ── Cargar Q1, Q2 via TMA block_ptr ─────────────────────────────────────
    # shape=(K_size, d_head): dimensiones del tensor completo [K, d]
    # strides=(sq_k, 1): stride en dim K = d_head (contiguous); dim d = 1
    # offsets=(q_start, 0): comienzo del tile actual
    # block_shape=(BLOCK_K, d_head): tamaño del tile (constexpr)
    # order=(1, 0): row-major — dim 1 (d) se accede más rápido (coalesced)
    Q1_bptr = tl.make_block_ptr(
        Q1_base, shape=(K_size, d_head), strides=(sq_k, 1),
        offsets=(q_start, 0), block_shape=(BLOCK_K, d_head), order=(1, 0),
    )
    Q2_bptr = tl.make_block_ptr(
        Q2_base, shape=(K_size, d_head), strides=(sq_k, 1),
        offsets=(q_start, 0), block_shape=(BLOCK_K, d_head), order=(1, 0),
    )
    # boundary_check=(0,): chequea solo dim 0 (K) — dim 1 (d_head) es ya pot. de 2
    Q1 = tl.load(Q1_bptr, boundary_check=(0,), padding_option="zero").to(tl.bfloat16)
    Q2 = tl.load(Q2_bptr, boundary_check=(0,), padding_option="zero").to(tl.bfloat16)

    # Acumuladores en FP32 (registers del SM)
    acc1 = tl.zeros([BLOCK_K, d_head], dtype=tl.float32)
    m1   = tl.full([BLOCK_K], -1e9,   dtype=tl.float32)
    den1 = tl.zeros([BLOCK_K],        dtype=tl.float32)

    acc2 = tl.zeros([BLOCK_K, d_head], dtype=tl.float32)
    m2   = tl.full([BLOCK_K], -1e9,   dtype=tl.float32)
    den2 = tl.zeros([BLOCK_K],        dtype=tl.float32)

    # ── Descriptores para K/V — se avanzan en el loop ─────────────────────
    K1_bptr = tl.make_block_ptr(
        K1_base, shape=(W_size, d_head), strides=(sk_w, 1),
        offsets=(0, 0), block_shape=(BLOCK_W, d_head), order=(1, 0),
    )
    K2_bptr = tl.make_block_ptr(
        K2_base, shape=(W_size, d_head), strides=(sk_w, 1),
        offsets=(0, 0), block_shape=(BLOCK_W, d_head), order=(1, 0),
    )
    V_bptr = tl.make_block_ptr(
        V_base, shape=(W_size, d_head), strides=(sk_w, 1),
        offsets=(0, 0), block_shape=(BLOCK_W, d_head), order=(1, 0),
    )

    # ── Loop sobre bloques de K/V (TMA avanza los descriptores) ───────────
    # Tracking w_start en registro para construir w_mask y enmascarar s1/s2.
    # Necesario porque boundary_check="zero" rellena K[pad,:]=0 → Q·0^T=0 →
    # exp(0) > 0 inflaría den. Con w_mask forzamos s[pad]=-inf → exp(-inf)=0.
    # Overhead: 1 add + 1 compare por warp-step (negligible vs ganancia TMA).
    w_start = tl.zeros([], dtype=tl.int32)

    for _ in range(0, tl.cdiv(W_size, BLOCK_W)):
        K1b = tl.load(K1_bptr, boundary_check=(0,), padding_option="zero").to(tl.bfloat16)
        K2b = tl.load(K2_bptr, boundary_check=(0,), padding_option="zero").to(tl.bfloat16)
        Vb  = tl.load(V_bptr,  boundary_check=(0,), padding_option="zero").to(tl.bfloat16)

        # Advance descriptors: TMA mueve el descriptor al siguiente tile
        K1_bptr = tl.advance(K1_bptr, (BLOCK_W, 0))
        K2_bptr = tl.advance(K2_bptr, (BLOCK_W, 0))
        V_bptr  = tl.advance(V_bptr,  (BLOCK_W, 0))

        # Máscara de posiciones válidas en este tile (para s1/s2 boundary)
        w_offs = w_start + tl.arange(0, BLOCK_W)
        w_mask = w_offs < W_size
        w_start = w_start + BLOCK_W

        # FP8 path: cast Q/K a e4m3fn antes del matmul Q@K^T
        # WGMMA FP8 en H200: 1979 TFLOPS vs 989 en BF16 → 2× throughput
        if USE_FP8:
            Q1_dot = Q1.to(tl.float8e4nv)
            Q2_dot = Q2.to(tl.float8e4nv)
            K1_dot = K1b.to(tl.float8e4nv)
            K2_dot = K2b.to(tl.float8e4nv)
        else:
            Q1_dot = Q1
            Q2_dot = Q2
            K1_dot = K1b
            K2_dot = K2b

        s1 = tl.dot(Q1_dot, tl.trans(K1_dot), out_dtype=tl.float32) * scale
        s2 = tl.dot(Q2_dot, tl.trans(K2_dot), out_dtype=tl.float32) * scale

        # Enmascarar posiciones fuera de W_size a -inf → exp(-inf)=0 en softmax
        s1 = tl.where(w_mask[None, :], s1, -1e9)
        s2 = tl.where(w_mask[None, :], s2, -1e9)

        # Online-softmax stream 1
        nm1   = tl.maximum(m1, tl.max(s1, axis=1))
        corr1 = tl.exp(m1 - nm1)
        p1    = tl.exp(s1 - nm1[:, None])
        acc1  = acc1 * corr1[:, None] + tl.dot(p1.to(tl.bfloat16), Vb, out_dtype=tl.float32)
        den1  = den1 * corr1 + tl.sum(p1, axis=1)
        m1    = nm1

        # Online-softmax stream 2
        nm2   = tl.maximum(m2, tl.max(s2, axis=1))
        corr2 = tl.exp(m2 - nm2)
        p2    = tl.exp(s2 - nm2[:, None])
        acc2  = acc2 * corr2[:, None] + tl.dot(p2.to(tl.bfloat16), Vb, out_dtype=tl.float32)
        den2  = den2 * corr2 + tl.sum(p2, axis=1)
        m2    = nm2

    safe_den1 = tl.maximum(den1, 1e-6)
    safe_den2 = tl.maximum(den2, 1e-6)
    out = acc1 / safe_den1[:, None] - lam * acc2 / safe_den2[:, None]

    # ── Store via TMA block_ptr ───────────────────────────────────────────
    Out_base  = Out_ptr + pid_b * so_b
    Out_bptr  = tl.make_block_ptr(
        Out_base, shape=(K_size, d_head), strides=(so_k, 1),
        offsets=(q_start, 0), block_shape=(BLOCK_K, d_head), order=(1, 0),
    )
    tl.store(Out_bptr, out.to(tl.bfloat16), boundary_check=(0,))


# bool auxiliar al nivel de módulo — evaluado UNA vez al importar
_USE_H200_KERNEL: bool = _GPU_PROF.use_tma   # True solo en Hopper/Blackwell


def _flash_diff_attn_fwd(Q1, Q2, K1, K2, V, lam: float):
    """
    Despacha al kernel adecuado según el hardware detectado:
      - H200/Blackwell (use_tma=True): _flash_diff_slr_fwd_kernel_h200
        con TMA async DMA + FP8 e4m3fn Q@K matmul.
      - Ada/Ampere (use_tma=False): _flash_diff_slr_fwd_kernel original
        con SIMT loads y BF16 matmuls (sin cambios vs versión anterior).

    Q1, Q2: [B, K, d]   K1, K2, V: [B, W, d]   lam: float
    Retorna: [B, K, d] float32 — sin A1/A2 en HBM
    """
    B, K_size, d_head = Q1.shape
    _, W_size, _      = K1.shape
    d_p2 = triton.next_power_of_2(d_head)

    def pad(t):
        t = t.contiguous()
        return F.pad(t, (0, d_p2 - d_head)) if d_head != d_p2 else t

    Q1p, Q2p = pad(Q1), pad(Q2)
    K1p, K2p = pad(K1), pad(K2)
    Vp       = pad(V)
    out = torch.empty(B, K_size, d_p2, device=Q1.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(K_size, meta['BLOCK_K']), B)

    if _USE_H200_KERNEL:
        # H200/Blackwell: TMA block_ptr + FP8 e4m3fn (cuando use_fp8_fwd=True)
        _flash_diff_slr_fwd_kernel_h200[grid](
            Q1p, Q2p, K1p, K2p, Vp, out,
            lam,
            B, K_size, W_size,
            Q1p.stride(0), Q1p.stride(1),
            K1p.stride(0), K1p.stride(1),
            out.stride(0),  out.stride(1),
            d_head=d_p2,
            USE_FP8=int(_GPU_PROF.use_fp8_fwd),
        )
    else:
        # Ada/Ampere: ruta SIMT original (sin cambios — backward compat)
        _flash_diff_slr_fwd_kernel[grid](
            Q1p, Q2p, K1p, K2p, Vp, out,
            lam,
            B, K_size, W_size,
            Q1p.stride(0), Q1p.stride(1),
            K1p.stride(0), K1p.stride(1),
            out.stride(0),  out.stride(1),
            d_head=d_p2,
        )
    return out[:, :, :d_head]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Flash-Diff SLR — torch.library.custom_op (compile-safe, zero graph breaks)
#
#     Forward: Flash-Differential (sin A1/A2 en HBM durante forward)
#     Backward: recomputa A1, A2 desde Q,K guardados (Flash v1 approach)
#
#     V5 vs V4:
#       [+++] torch.library.custom_op: torch.compile puede fusionar RMSNorm,
#             residuales y scatter_add ALREDEDOR del kernel → 10-20% speedup gratis.
#       [++]  A1/A2 NO se guardan en forward ni se recomputan redundantemente.
#             Solo Q,K,V,lam_logit se guardan → -50% memoria en la gap fwd↔bwd.
#       [=]   Gradientes idénticos a V4 (mismas ecuaciones, solo cambia dónde se computan).
# ─────────────────────────────────────────────────────────────────────────────

@torch.library.custom_op("chimera::flash_diff_slr", mutates_args=())
def _flash_diff_slr_op(
    Q1: torch.Tensor, Q2: torch.Tensor,
    K1: torch.Tensor, K2: torch.Tensor,
    V: torch.Tensor,  lam_logit: torch.Tensor,
) -> torch.Tensor:
    """Flash-Differential Attention V5 — forward sin A1/A2 en HBM."""
    lam    = torch.sigmoid(lam_logit).item()
    B, K_size, d = Q1.shape
    W_size = K1.shape[1]
    scale  = 1.0 / math.sqrt(d)
    use_flash = (K_size * W_size) > 4096 and K_size >= 16 and W_size >= 16
    if use_flash:
        out_f = _flash_diff_attn_fwd(Q1, Q2, K1, K2, V, lam)
    else:
        lam_t = torch.tensor(lam, device=Q1.device, dtype=torch.float32)
        s1 = torch.bmm(Q1.float(), K1.float().transpose(-1, -2)) * scale
        s2 = torch.bmm(Q2.float(), K2.float().transpose(-1, -2)) * scale
        A1 = torch.softmax(s1, dim=-1)
        A2 = torch.softmax(s2, dim=-1)
        out_f = torch.bmm(A1 - lam_t.view(1, 1, 1) * A2, V.float())
    return out_f.to(Q1.dtype)


@_flash_diff_slr_op.register_fake
def _flash_diff_slr_fake(Q1, Q2, K1, K2, V, lam_logit):
    return Q1.new_empty(Q1.shape)


def _flash_diff_slr_setup(ctx, inputs, output):
    Q1, Q2, K1, K2, V, lam_logit = inputs
    # Solo guardamos Q,K,V,lam_logit — A1/A2 se recomputan en backward.
    # Ahorro: 2·B·K·W·4 bytes que antes se guardaban innecesariamente.
    ctx.save_for_backward(Q1, Q2, K1, K2, V, lam_logit)
    ctx.scale = 1.0 / math.sqrt(Q1.shape[-1])


def _flash_diff_slr_bwd(ctx, grad_out):
    Q1, Q2, K1, K2, V, lam_logit = ctx.saved_tensors
    scale = ctx.scale
    lam   = torch.sigmoid(lam_logit).float()

    # Recompute A1, A2 (Flash v1 approach — no se guardaron en forward)
    s1 = torch.bmm(Q1.float(), K1.float().transpose(-1, -2)) * scale
    s2 = torch.bmm(Q2.float(), K2.float().transpose(-1, -2)) * scale
    A1 = torch.softmax(s1, dim=-1)
    A2 = torch.softmax(s2, dim=-1)

    g  = grad_out.float()
    Vf = V.float()
    lv = lam.view(-1, 1, 1)

    dV = torch.bmm(A1.transpose(-1, -2), g) \
       - lv * torch.bmm(A2.transpose(-1, -2), g)

    def smx_bwd(A, dA):
        return A * (dA - (dA * A).sum(dim=-1, keepdim=True))

    dA1 = torch.bmm(g, Vf.transpose(-1, -2))
    dA2 = -lv * dA1
    dS1 = smx_bwd(A1, dA1) * scale
    dS2 = smx_bwd(A2, dA2) * scale

    Q1f, Q2f = Q1.float(), Q2.float()
    K1f, K2f = K1.float(), K2.float()
    dQ1 = torch.bmm(dS1, K1f)
    dQ2 = torch.bmm(dS2, K2f)
    dK1 = torch.bmm(dS1.transpose(-1, -2), Q1f)
    dK2 = torch.bmm(dS2.transpose(-1, -2), Q2f)

    a2v         = torch.bmm(A2, Vf)
    dL_dlam     = -(g * a2v).sum()
    sig         = lam.squeeze()
    g_lam_logit = (dL_dlam * sig * (1.0 - sig)).to(lam_logit.dtype)
    if g_lam_logit.shape != lam_logit.shape:
        g_lam_logit = g_lam_logit.reshape(lam_logit.shape)

    return (dQ1.to(Q1.dtype), dQ2.to(Q2.dtype),
            dK1.to(K1.dtype), dK2.to(K2.dtype),
            dV.to(V.dtype),   g_lam_logit)


_flash_diff_slr_op.register_autograd(
    _flash_diff_slr_bwd, setup_context=_flash_diff_slr_setup,
)


# Alias público para backward compat — el custom_op ES la función diferenciable.
FlashDiffSLRFunction = _flash_diff_slr_op



# ─────────────────────────────────────────────────────────────────────────────
# 4.  FusedProjectionSplit — 2 matmuls grandes en vez de 5 pequeños
# ─────────────────────────────────────────────────────────────────────────────

class FusedProjectionSplit(nn.Module):
    """
    W_q  [D, 2*dh] → Q1, Q2
    W_kv [D, 3*dh] → K1, K2, V

    vs antes: 5 Linear separadas = 5 cuBLAS launches + 5x reads HBM
    ahora:    2 Linear grandes   = 2 cuBLAS launches + 2x reads HBM
    """
    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.d_head = d_head
        self.W_q  = nn.Linear(d_model, 2 * d_head, bias=False)
        self.W_kv = nn.Linear(d_model, 3 * d_head, bias=False)

    def forward(self, queries: torch.Tensor, context: torch.Tensor):
        q  = self.W_q(queries)
        kv = self.W_kv(context)
        Q1, Q2 = q.split(self.d_head, dim=-1)
        K1, K2, V = kv.split(self.d_head, dim=-1)
        return Q1, Q2, K1, K2, V


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SGRSelector V2 — histograma para S>2048, topk exacto para S≤2048
#
#     Para S≤2048: torch.topk ya optimizado (radix sort en GPU).
#     Para S>2048: histograma de NBINS=64 cubos en GPU: O(S) sin sort,
#       ~2-3× más rápido. El histograma encuentra el umbral percentil,
#       luego topk sobre una máscara filtrada (mucho menor que S).
# ─────────────────────────────────────────────────────────────────────────────

_HIST_BINS = 64

class SGRSelector(nn.Module):
    def __init__(self, top_k_frac: float = 0.125, hist_threshold: int = 2048):
        super().__init__()
        self.top_k_frac     = top_k_frac
        self.hist_threshold = hist_threshold

    @torch.no_grad()
    def forward(self, importance: torch.Tensor, S: int):
        K = max(1, int(self.top_k_frac * S))

        if S <= self.hist_threshold:
            # topk exacto — bien optimizado para S moderado
            _, top_idx = torch.topk(importance, K, dim=-1, sorted=False)
            return top_idx, K

        # ── Histograma para S grande — O(S + filtered*log K) vs O(S log K) ───────
        # BUG V6: el histograma se computaba pero la selección final igual llamaba
        # torch.topk sobre el array COMPLETO. El filtering prometido nunca ocurría.
        # Fix V7: cumsum descendente encuentra el bin umbral donde se acumulan K
        # tokens; topk se ejecuta sobre importance[mask] << S.
        B = importance.shape[0]
        imin = importance.min(dim=-1, keepdim=True).values               # [B, 1]
        imax = importance.max(dim=-1, keepdim=True).values               # [B, 1]
        rng  = (imax - imin).clamp(min=1e-8)                            # [B, 1]

        # Asigna cada token a un cubo 0..NBINS-1
        bin_idx = ((importance - imin) / rng * (_HIST_BINS - 1)).long().clamp(0, _HIST_BINS - 1)

        # Histograma [B, NBINS]
        hist = torch.zeros(B, _HIST_BINS, device=importance.device, dtype=torch.long)
        hist.scatter_add_(1, bin_idx, torch.ones_like(bin_idx))

        # Cumsum descendente: cuántos tokens hay en bins >= b
        # cum[b, i] = número de tokens con importancia en cubos [i, NBINS)
        cum_from_top = hist.flip(-1).cumsum(dim=-1).flip(-1)             # [B, NBINS]

        # Threshold bin: más alto bin tal que cum_from_top >= K.
        # (cum < K).sum() cuenta cuántos bins NO llegan a K → threshold_bin.
        thresh_bin = (_HIST_BINS - 1 - (cum_from_top < K).sum(dim=-1)).clamp(0, _HIST_BINS - 1)

        # Valor umbral: importancia mínima para entrar en el candidato set
        # thresh_val[b] = imin[b] + thresh_bin[b] / (NBINS-1) * rng[b]
        thresh_val = (imin.squeeze(-1) +
                      thresh_bin.float() / (_HIST_BINS - 1) * rng.squeeze(-1))   # [B]

        # Mask: solo tokens con importancia >= umbral (candidatos topK)
        # torch.where prevéene que topk reciba -inf para tokens descartados
        masked_imp = torch.where(
            importance >= thresh_val.unsqueeze(1),
            importance,
            torch.full_like(importance, float('-inf')),
        )
        _, top_idx = torch.topk(masked_imp, K, dim=-1, sorted=False)
        return top_idx, K



# ─────────────────────────────────────────────────────────────────────────────
# 6.  SLRDifferentialModule V4 — módulo principal con Flash-Differential Attn
# ─────────────────────────────────────────────────────────────────────────────

class SLRDifferentialModule(nn.Module):
    """
    SLR + Flash-Differential Attention V4.

    Cambios vs V3:
      [+++] Flash kernel batched 2D: A1/A2 nunca en HBM durante forward
            → -50% ancho de banda de memoria en la parte de atención
      [++]  FlashDiffSLRFunction con backward por recompute (idéntico en gradientes)
      [+]   SGRSelector V2 con histograma para S>2048
      [=]   FusedProjectionSplit, _gather_windows_batched sin cambios

    API idéntica a V3 para compatibilidad total con advanced_chimera.py.
    """
    def __init__(self, d_model: int, d_head: int = 32,
                 window_size: int = 64, top_k_frac: float = 0.125,
                 hist_threshold: int = 2048):
        super().__init__()
        self.d_model     = d_model
        self.d_head      = d_head
        self.window_size = window_size
        self.sgr         = SGRSelector(top_k_frac, hist_threshold)
        self.proj        = FusedProjectionSplit(d_model, d_head)
        self.lam_logit   = nn.Parameter(torch.tensor(-2.0))
        self.rms_norm    = nn.RMSNorm(d_head)
        self.out_proj    = nn.Linear(d_head, d_model, bias=False)

    def _gather_windows_batched(self, query_base, context_base, top_idx):
        """
        Cross-attention gather: queries from query_base, windows from context_base.

        win_start = clamp(min_idx - W//2, 0, S-W)  -- en GPU
        torch.gather para queries y windows          -- batched, sin loop Python
        """
        B, S, D = query_base.shape
        W       = self.window_size

        # Queries extraídas de query_base (x_norm — input raw)
        idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, D)
        queries = torch.gather(query_base, 1, idx_exp)

        # Ventanas (K/V) extraídas de context_base (mamba_out — info secuencial)
        # V7 FIX: idx_min usaba el MÍNIMO de los índices seleccionados como ancla.
        # Bug: si top_idx = [10, 50, 900], win_start ≈ 10 → el token en pos 900
        # atiende al contexto de posición [10:74] — completamente irrelevante.
        # Fix: usar la posición MEDIA de los top-K tokens como ancla. La ventana
        # se centra en el “centro de gravedad” de la complejidad del batch.
        # Para top_idx muy dispersos, el usuario debería usar per-query windows
        # (requiere cambio de kernel); la media es la mejor aproximación con un
        # solo tensor de ventana compartida.
        idx_anchor = top_idx.float().mean(dim=-1).long()                 # [B] GPU
        win_start  = (idx_anchor - W // 2).clamp(min=0)
        win_start  = (win_start + W).clamp(max=S) - W                    # evita doble clamp
        w_indices = win_start.unsqueeze(1).long() + \
                    torch.arange(W, device=context_base.device).unsqueeze(0) # [B, W]
        w_indices = w_indices.clamp(0, S - 1)
        w_idx_exp = w_indices.unsqueeze(-1).expand(-1, -1, D)
        windows   = torch.gather(context_base, 1, w_idx_exp)

        return queries, windows, idx_exp

    def forward(self, query_base, context_base=None, importance=None, pre_idx=None):
        """
        Cross-attention SLR: queries from query_base, K/V from context_base.

        query_base:   [B, S, D]  — raw input (x_norm), provides independent gradient path
        context_base: [B, S, D]  — enriched context (mamba_out). If None, uses query_base.
        importance:   [B, S]     (None -> usa norma de query_base)
        pre_idx:      [B, K]     (indices SGR pre-computados, opcional)
        Returns:      (slr_out [B, S, D], top_idx [B, K])
        """
        if context_base is None:
            context_base = query_base
        B, S, D = query_base.shape

        if importance is None:
            imp        = query_base.detach().norm(dim=-1)
            importance = imp / (imp.mean(dim=-1, keepdim=True) + 1e-6)

        top_idx, _ = (None, None)
        if pre_idx is not None:
            top_idx = pre_idx
        else:
            top_idx, _ = self.sgr(importance, S)

        # Gather batched (zero sync) — cross-attention: Q from query_base, KV from context_base
        queries, windows, idx_exp = self._gather_windows_batched(query_base, context_base, top_idx)

        # Proyecciones fusionadas: 2 matmuls sobre [B,K,D] y [B,W,D]
        Q1, Q2, K1, K2, V = self.proj(queries, windows)

        # Flash-Differential Attention V5 — custom_op (compile-safe, no graph breaks)
        attn_out = _flash_diff_slr_op(Q1, Q2, K1, K2, V, self.lam_logit)
        attn_out = self.rms_norm(attn_out.to(context_base.dtype))

        # Proyectar + scatter_add — sin merge_gate (el Router es el único controlador).
        # out_proj ya es un linear aprendido que controla magnitud por sí solo.
        projected = self.out_proj(attn_out).to(context_base.dtype)  # [B, K, D]
        delta     = torch.zeros_like(context_base)
        delta.scatter_add_(1, idx_exp, projected)

        return context_base + delta, top_idx


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Test rapido
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    DEVICE = "cuda"
    assert torch.cuda.is_available(), "Necesita GPU CUDA"
    B, S, D, dh = 2, 512, 256, 32
    print("=" * 70)
    print("  SLRDifferentialModule V4 — Flash-Differential Attention")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    model = SLRDifferentialModule(d_model=D, d_head=dh, window_size=64).to(DEVICE)
    x   = torch.randn(B, S, D, device=DEVICE, requires_grad=True)
    # Cross-attention: query_base and context_base can differ
    ctx = torch.randn(B, S, D, device=DEVICE, requires_grad=True)
    imp = torch.rand(B, S, device=DEVICE)
    imp[0, 50:100] *= 5.0

    # ── Warmup del kernel Triton (autotune + compilación JIT) ─────────────────
    for _ in range(3):
        x2 = x.detach().requires_grad_(True)
        c2 = ctx.detach().requires_grad_(True)
        out, idx = model(x2, context_base=c2, importance=imp)
        out.mean().backward()
    torch.cuda.synchronize()

    # ── Benchmark forward ─────────────────────────────────────────────────────
    N_RUNS = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        x2 = x.detach().requires_grad_(True)
        c2 = ctx.detach().requires_grad_(True)
        out, idx = model(x2, context_base=c2, importance=imp)
    torch.cuda.synchronize()
    t_fwd = (time.perf_counter() - t0) / N_RUNS * 1000

    # ── Benchmark forward + backward ─────────────────────────────────────────
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        x2 = x.detach().requires_grad_(True)
        c2 = ctx.detach().requires_grad_(True)
        out, idx = model(x2, context_base=c2, importance=imp)
        out.mean().backward()
    torch.cuda.synchronize()
    t_fwdbwd = (time.perf_counter() - t0) / N_RUNS * 1000
    t_bwd = t_fwdbwd - t_fwd

    K = idx.shape[1]
    print(f"\n  B={B}  S={S}  D={D}  d_head={dh}  K={K} ({100*K/S:.0f}%)")
    print(f"  Forward:   {t_fwd:.2f} ms")
    print(f"  Backward:  {t_bwd:.2f} ms")
    print(f"  F+B total: {t_fwdbwd:.2f} ms")
    print(f"  Output:    {tuple(out.shape)}  finite={out.isfinite().all().item()}")

    # ── Grad check ───────────────────────────────────────────────────────────
    print("\n  Gradientes tras backward:")
    all_ok = True
    for name, p in model.named_parameters():
        gn = p.grad.norm().item() if p.grad is not None else 0.0
        ok = p.grad is not None and gn > 1e-12
        status = "✓ ALIVE" if ok else "✗ DEAD "
        print(f"    {status}  {name:<28}  grad_norm={gn:.2e}")
        if not ok:
            all_ok = False
    print(f"\n  lam actual: {model.lam_logit.sigmoid().item():.4f}")

    # ── Corrección numérica vs PyTorch baseline ───────────────────────────────
    print("\n  Verificación numérica vs PyTorch:")
    with torch.no_grad():
        B2, K2, W2 = 2, 32, 48
        q = torch.randn(B2, K2, dh, device=DEVICE)
        k = torch.randn(B2, W2, dh, device=DEVICE)
        v = torch.randn(B2, W2, dh, device=DEVICE)
        lam_val = 0.3
        sc  = 1.0 / math.sqrt(dh)
        # Reference: PyTorch bmm
        a1  = torch.softmax((torch.bmm(q, k.transpose(-1,-2))) * sc, dim=-1)
        a2  = torch.softmax((torch.bmm(q, k.transpose(-1,-2))) * sc, dim=-1)
        ref = torch.bmm(a1, v) - lam_val * torch.bmm(a2, v)
        # Flash kernel (BF16 Tensor Core path — tolerance 5e-3 for BF16 mantissa)
        flash = _flash_diff_attn_fwd(q, q, k, k, v, lam_val)
        err = (flash - ref).abs().max().item()
        print(f"    Max error (batched Flash vs PyTorch): {err:.2e}  "
              f"{'✓ OK' if err < 5e-3 else '✗ ERROR'}")

        # 1D wrapper (landmark_native)
        ref1d = (a1[0] @ v[0]) - lam_val * (a2[0] @ v[0])
        t1d   = diff_attn_v2_triton(q[0], q[0], k[0], k[0], v[0], lam_val)
        err1d = (t1d - ref1d).abs().max().item()
        print(f"    Max error (1D wrapper vs PyTorch):    {err1d:.2e}  "
              f"{'✓ OK' if err1d < 2e-3 else '✗ ERROR'}")

    print(f"\n  {'[SUCCESS] Todos los gradientes vivos' if all_ok else '[FAIL] Gradientes muertos detectados'}")
    print("=" * 70)
