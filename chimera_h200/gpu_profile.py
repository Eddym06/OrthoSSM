"""
gpu_profile.py — Hardware-Adaptive JIT Configuration
=====================================================
Detecta el GPU en tiempo real y genera configuraciones óptimas para:
  - Triton autotune (BLOCK sizes, num_warps, num_stages por clase de GPU)
  - torch.compile (mode, fullgraph, dynamic)
  - CUDA Graphs (flag seguro + ring_size del bus en decode)
  - chunk_size por defecto para prefill chunked

Cómo funciona:
  1. get_gpu_profile() detecta el GPU con torch.cuda.get_device_properties()
  2. Mapea (sm_major, sm_minor, n_sms, vram_gb) → GPUClass
  3. Para cada GPUClass, define el espacio de búsqueda Triton y los parámetros
     de torch.compile específicos a esa microarquitectura
  4. Se llama una vez en import; resultado cacheado en _PROFILE_CACHE

GPU Classes soportadas:
  CPU          → no GPU (fallback seguro)
  LAPTOP_ADA   → RTX 4050/4060/4070 Laptop  (SM=8.9, ≤28 SMs, ≤8GB GDDR6)
  DESKTOP_ADA  → RTX 4070Ti/4080/4090        (SM=8.9, ≥36 SMs, ≥12GB GDDR6X)
  AMPERE_MID   → RTX 3060/3070/3080/3090     (SM=8.6-8.7, ≤82 SMs)
  AMPERE_HPC   → A100 40/80GB                (SM=8.0, 108 SMs, HBM2e)
  HOPPER       → H100/H200 SXM/NVL           (SM=9.0, 132 SMs, HBM3e, TMA, FP8)
  BLACKWELL    → B100/B200/GB200             (SM=10.0, ≥192 SMs, HBM3e)

Uso:
    from gpu_profile import get_gpu_profile, get_triton_configs_flash
    from gpu_profile import get_torch_compile_kwargs

    profile = get_gpu_profile()
    print(profile)  # "GPUProfile(hopper) | H200 SXM | SM=9.0 | ..."

    # Autotune adaptativo en Triton:
    @triton.autotune(configs=get_triton_configs_flash(), key=['K_size','W_size','d_head'])
    @triton.jit
    def my_kernel(...): ...

    # torch.compile adaptativo:
    model = torch.compile(model, **get_torch_compile_kwargs(mode='train'))
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import List, Optional

try:
    import torch
    import triton
    _HAS_CUDA = torch.cuda.is_available()
    _HAS_TRITON = True
except ImportError:
    _HAS_CUDA = False
    _HAS_TRITON = False


# ─────────────────────────────────────────────────────────────────────────────
# GPU Class enum
# ─────────────────────────────────────────────────────────────────────────────

class GPUClass(enum.Enum):
    CPU          = "cpu"
    LAPTOP_ADA   = "laptop_ada"    # RTX 40xx Mobile
    DESKTOP_ADA  = "desktop_ada"   # RTX 4080/4090 Desktop
    AMPERE_MID   = "ampere_mid"    # RTX 30xx consumer
    AMPERE_HPC   = "ampere_hpc"    # A100 (HBM2e, NVLink)
    HOPPER       = "hopper"        # H100 / H200 (HBM3e, TMA, FP8)
    BLACKWELL    = "blackwell"     # B200 / GB200 (HBM3e next-gen)


# ─────────────────────────────────────────────────────────────────────────────
# GPUProfile dataclass — UN PERFIL POR GPU DETECTADO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GPUProfile:
    """
    Contiene todas las configuraciones óptimas para un GPU específico.
    Se genera una vez y se usa en todo el código Triton y torch.compile.
    """
    # ── Identificación hardware ───────────────────────────────────────────────
    gpu_class:     GPUClass
    name:          str
    sm_major:      int
    sm_minor:      int
    n_sms:         int
    vram_gb:       float
    l2_mb:         float
    bw_gbps_est:   float        # estimado (torch no expone BW directamente)

    # ── Triton: kernel flash diff-attn (SLR) ──────────────────────────────────
    # BLOCK_K: queries por tile; BLOCK_W: keys/vals por tile
    # num_stages: profundidad de pipelining software (+ BW → + stages)
    # num_warps: paralelismo CTA (+ SMs → más warps = mejor occupancy)
    triton_stages_flash:   int   # num_stages para flash diff-attn
    triton_stages_ema:     int   # num_stages para Clenshaw EMA kernel
    triton_warps_base:     int   # num_warps base (se dobla en configs grandes)
    triton_block_k_max:    int   # BLOCK_K máximo (queries/tile)
    triton_block_w_max:    int   # BLOCK_W máximo (keys-vals/tile)
    triton_block_s_ema:    int   # BLOCK_S para EMA Clenshaw

    @property
    def is_ampere_or_better(self) -> bool:
        return self.sm_major >= 8

    @property
    def is_hopper_or_better(self) -> bool:
        return self.sm_major >= 9



    # ── torch.compile ─────────────────────────────────────────────────────────
    # max-autotune: benchmark todos los kernels en la primera ejecución → más lento
    # reduce-overhead: 2× más rápido para compilar (bueno para GPUs pequeños)
    # max-autotune-no-cudagraphs: para cuando gestionamos graphs manualmente
    compile_mode_train:  str
    compile_mode_infer:  str
    compile_fullgraph:   bool   # True en Hopper+ → graph completo sin breaks

    # ── CUDA Graphs ───────────────────────────────────────────────────────────
    cuda_graph_safe:     bool   # si el step() puede capturarse en CUDA graph
    ring_size:           int    # tamaño ring buffer bus en decode (fixed shape)
    graph_warmup_iters:  int    # iteraciones warmup antes de capture

    # ── Miscelánea ────────────────────────────────────────────────────────────
    use_fp8_fwd:         bool   # FP8 e4m3 en forward (Hopper+ solamente)
    use_tma:             bool   # TMA (Tensor Memory Accelerator) en kernels Triton (SM≥9.0)
    spsa_fused:          bool   # SPSA dual-scan Triton kernel (SM≥9.0, ≥128KB SRAM por SM)
    chunk_size_default:  int    # chunk_size para chunked_prefill

    def __str__(self) -> str:
        return (
            f"GPUProfile({self.gpu_class.value}) | {self.name} | "
            f"SM={self.sm_major}.{self.sm_minor} | {self.n_sms} SMs | "
            f"VRAM={self.vram_gb:.1f}GB | BW~{self.bw_gbps_est:.0f}GB/s | "
            f"compile[train]={self.compile_mode_train} | "
            f"cuda_graph={self.cuda_graph_safe}"
        )

    def summary(self) -> str:
        """Resumen multilínea para logging al inicio del training."""
        lines = [
            f"{'─'*60}",
            f"  GPU Hardware Profile",
            f"{'─'*60}",
            f"  Dispositivo : {self.name}",
            f"  Clase GPU   : {self.gpu_class.value}",
            f"  SM version  : {self.sm_major}.{self.sm_minor}  |  SMs: {self.n_sms}",
            f"  VRAM        : {self.vram_gb:.1f} GB",
            f"  L2 cache    : {self.l2_mb:.1f} MB (estimado)",
            f"  BW estimado : {self.bw_gbps_est:.0f} GB/s",
            f"",
            f"  Triton flash: stages={self.triton_stages_flash}  "
            f"warps={self.triton_warps_base}  "
            f"BLOCK_K≤{self.triton_block_k_max}  BLOCK_W≤{self.triton_block_w_max}",
            f"  Triton EMA  : stages={self.triton_stages_ema}  BLOCK_S={self.triton_block_s_ema}",
            f"  compile(train)  : mode='{self.compile_mode_train}'  "
            f"fullgraph={self.compile_fullgraph}",
            f"  compile(infer)  : mode='{self.compile_mode_infer}'",
            f"  CUDA Graphs : {'✓ habilitado' if self.cuda_graph_safe else '✗ deshabilitado'}  "
            f"ring_size={self.ring_size}",
            f"  FP8 forward : {'✓ disponible' if self.use_fp8_fwd else '✗ no disponible (SM<9.0)'}",
            f"  TMA kernels : {'✓ habilitado (block_ptr async DMA)' if self.use_tma else '✗ deshabilitado (SIMT loads)'}",
            f"  SPSA fused  : {'✓ dual-scan Triton (SRAM registers)' if self.spsa_fused else '✗ Python-level (3× mamba2 forward)'}",
            f"  chunk_size  : {self.chunk_size_default}",
            f"{'─'*60}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tabla de parámetros por GPUClass
# Rationale por clase:
#   LAPTOP_ADA  : 24MB L2 → tiles pequeños; 192 GB/s → stages=2-3; SMs=20
#   DESKTOP_ADA : 72MB L2 (RTX4090); ~1008 GB/s GDDR6X; stages=3; 128 SMs
#   AMPERE_MID  : 6MB L2 (RTX3090 — mucho menor que Ada); stages=3; 82 SMs
#   AMPERE_HPC  : 40MB L2; HBM2e ~2000 GB/s; stages=4; 108 SMs; fullgraph OK
#   HOPPER      : 50MB L2; HBM3e 3350-4800 GB/s; TMA; FP8; stages=5; 132 SMs
#   BLACKWELL   : 96MB L2; HBM3e ~8000 GB/s; stages=6; 192+ SMs; biggest tiles
# ─────────────────────────────────────────────────────────────────────────────

_TABLE: dict[GPUClass, dict] = {
    GPUClass.CPU: dict(
        triton_stages_flash=1, triton_stages_ema=1,
        triton_warps_base=1,
        triton_block_k_max=32, triton_block_w_max=64, triton_block_s_ema=64,
        compile_mode_train='default', compile_mode_infer='default',
        compile_fullgraph=False,
        cuda_graph_safe=False, ring_size=8, graph_warmup_iters=2,
        use_fp8_fwd=False, use_tma=False, spsa_fused=False, chunk_size_default=512,
    ),
    GPUClass.LAPTOP_ADA: dict(
        # RTX 4050/4060 Laptop: L2=24MB, BW=192GB/s, SMs=20-28
        # Con KW=64×64=4096 → A1 matrix (32KB) cabe en L2 → cuBLAS gana (fallback ✓)
        # Para KW>4096: Flash Triton úil, stages moderados
        # Triton 3.4: mejor async mem copy pipeline → stages_flash 2→3 es seguro
        triton_stages_flash=3, triton_stages_ema=3,
        triton_warps_base=4,
        triton_block_k_max=32, triton_block_w_max=64, triton_block_s_ema=128,
        compile_mode_train='reduce-overhead', compile_mode_infer='reduce-overhead',
        compile_fullgraph=False,
        cuda_graph_safe=True, ring_size=16, graph_warmup_iters=3,
        use_fp8_fwd=False, use_tma=False, spsa_fused=False, chunk_size_default=2048,
    ),
    GPUClass.DESKTOP_ADA: dict(
        # RTX 4080/4090: L2=64-72MB, BW=1008GB/s, SMs=76-128
        # Tiles grandes rentables; max-autotune encuentra mejores configs
        triton_stages_flash=3, triton_stages_ema=4,
        triton_warps_base=4,
        triton_block_k_max=64, triton_block_w_max=128, triton_block_s_ema=256,
        compile_mode_train='max-autotune', compile_mode_infer='reduce-overhead',
        compile_fullgraph=False,
        cuda_graph_safe=True, ring_size=24, graph_warmup_iters=3,
        use_fp8_fwd=False, use_tma=False, spsa_fused=False, chunk_size_default=4096,
    ),
    GPUClass.AMPERE_MID: dict(
        # RTX 3060-3090: L2=3-6MB (PEQUEÑO vs Ada), BW=936GB/s, SMs=28-82
        # L2 pequeño: tiles medianos para no thrash; BW alta → stages=3
        triton_stages_flash=3, triton_stages_ema=3,
        triton_warps_base=4,
        triton_block_k_max=64, triton_block_w_max=128, triton_block_s_ema=256,
        compile_mode_train='max-autotune', compile_mode_infer='reduce-overhead',
        compile_fullgraph=False,
        cuda_graph_safe=True, ring_size=16, graph_warmup_iters=3,
        use_fp8_fwd=False, use_tma=False, spsa_fused=False, chunk_size_default=4096,
    ),
    GPUClass.AMPERE_HPC: dict(
        # A100 40/80GB: L2=40MB, HBM2e ~2000GB/s, 108 SMs
        # fullgraph=True seguro (sin breakpoints dinámicos)
        # stages=4: BW alta permite pipeline más profundo para esconder latencia HBM
        triton_stages_flash=4, triton_stages_ema=4,
        triton_warps_base=8,
        triton_block_k_max=128, triton_block_w_max=256, triton_block_s_ema=512,
        compile_mode_train='max-autotune', compile_mode_infer='reduce-overhead',
        compile_fullgraph=True,
        cuda_graph_safe=True, ring_size=32, graph_warmup_iters=5,
        use_fp8_fwd=False, use_tma=False, spsa_fused=False, chunk_size_default=8192,
    ),
    GPUClass.HOPPER: dict(
        # H100/H200 SXM: L2=50MB, HBM3(e) 3350-4800GB/s, 132 SMs
        # TMA (Tensor Memory Accelerator): async mem copies → stages=5 amplamente útil
        # WGMMA: FP16/BF16 matmuls más rápidos → tiles BLOCK_K=128 óptimos
        # FP8: e4m3/e5m2 disponible → 2× throughput en forward pass
        # max-autotune-no-cudagraphs en infer: gestionamos graphs manualmente
        triton_stages_flash=5, triton_stages_ema=5,
        triton_warps_base=8,
        triton_block_k_max=128, triton_block_w_max=256, triton_block_s_ema=512,
        compile_mode_train='max-autotune', compile_mode_infer='max-autotune-no-cudagraphs',
        compile_fullgraph=True,
        cuda_graph_safe=True, ring_size=32, graph_warmup_iters=5,
        # H200: TMA (SM=9.0) + FP8 e4m3fn → 2× throughput en matmuls Q@K
        # SPSA fused: dual SSM scan in SRAM registers (~18KB/CTA) — 227KB H200 SRAM
        # chunk_size 32768: H200 tiene 141GB VRAM → chunks enormes en prefill
        use_fp8_fwd=True, use_tma=True, spsa_fused=True, chunk_size_default=32768,
    ),
    GPUClass.BLACKWELL: dict(
        # B200/GB200: L2=96MB, HBM3e ~8000GB/s, 192+ SMs, NVLink 1.8TB/s
        # stages=6: BW extrema hace que pipeline de 6 etapas sea rentable
        # tiles gigantes: BLOCK_K=256 → matmul [256,d]×[d,256] satura Tensor Cores
        # ring_size=64: bus capture 64 tokens recientes en decode
        triton_stages_flash=6, triton_stages_ema=6,
        triton_warps_base=16,
        triton_block_k_max=256, triton_block_w_max=512, triton_block_s_ema=1024,
        compile_mode_train='max-autotune', compile_mode_infer='max-autotune-no-cudagraphs',
        compile_fullgraph=True,
        cuda_graph_safe=True, ring_size=64, graph_warmup_iters=5,
        # Blackwell: TMA-2 (SM=10.0) + FP8 e4m3fn → tiles 256×512 saturan WGMMA
        # SPSA fused: dual SSM scan in SRAM (~18KB/CTA) — Blackwell 256KB SRAM
        use_fp8_fwd=True, use_tma=True, spsa_fused=True, chunk_size_default=65536,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Detección y singleton
# ─────────────────────────────────────────────────────────────────────────────

_PROFILE_CACHE: Optional[GPUProfile] = None


def detect_gpu(device: int = 0) -> GPUProfile:
    """
    Detecta el GPU en `device` y retorna un GPUProfile completo.

    La clasificación usa:
      - sm_major, sm_minor: compute capability (determinístico)
      - n_sms: diferencia laptop vs desktop dentro de la misma generación
      - vram_gb: diferencia A100-40 vs A100-80, H100 vs H200
    """
    if not _HAS_CUDA:
        return GPUProfile(
            gpu_class=GPUClass.CPU, name="CPU (no CUDA)",
            sm_major=0, sm_minor=0, n_sms=0,
            vram_gb=0.0, l2_mb=0.0, bw_gbps_est=0.0,
            **_TABLE[GPUClass.CPU],
        )

    try:
        props = torch.cuda.get_device_properties(device)
    except Exception:
        return GPUProfile(
            gpu_class=GPUClass.CPU, name="Unknown GPU",
            sm_major=0, sm_minor=0, n_sms=0,
            vram_gb=0.0, l2_mb=0.0, bw_gbps_est=0.0,
            **_TABLE[GPUClass.CPU],
        )

    sm_major = props.major
    sm_minor = props.minor
    n_sms    = props.multi_processor_count
    vram_gb  = props.total_memory / 1e9
    l2_mb    = getattr(props, 'l2_cache_size', 0) / 1e6
    name     = props.name

    # ── Clasificación por compute capability ─────────────────────────────────
    if sm_major >= 10:
        # Blackwell: B100, B200, GB200 (SM=10.0)
        gpu_class = GPUClass.BLACKWELL
        bw_gbps   = 8000.0

    elif sm_major == 9:
        # Hopper: H100, H200 (SM=9.0)
        gpu_class = GPUClass.HOPPER
        # H200 tiene más VRAM que H100 SXM (141GB vs 80GB)
        bw_gbps   = 4800.0 if vram_gb > 130 else 3350.0

    elif sm_major == 8 and sm_minor == 0:
        # Ampere HPC: A100 (SM=8.0 — solo la versión datacenter)
        gpu_class = GPUClass.AMPERE_HPC
        bw_gbps   = 2000.0 if vram_gb > 60 else 1555.0

    elif sm_major == 8 and sm_minor == 9:
        # Ada Lovelace: RTX 4050-4090 (SM=8.9)
        # Laptop vs Desktop: umbral de SMs (RTX 4070 Ti = 60 SMs, RTX 4060 Laptop = 24 SMs)
        if n_sms <= 28:
            gpu_class = GPUClass.LAPTOP_ADA
            bw_gbps   = 192.0   # GDDR6 — 6GB laptop
        else:
            gpu_class = GPUClass.DESKTOP_ADA
            bw_gbps   = 1008.0  # GDDR6X — RTX 4090 peak

    elif sm_major == 8 and sm_minor in (6, 7):
        # Ampere consumer: RTX 3060-3090 (SM=8.6), Jetson Orin (SM=8.7)
        gpu_class = GPUClass.AMPERE_MID
        bw_gbps   = 936.0 if n_sms >= 60 else 320.0  # 3090 vs 3060

    else:
        # Turing (7.5), Volta (7.0), Pascal (6.x) → conservative consumer profile
        gpu_class = GPUClass.AMPERE_MID
        bw_gbps   = 448.0

    tbl = _TABLE[gpu_class]
    return GPUProfile(
        gpu_class=gpu_class, name=name,
        sm_major=sm_major, sm_minor=sm_minor,
        n_sms=n_sms, vram_gb=vram_gb,
        l2_mb=l2_mb, bw_gbps_est=bw_gbps,
        **tbl,
    )


def get_gpu_profile(device: int = 0) -> GPUProfile:
    """
    Singleton cacheado — detección corre solo una vez por proceso.
    Thread-safe bajo GIL de CPython.

    El singleton puede forzarse a re-detectar con:
        import gpu_profile
        gpu_profile._PROFILE_CACHE = None
        profile = gpu_profile.get_gpu_profile()
    """
    global _PROFILE_CACHE
    if _PROFILE_CACHE is None:
        _PROFILE_CACHE = detect_gpu(device)
    return _PROFILE_CACHE


# ─────────────────────────────────────────────────────────────────────────────
# Generadores de configuración Triton
# ─────────────────────────────────────────────────────────────────────────────

def get_triton_configs_flash(profile: Optional[GPUProfile] = None) -> list:
    """
    Genera la lista de triton.Config para el kernel flash diff-attn (SLR).

    Estrategia de cobertura:
      - Siempre incluye configs pequeños (BLOCK_K=16, BLOCK_W=32) → seguro en
        cualquier GPU con pocas SRAM. Son los configs de fallback.
      - Para GPUs con BW alta: añade tiles más grandes + más stages. Triton
        seleccionará automáticamente el mejor mediante benchmarking real.
      - BLOCK_K escalado por d_model: con d_model=256, K=32→OK; d=2048, K=128→OK.

    Los configs se generan en función del perfil GPU, NO hardcodeados.
    Triton realiza el autotune con benchmarks reales en el hardware destino.
    """
    if not _HAS_TRITON:
        return []

    p   = profile or get_gpu_profile()
    s   = p.triton_stages_flash
    w   = p.triton_warps_base
    bk  = p.triton_block_k_max
    bww = p.triton_block_w_max

    configs = []

    # ── Tier 1: configs base (SIEMPRE incluidos — seguros en cualquier GPU) ──
    configs += [
        triton.Config({'BLOCK_K': 16, 'BLOCK_W':  32}, num_warps=2,       num_stages=max(1, s-2)),
        triton.Config({'BLOCK_K': 16, 'BLOCK_W':  64}, num_warps=w,       num_stages=max(1, s-1)),
        triton.Config({'BLOCK_K': 32, 'BLOCK_W':  64}, num_warps=w,       num_stages=s),
        triton.Config({'BLOCK_K': 32, 'BLOCK_W': 128}, num_warps=w,       num_stages=s),
    ]

    # ── Tier 2: tiles medianos (GPUs con BLOCK_K_max ≥ 64) ───────────────────
    if bk >= 64:
        configs += [
            triton.Config({'BLOCK_K':  64, 'BLOCK_W':  64}, num_warps=w,     num_stages=s),
            triton.Config({'BLOCK_K':  64, 'BLOCK_W': 128}, num_warps=w * 2, num_stages=s),
        ]
        if bww >= 128:
            configs.append(
                triton.Config({'BLOCK_K': 64, 'BLOCK_W': bww}, num_warps=w * 2, num_stages=s),
            )

    # ── Tier 3: tiles grandes (A100, H100/H200, Blackwell) ───────────────────
    if bk >= 128:
        configs += [
            triton.Config({'BLOCK_K': 128, 'BLOCK_W': 128}, num_warps=w * 2, num_stages=s),
            triton.Config({'BLOCK_K': 128, 'BLOCK_W': 256}, num_warps=w * 2, num_stages=max(1, s - 1)),
        ]
        if bww >= 256:
            configs.append(
                triton.Config({'BLOCK_K': 128, 'BLOCK_W': bww}, num_warps=w * 2, num_stages=s),
            )

    # ── Tier 4: tiles máximos (H200/Blackwell únicamente) ────────────────────
    if bk >= 256:
        configs += [
            triton.Config({'BLOCK_K': 256, 'BLOCK_W': 256}, num_warps=w * 4, num_stages=s),
            triton.Config({'BLOCK_K': 256, 'BLOCK_W': 512}, num_warps=w * 4, num_stages=max(1, s - 1)),
        ]

    return configs


def get_triton_configs_ema(profile: Optional[GPUProfile] = None) -> list:
    """
    Genera configs para el kernel EMA Clenshaw (sdpc_kernel.py).
    BLOCK_S: tokens por tile; BLOCK_HD: dimensión head por tile.
    """
    if not _HAS_TRITON:
        return []

    p  = profile or get_gpu_profile()
    s  = p.triton_stages_ema
    w  = p.triton_warps_base
    bs = p.triton_block_s_ema

    configs = [
        triton.Config({'BLOCK_S':  64, 'BLOCK_HD': 32}, num_warps=max(2, w // 2), num_stages=max(1, s - 1)),
        triton.Config({'BLOCK_S': 128, 'BLOCK_HD': 32}, num_warps=w,              num_stages=s),
        triton.Config({'BLOCK_S': bs,  'BLOCK_HD': 64}, num_warps=w,              num_stages=s),
    ]
    if bs >= 256:
        configs.append(
            triton.Config({'BLOCK_S': 256, 'BLOCK_HD': 64}, num_warps=w * 2, num_stages=s),
        )
    if bs >= 512:
        configs.append(
            triton.Config({'BLOCK_S': 512, 'BLOCK_HD': 64}, num_warps=w * 2, num_stages=max(1, s - 1)),
        )
    if bs >= 1024:
        configs.append(
            triton.Config({'BLOCK_S': 1024, 'BLOCK_HD': 64}, num_warps=w * 4, num_stages=max(1, s - 2)),
        )
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# torch.compile kwargs
# ─────────────────────────────────────────────────────────────────────────────

def get_torch_compile_kwargs(
    profile: Optional[GPUProfile] = None,
    mode: str = 'train',
) -> dict:
    """
    Retorna kwargs para torch.compile() optimizados para el GPU detectado.

    Args:
        profile: GPUProfile; si None, usa get_gpu_profile()
        mode:    'train' | 'infer'

    Returns:
        Dict pasable directamente a torch.compile(**kwargs).

    Ejemplo:
        model = torch.compile(model, **get_torch_compile_kwargs(mode='train'))

    Comportamiento por GPU:
      LAPTOP_ADA   → reduce-overhead (GPU débil; max-autotune demasiado lento)
      DESKTOP_ADA  → max-autotune train / reduce-overhead infer
      AMPERE_HPC   → max-autotune + fullgraph=True
      HOPPER       → max-autotune + fullgraph=True; infer=max-autotune-no-cudagraphs
                     (CUDA graphs se gestionan manualmente en make_cuda_graph_step)
      BLACKWELL    → igual que Hopper
    """
    p = profile or get_gpu_profile()
    m = p.compile_mode_train if mode == 'train' else p.compile_mode_infer

    return dict(
        mode=m,
        fullgraph=p.compile_fullgraph,
        dynamic=False,          # formas estáticas → mejor optimización torch.compile
    )


# ─────────────────────────────────────────────────────────────────────────────
# Script de diagnóstico standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    profile = get_gpu_profile()
    print(profile.summary())
    print(f"\n  Configs flash diff-attn : {len(get_triton_configs_flash(profile))} configs")
    print(f"  Configs EMA Clenshaw    : {len(get_triton_configs_ema(profile))} configs")
    print(f"\n  torch.compile train : {get_torch_compile_kwargs(profile, mode='train')}")
    print(f"  torch.compile infer : {get_torch_compile_kwargs(profile, mode='infer')}")
