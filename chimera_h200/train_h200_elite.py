"""
train_h200_elite.py — CHIMERA Elite Trainer para H200 140 GB
=============================================================

Objetivo: entrenar CHIMERA-130M en < 2 horas en una sola H200.

VELOCIDAD (cada capa elimina un cuello de botella diferente):
  1. Dual-Stream Data Pipeline   — Stream 0 = cómputo, Stream 1 = prefetch H2D.
                                   Overlap total de transferencia con forward+backward.
  2. CUDA Graph F+B completo     — Cero Python dispatch overhead en el hot path.
                                   zero_grad fuera del grafo; backward acumula in-place.
  3. NS5 con buffers pre-alloc   — Newton-Schulz 5 iters sin allocations en cada step.
                                   Usa torch.mm(out=buf) para evitar allocs intermedias.
  4. Triton NS5 fusionado        — Para matrices min(M,N) ∈ [16,64]: 1 kernel launch
                                   vs 15 separados. Mantiene A,B en registros GPU.
  5. BF16 + TF32 residual        — Tensor Cores BF16 everywhere, TF32 para FP32 residual.
  6. Async checkpoint            — torch.save en background thread. Sin blocking.

ESTABILIDAD (todo en GPU, sin sync CPU hasta logging):
  7. GradHealthMonitor           — _foreach_is_finite sobre todos los grads. O(1) CPU sync
                                   solo cuando hay NaN (infrecuente). Skip step automático.
  8. LossSpikeDetector           — EMA + varianza mantenidos como GPU tensors.
                                   Spike detectado como op tensorial: (loss > ema+3σ).
  9. RouterEntropyWatchdog       — Monitorea H_bits del router en cada step.
                                   Si colapsa < 1.5 bits por 10 steps → entropy_coeff ×10
                                   auto (100 steps de recuperación).
 10. TTTGradSupervisor           — Aplica _pending_ttt_grad con su propio clip presupuesto,
                                   aislado del clip principal. Evita que TTT desestabilice
                                   el scan de Mamba2.
 11. ChimeraAnnealer             — Hyperparams Chimera-específicos:
                                   • slr_threshold: 0.75→0.50 en primeros 5k steps
                                   • z_loss_coeff:  1e-3→1e-4 post-warmup
                                   • Bus cache reset cada 2k steps (evita drift acumulado)

Uso mínimo:
    python train_h200_elite.py --data_dir /data/tokens

Uso completo H200 (< 2h para 3B tokens @ 130M):
    python train_h200_elite.py \\
        --data_dir /data/tokens --model 125M \\
        --batch 32 --seq_len 2048 --grad_accum 1 \\
        --lr 3e-4 --muon_lr 0.02 \\
        --total_tokens 3e9 \\
        --hbm_dataset --cuda_graphs \\
        --ckpt_dir ./ckpt_elite --log_every 10

Benchmark sin datos:
    python train_h200_elite.py --benchmark --model 125M --batch 32
"""

from __future__ import annotations

import argparse, json, math, os, sys, threading, time
from pathlib import Path
from queue import Queue, Empty
from dataclasses import asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from chimera_config import ChimeraConfig
from chimera_lm     import ChimeraLM, build_chimera_125M, build_chimera_350M
from gpu_profile    import get_torch_compile_kwargs as _get_compile_kwargs

# ─────────────────────────────────────────────────────────────────────────────
# Flags globales H200 — máximo rendimiento en HBM3e + SM90 Tensor Cores
# ─────────────────────────────────────────────────────────────────────────────
torch.set_float32_matmul_precision('high')                              # TF32 activado para residuales
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True  # HMMA BF16 acumulación
torch.backends.cudnn.benchmark = True                                   # autotuning de kernels cuDNN
# H200: FlashAttention-3 via sdp_kernel (usa TMA + WGMMA → 2-3× FA2 en SLR/Archive)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)                              # deshabilitar fallback lento


# ─────────────────────────────────────────────────────────────────────────────
# Triton NS5 fusionado (un kernel launch para todo Newton-Schulz)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception:
    _TRITON_OK = False

if _TRITON_OK:
    @triton.jit
    def _ns5_fused_kernel(
        X_ptr, out_ptr,
        norm: tl.float32,       # ‖X‖₂ pre-computado
        M: tl.constexpr,        # dim pequeña (≤ 64), ya orientado M ≤ N
        N: tl.constexpr,        # dim grande
    ):
        """
        Calcula 5 iteraciones de Newton-Schulz enteramente en registros GPU.
        Grid = (1,). Un único programa procesa la matriz entera.
        X_ptr: [M, N] BF16 (row major, stride N).
        """
        a: tl.constexpr = 3.4445
        b: tl.constexpr = -4.7750
        c: tl.constexpr = 2.0315

        row = tl.arange(0, M)   # [M]
        col = tl.arange(0, N)   # [N]

        # Cargar X y normalizar
        X = tl.load(X_ptr + row[:, None] * N + col[None, :]).to(tl.float32)
        X = X / (norm + 1e-7)

        # 5 iteraciones NS (todo en registros — sin writes a HBM)
        for _ in tl.static_range(5):
            A  = tl.dot(X, tl.trans(X))            # [M, N] @ [N, M] = [M, M]
            B  = b * A + c * tl.dot(A, A)          # [M, M]
            X  = a * X + tl.dot(B, X)              # [M, N]

        tl.store(out_ptr + row[:, None] * N + col[None, :], X.to(tl.bfloat16))


def _ns5_triton(G: torch.Tensor) -> torch.Tensor | None:
    """
    Intenta NS5 con Triton para matrices min(M,N) ∈ [16,64] y N ≤ 2048.
    Retorna None si el shape no es elegible (caller usa fallback PyTorch).
    """
    if not _TRITON_OK:
        return None
    m, n = G.shape
    transposed = False
    if m > n:
        G = G.T.contiguous()
        m, n = n, m
        transposed = True
    if not (16 <= m <= 64 and n <= 2048 and n % 16 == 0 and m % 16 == 0):
        if transposed:
            G = G.T.contiguous()
        return None

    G_bf16 = G.bfloat16().contiguous()
    out    = torch.empty_like(G_bf16)
    norm   = float(G_bf16.float().norm().item())
    try:
        _ns5_fused_kernel[(1,)](G_bf16, out, norm, M=m, N=n)
    except Exception:
        if transposed:
            G = G.T.contiguous()
        return None

    return out.T if transposed else out


# ─────────────────────────────────────────────────────────────────────────────
# Newton-Schulz PyTorch con buffers pre-allocated (fallback + matrices grandes)
# ─────────────────────────────────────────────────────────────────────────────

class NS5Buffers:
    """Caché de buffers pre-allocados para NS5. Sin allocations en el hot path."""

    def __init__(
        self,
        shard_paths: list[str],
        seq_len: int,
        batch_size: int,
        device: torch.device,
        hbm: bool = False,
        n_prefetch: int = 2,
        token_dtype: str = "uint16",
    ):
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.device     = device
        self.hbm        = hbm
        self._step      = 0
        self._rng       = torch.Generator(device=self.device)
        self._rng.manual_seed(0)
        self._token_dtype = np.dtype(token_dtype)  # uint16 o uint32
        self._ptr       = 0

        if hbm:
            # Cargar todo en GPU — zero latency
            shards = []
            for p in shard_paths:
                arr = np.fromfile(p, dtype=self._token_dtype).astype(np.int64)
                shards.append(torch.from_numpy(arr).to(device))
            self._data_gpu  = torch.cat(shards)
            self._n_toks    = self._data_gpu.numel()
            self._stream_d  = None
        else:
            # mmap → pinned CPU → async H2D en stream_data
            arrays = [np.memmap(p, dtype=self._token_dtype, mode='r') for p in shard_paths]
            self._mmap      = np.concatenate(arrays)
            self._n_toks    = len(self._mmap)
            # El stream de datos solo existe en CUDA; en CPU el copy es síncrono
            self._stream_d  = torch.cuda.Stream() if device.type == 'cuda' else None
            use_pin = device.type == 'cuda'   # pinned memory solo útil con CUDA
            self._pinned    = [
                torch.empty(batch_size * (seq_len + 1), dtype=torch.int64).pin_memory()
                if use_pin else
                torch.empty(batch_size * (seq_len + 1), dtype=torch.int64)
                for _ in range(n_prefetch)
            ]
            self._gpu_bufs  = [
                torch.empty(batch_size, seq_len + 1, dtype=torch.int64, device=device)
                for _ in range(n_prefetch)
            ]
            self._ready     = [False] * n_prefetch
            self._slot      = 0
            self._n_prefetch = n_prefetch
            # Iniciar primer prefetch
            self._prefetch_to_slot(0)

    def _prefetch_to_slot(self, slot: int):
        """Carga un batch secuencial directo CPU → pinned → GPU async."""
        B = self.batch_size
        T = self.seq_len + 1
        chunk_size = B * T
        
        if self._ptr + chunk_size > self._n_toks:
            self._ptr = 0 # loop dataset
            
        # Zero-copy slicing from mmap, cast to int64, into pinned buffer
        raw = self._mmap[self._ptr : self._ptr + chunk_size].astype(np.int64)
        buf = self._pinned[slot]
        buf.copy_(torch.from_numpy(raw))
        
        self._ptr += chunk_size
        
        if self._stream_d is not None:
            with torch.cuda.stream(self._stream_d):
                self._gpu_bufs[slot].copy_(buf.view(B, T), non_blocking=True)
        else:
            self._gpu_bufs[slot].copy_(buf.view(B, T))
            
        self._ready[slot] = True

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Retorna (input_ids, labels) en GPU. Non-blocking para el stream de cómputo."""
        if self.hbm:
            limit  = self._n_toks - self.seq_len - 2
            starts = torch.randint(0, limit, (self.batch_size,),
                                   device=self.device, generator=self._rng)
            idx    = starts.unsqueeze(1) + torch.arange(
                self.seq_len + 1, device=self.device)
            flat   = self._data_gpu[idx.view(-1)].view(self.batch_size, self.seq_len + 1)
            return flat[:, :self.seq_len], flat[:, 1:]

        # mmap + async stream: esperar a que el slot actual esté listo
        slot = self._slot
        # Sincronizar stream_data antes de usar el buffer (el cómputo sigue en stream 0)
        if self._stream_d is not None:
            torch.cuda.current_stream().wait_stream(self._stream_d)
        chunk  = self._gpu_bufs[slot]
        ids    = chunk[:, :self.seq_len]
        labels = chunk[:, 1:]

        # Iniciar prefetch del próximo slot en parallel
        next_slot = (slot + 1) % self._n_prefetch
        self._prefetch_to_slot(next_slot)
        self._slot = next_slot

        return ids, labels


# ─────────────────────────────────────────────────────────────────────────────
# CUDAGraphElite — captura F+B con replay rápido y soporte a accum externo
# ─────────────────────────────────────────────────────────────────────────────

class CUDAGraphElite:
    """
    Captura el paso forward+backward en un CUDA Graph para cero overhead Python.

    Diseño:
      - Captura UN paso F+B (no el grad_accum completo)
      - Para grad_accum > 1: el caller hace N replays con diferentes inputs
        → los gradientes se acumulan naturalmente en .grad (sum de N backward)
      - zero_grad() siempre ocurre FUERA del grafo (Python call)
      - El grafo captura: forward + backward (autograd incluido) + loss retorno

    Restricciones para capture exitosa:
      1. graph_mode=True en todas las AdvancedChimeraLayer
      2. TTT update fuera del grafo (via TTTGradSupervisor)
      3. Shapes estáticos (batch_size, seq_len fijos)
      4. No Python-if sobre tensores (Chimera usa soft-gating → OK)
    """

    def __init__(self, model: ChimeraLM, amp_dtype: torch.dtype, aux_weight: float = 0.01):
        self.model      = model
        self.amp_dtype  = amp_dtype
        self.aux_weight = aux_weight
        self._graph     = None
        self._s_ids     = None
        self._s_labels  = None
        self._s_loss    = None
        self._s_ld      = None
        self._captured  = False

    def _fwd_bwd(self) -> tuple[torch.Tensor, dict]:
        """Forward + backward sobre tensores estáticos."""
        with torch.amp.autocast('cuda', dtype=self.amp_dtype):
            logits, loss, ld = self.model(
                self._s_ids, labels=self._s_labels, aux_weight=self.aux_weight
            )
        loss.backward()
        return loss.detach(), ld

    def warmup(self, data_stream: ZeroCopyBinaryStream, n: int = 25):
        """Ejecuta n pasos normales para inicializar buffers lazy y archives."""
        print(f"[CUDAGraph] Warmup: {n} pasos sin grafo...")
        self.model.train()
        for i in range(n):
            ids, labels = data_stream.get_batch()
            self.model.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                _, loss, _ = self.model(ids, labels=labels, aux_weight=self.aux_weight)
            loss.backward()
        torch.cuda.synchronize()

        # Activar graph_mode
        n_set = 0
        for m in self.model.modules():
            if hasattr(m, 'graph_mode'):
                m.graph_mode = True
                n_set += 1
        print(f"[CUDAGraph] graph_mode=True en {n_set} módulos")

    def capture(self, ids: torch.Tensor, labels: torch.Tensor):
        """Captura el CUDA Graph con tensores estáticos."""
        B, S = ids.shape
        self._s_ids    = ids.clone().contiguous()
        self._s_labels = labels.clone().contiguous()

        # Pre-warmup del stream antes de capture
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                self.model.zero_grad(set_to_none=False)
                self._fwd_bwd()
        torch.cuda.current_stream().wait_stream(s)
        # Inicializar todos los .grad como zero (no None) para acumulación correcta
        self.model.zero_grad(set_to_none=False)

        # Capture
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._s_loss, self._s_ld = self._fwd_bwd()

        torch.cuda.synchronize()
        self._captured = True
        print(f"[CUDAGraph] ✓ Grafo capturado. B={B} S={S}")

    def step(self, ids: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Copia inputs y hace replay. Acumula grads si llamado múltiples veces."""
        assert self._captured, "Llamar capture() primero"
        self._s_ids.copy_(ids, non_blocking=True)
        self._s_labels.copy_(labels, non_blocking=True)
        self._graph.replay()
        # Retornar references a los buffers estáticos del grafo
        return self._s_loss, self._s_ld

    @property
    def is_captured(self) -> bool:
        return self._captured


# ─────────────────────────────────────────────────────────────────────────────
# WSD LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def wsd_lr(step: int, total_steps: int, warmup: int, decay_frac: float = 0.1) -> float:
    decay_start = int(total_steps * (1 - decay_frac))
    if step < warmup:
        return max(step, 1) / max(warmup, 1)
    if step < decay_start:
        return 1.0
    t = (step - decay_start) / max(total_steps - decay_start, 1)
    return max(0.01, 0.5 * (1 + math.cos(math.pi * t)))


# ─────────────────────────────────────────────────────────────────────────────
# build_model / build_optimizer helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_size: str, vocab_size: int, device: torch.device,
                ckpt_interval: int = 2) -> ChimeraLM:
    if model_size == '125M':
        return build_chimera_125M(vocab_size=vocab_size, ckpt_interval=ckpt_interval).to(device)
    if model_size == '350M':
        return build_chimera_350M(vocab_size=vocab_size, ckpt_interval=ckpt_interval).to(device)
    cfg = ChimeraConfig(d_model=256, n_layers=4, expand=2, headdim=32)
    return ChimeraLM(cfg, vocab_size=vocab_size, ckpt_interval=ckpt_interval).to(device)


def build_optimizer(model: ChimeraLM, use_muon: bool, lr: float, muon_lr: float,
                    adamw_lr: float, wd: float) -> torch.optim.Optimizer:
    # Separar parámetros: 2D matrices (no embedding) → Muon
    #                      resto → AdamW
    emb_ids = {id(model.embedding.weight), id(model.lm_head.weight)}

    muon_params  = []
    adamw_decay  = []
    adamw_nodecay= []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embed = id(p) in emb_ids or 'embedding' in name
        if use_muon and p.ndim == 2 and not is_embed and min(p.shape) >= 2:
            muon_params.append(p)
        elif p.ndim == 1 or is_embed or 'norm' in name or 'bias' in name:
            adamw_nodecay.append(p)
        else:
            adamw_decay.append(p)

    if use_muon:
        # Grupo Muon (matrices 2D) + dos grupos AdamW (decay y no-decay)
        # adamw_decay  : proyecciones 2D no-matrix-eligible, residuales → wd
        # adamw_nodecay: bias, norm, embedding, lm_head               → wd=0
        opt = MuonElite(
            muon_params,
            lr=muon_lr,
            momentum=0.95,
            nesterov=True,
            adamw_params=adamw_decay,       # solo params con weight decay
            adamw_lr=adamw_lr,
            adamw_betas=(0.9, 0.95),
            adamw_wd=wd,
        )
        # Añadir grupo nodecay explícitamente con wd=0
        if adamw_nodecay:
            opt.add_param_group({
                'params':   adamw_nodecay,
                'lr':       adamw_lr,
                'momentum': 0.95,
                'nesterov': True,
                '_mode':    'adamw',
                '_betas':   (0.9, 0.95),
                '_wd':      0.0,          # sin weight decay para bias/norm/emb
                'initial_lr': adamw_lr,
            })
    else:
        opt = torch.optim.AdamW(
            [
                {'params': adamw_decay,   'weight_decay': wd},
                {'params': adamw_nodecay, 'weight_decay': 0.0},
            ],
            lr=lr, betas=(0.9, 0.95), fused=True,
        )

    # Guardar initial_lr en cada grupo para que el scheduler funcione
    # correctamente también al reanudar desde checkpoint
    for pg in opt.param_groups:
        if 'initial_lr' not in pg:
            pg['initial_lr'] = pg['lr']

    return opt


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop Principal
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        prop = torch.cuda.get_device_properties(0)
        vram_gb = prop.total_memory / 1e9
        print(f"[H200 Elite] GPU: {prop.name}  VRAM: {vram_gb:.0f}GB")
        sm = torch.cuda.get_device_capability(0)
        if sm[0] >= 9 and not args.compile:
            print(f"[perf] GPU SM{sm[0]}.{sm[1]} detectada (Hopper+). "
                  f"Recomendado: --compile para +30-50% throughput.")
        if vram_gb >= 140:
            suggested = 128 if args.compile else 64
        elif vram_gb >= 80:
            suggested = 64 if args.compile else 32
        else:
            suggested = min(32, args.batch)
        if args.batch < suggested and args.ckpt_interval < 999:
            print(f"[perf] Con grad-ckpt activo (interval={args.ckpt_interval}), "
                  f"batch={args.batch} es conservador. Sugerido: --batch {suggested}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    meta_path = Path(args.data_dir) / 'meta.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        shard_paths = [s['path'] for s in meta['shards']]
        vocab_size  = args.vocab or meta.get('vocab_size', 32000)
        token_dtype = meta.get('dtype', 'uint16')
    else:
        bins = sorted(Path(args.data_dir).glob('*.bin'))
        if not bins:
            raise FileNotFoundError(
                f"No se encontraron .bin ni meta.json en {args.data_dir}\n"
                "Ejecutar: python tokenize_dataset.py --data_dir /raw --out_dir ."
            )
        shard_paths = [str(b) for b in bins]
        vocab_size  = args.vocab or 32000
        token_dtype = 'uint16'
        print(f"[data] {len(shard_paths)} shards encontrados (sin meta.json)")

    data_stream = ZeroCopyBinaryStream(
        shard_paths = shard_paths,
        seq_len     = args.seq_len,
        batch_size  = args.batch,
        device      = device,
        hbm         = args.hbm_dataset and device.type == 'cuda',
        n_prefetch  = 3,
        token_dtype = token_dtype,
    )
    mode_str = 'HBM (zero-latency)' if (args.hbm_dataset and device.type == 'cuda') else 'mmap+async'
    print(f"[data] Dataset: {mode_str}  |  seq_len={args.seq_len}  batch={args.batch}")

    # ── Modelo ────────────────────────────────────────────────────────────────
    print(f"\n[model] Construyendo CHIMERA-{args.model}...")
    model = build_model(args.model, vocab_size, device, ckpt_interval=args.ckpt_interval)
    amp_dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32
    if args.dtype == 'bfloat16':
        model = model.bfloat16()
    model.train()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {args.model}  |  {n_params/1e6:.2f}M params  |  {args.dtype}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # IMPORTANTE: build_optimizer usa id(param) para separar grupos Muon/AdamW.
    # Debe construirse sobre el modelo NO compilado para que los IDs sean estables.
    opt = build_optimizer(
        model,
        use_muon  = args.muon,
        lr        = args.lr,
        muon_lr   = args.muon_lr,
        adamw_lr  = args.adamw_lr,
        wd        = args.wd,
    )

    # ── torch.compile (Inductor + Triton backend — DESPUÉS del optimizer) ─────
    # Aplicar DESPUÉS de build_optimizer porque compile() envuelve el forward;
    # los parámetros siguen siendo los originales (OptimizedModule los delega).
    # Beneficio neto en H200 SM90: +20-40% throughput para modelos 130M.
    # Con fullgraph=False se permiten graph breaks en Python control flow
    # (routing soft-gates, TTT skip en graph_mode=True).
    # 'reduce-overhead' activa CUDA Graphs internamente (PyTorch 2.6+).
    # Desactivamos el CUDAGraphElite manual ya que es redundante.
    if getattr(args, 'compile', False) and device.type == 'cuda':
        # Kwargs adaptativos por GPU: LAPTOP_ADA→reduce-overhead,
        # A100/H100→max-autotune+fullgraph=True, Blackwell→max-autotune+fullgraph=True
        _ckwargs = _get_compile_kwargs(mode='train')
        print(f"[compile] torch.compile(mode='{_ckwargs['mode']}', "
              f"fullgraph={_ckwargs['fullgraph']})  — compilando...")
        _compile_t0 = time.perf_counter()
        model = torch.compile(
            model,
            **_ckwargs,
        )
        print(f"[compile] ✓ {time.perf_counter()-_compile_t0:.1f}s  "
              f"(primera inferencia aún más lenta — JIT de kernels Triton)")
        # reduce-overhead gestiona CUDA Graphs internamente; max-autotune no.
        args.cuda_graphs = _ckwargs['mode'] not in ('reduce-overhead',)

    # ── LR schedule ───────────────────────────────────────────────────────────
    tokens_per_step  = args.batch * args.seq_len * args.grad_accum
    total_steps      = int(args.total_tokens / tokens_per_step)
    warmup_steps     = min(max(20, int(total_steps * 0.01)), 500)
    print(f"[sched] WSD: total_steps={total_steps:,}  warmup={warmup_steps}  "
          f"tok/step={tokens_per_step:,}")

    # ── Sistemas de estabilidad ────────────────────────────────────────────────
    grad_monitor  = GradHealthMonitor(max_consecutive_skips=10)
    spike_detect  = LossSpikeDetector(alpha=0.98, spike_sigma=3.0,
                                       warmup_steps=warmup_steps * 2)
    router_watch  = RouterEntropyWatchdog(collapse_threshold=1.2, boost_factor=10.0)
    ttt_super     = TTTGradSupervisor(clip_budget=0.05, lr=args.lr * 0.1)
    chimera_ann   = ChimeraAnnealer(model, slr_steps=min(5000, total_steps // 5))

    # ── Routing loss obj (para ajustar entropy_weight dinámicamente) ──────────
    # NOTA: debe encontrarse ANTES del loop; si es None, el watchdog no ajusta.
    from chimera_losses import ChimeraRoutingLoss
    _routing_loss_obj = None
    _base_ent_w       = None
    for _m in model.modules():
        if isinstance(_m, ChimeraRoutingLoss):
            _routing_loss_obj = _m
            _base_ent_w       = getattr(_m, 'entropy_weight', 0.01)
            break

    # ── CUDA Graph setup ──────────────────────────────────────────────────────
    use_graphs = args.cuda_graphs and device.type == 'cuda'
    graph_elite = None
    if use_graphs:
        graph_elite = CUDAGraphElite(model, amp_dtype=amp_dtype, aux_weight=args.aux_weight)
        graph_elite.warmup(data_stream, n=25)
        ids_cap, lbl_cap = data_stream.get_batch()
        graph_elite.capture(ids_cap, lbl_cap)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt_dir   = Path(args.ckpt_dir)
    checkpointer = AsyncCheckpointer(ckpt_dir)

    global_step = 0
    total_toks  = 0
    loss_ema    = None

    # Cargar checkpoint previo si existe
    latest = ckpt_dir / 'latest.pt'
    if latest.exists() and not args.no_resume:
        # weights_only=False requerido porque guardamos dicts con scalars/strings.
        # Seguro: el archivo lo escribimos nosotros (no es modelo externo).
        ckpt = torch.load(str(latest), map_location=device, weights_only=False)
        
        # FIX: Compatibilidad de torch.compile (eliminar '_orig_mod.')
        sd = ckpt['model']
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        # Re-prefijar si el modelo actual está compilado
        if hasattr(model, '_orig_mod'):
            clean_sd = {f'_orig_mod.{k}': v for k, v in clean_sd.items()}
            
        model.load_state_dict(clean_sd)
        opt.load_state_dict(ckpt['opt'])
        global_step = ckpt.get('step', 0)
        total_toks  = ckpt.get('total_toks', 0)
        loss_ema    = ckpt.get('loss_ema', None)
        print(f"[resume] step={global_step}  tokens={total_toks:,}  ema={loss_ema:.4f}")

    # ── Log file ─────────────────────────────────────────────────────────────
    log_path  = ckpt_dir / 'train_elite.log'
    log_vals  = []

    def log(msg: str):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    log(f"\n{'='*72}")
    log(f"  CHIMERA Elite Trainer — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    log(f"  Modelo: {args.model} ({n_params/1e6:.2f}M)  |  CUDA Graph: {use_graphs}")
    log(f"  Target: {args.total_tokens/1e9:.1f}B tokens en {total_steps:,} steps")
    log(f"{'='*72}\n")

    # ── Main training loop ────────────────────────────────────────────────────
    t_start   = time.perf_counter()
    t_log_acc = 0.0

    while global_step < total_steps:
        t_step = time.perf_counter()

        # ── LR schedule (actualizar en param_groups manualmente) ─────────────
        lr_scale = wsd_lr(global_step, total_steps, warmup_steps) * spike_detect.lr_scale
        for pg in opt.param_groups:
            # initial_lr se guardó en build_optimizer; siempre debe existir
            pg['lr'] = pg['initial_lr'] * lr_scale

        # ── Chimera-specific annealing ────────────────────────────────────────
        chimera_ann.step(global_step)

        # ── Gradient accumulation loop ────────────────────────────────────────
        opt.zero_grad(set_to_none=False)   # False: mantiene tensors para acumulación en graph
        accum_loss = 0.0
        last_ld    = {}

        grad_clip_eff = args.grad_clip * spike_detect.clip_scale

        for accum_i in range(args.grad_accum):
            ids, labels = data_stream.get_batch()

            if use_graphs and graph_elite.is_captured:
                loss_t, ld_t = graph_elite.step(ids, labels)
                loss_val = loss_t.item() / args.grad_accum
                last_ld  = {k: (v.item() if hasattr(v, 'item') else v)
                            for k, v in ld_t.items()}
            else:
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(args.dtype == 'bfloat16')):
                    _, loss, ld_t = model(ids, labels=labels, aux_weight=args.aux_weight)
                (loss / args.grad_accum).backward()
                loss_val = loss.item() / args.grad_accum
                last_ld  = {k: (v.item() if hasattr(v, 'item') else v)
                            for k, v in ld_t.items()}

            accum_loss += loss_val

        # ── Gradient health check (PRIMERO — antes de cualquier escritura) ──────
        # IMPORTANTE: grad_monitor DEBE ir antes de ttt_super.apply().
        # Si el backward produjo NaN en los grads del modelo, el forward también
        # pudo haber generado _pending_ttt_grad corruptos (NaN desde el mini-chunk).
        # Aplicar esos grads a dt_bias corrompería el parámetro permanentemente.
        grads_ok = grad_monitor.check(model)
        if not grads_ok:
            # Descartar _pending_ttt_grad de todas las capas para evitar que el
            # siguiente paso aplique grads del backward corrupto.
            from advanced_chimera import AdvancedChimeraLayer
            for _layer in model.modules():
                if isinstance(_layer, AdvancedChimeraLayer):
                    _layer._pending_ttt_grad = None
            log(f"[warn] step={global_step} — NaN/Inf en grads, step descartado "
                f"(total_skips={grad_monitor._total_skips})")
            global_step += 1
            continue

        # ── TTT Grad supervisor (solo si grads son sanos) ─────────────────────
        ttt_super.apply(model)

        # ── Gradient clip ─────────────────────────────────────────────────────
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_eff)

        # ── Optimizer step ────────────────────────────────────────────────────
        opt.step()

        # ── Router entropy watchdog ───────────────────────────────────────────
        entropy_mult = router_watch.update(last_ld)
        # Aplicar multiplicador directamente al objeto routing_loss del modelo.
        # Fuera del CUDA Graph → seguro (entropy_weight es un float de Python).
        if _routing_loss_obj is not None:
            _routing_loss_obj.entropy_weight = _base_ent_w * entropy_mult

        # ── Loss spike detection ──────────────────────────────────────────────
        spike_info = spike_detect.update(accum_loss)
        if spike_info:
            log(f"[spike] step={global_step}  loss={spike_info['loss']:.4f}  "
                f"ema={spike_info['ema']:.4f}  ({spike_info['std']:.4f}σ)  "
                f"→ LR×0.5 + clip×0.5 por {spike_info['recovery_for']} pasos "
                f"[spikes totales: {spike_info['n_total']}]")

        # ── Métricas ──────────────────────────────────────────────────────────
        global_step += 1
        step_toks    = args.batch * args.seq_len * args.grad_accum
        total_toks  += step_toks
        step_ms      = (time.perf_counter() - t_step) * 1000
        toks_per_s   = step_toks / (step_ms / 1000)

        alpha    = 0.98
        loss_ema = (accum_loss if loss_ema is None
                    else alpha * loss_ema + (1 - alpha) * accum_loss)

        # ── Logging ───────────────────────────────────────────────────────────
        if global_step % args.log_every == 0:
            elapsed     = time.perf_counter() - t_start
            toks_all    = total_toks / elapsed
            ppl         = math.exp(min(loss_ema, 20.0))
            # ── Loss breakdown ─────────────────────────────────────────────
            lm_loss_v   = last_ld.get('lm',                   float('nan'))
            rout_loss_v = last_ld.get('routing',              float('nan'))
            z_loss_v    = last_ld.get('routing/z_loss',       float('nan'))
            h_bits      = last_ld.get('routing/H_bits',       float('nan'))
            p_fast      = last_ld.get('routing/p_fast',       float('nan'))
            p_hybrid    = last_ld.get('routing/p_hybrid',     float('nan'))
            bal_loss_v  = last_ld.get('routing/balance_loss', float('nan'))
            cur_lr      = opt.param_groups[0].get('lr', args.lr)
            # ── fast_prob_ema: detección de colapso de tier ────────────────
            # Cada AdvancedChimeraLayer actualiza fast_prob_ema (EMA del p_FAST
            # promedio). Si cualquier capa supera 0.90, el routing es degenerate.
            _fast_emas = [
                m.fast_prob_ema.item()
                for m in model.modules()
                if hasattr(m, 'fast_prob_ema')
            ]
            max_fast_ema = max(_fast_emas) if _fast_emas else float('nan')
            if max_fast_ema > 0.90:
                log(f"[warn-router] step={global_step}  max_fast_ema={max_fast_ema:.3f} "
                    f"— posible colapso al tier FAST. "
                    f"RouterEntropyWatchdog activo en {router_watch._recovering} pasos.")
            # ── Skip % ─────────────────────────────────────────────────────
            nan_skip_pct   = 100.0 * grad_monitor._total_skips / max(global_step, 1)
            spike_adj_pct  = 100.0 * spike_detect._steps_in_recovery_total / max(global_step, 1)

            # ── Métricas adicionales ────────────────────────────────────────
            bpb = loss_ema / math.log(2)              # bits per byte (bpb = loss/ln2)
            vram_used  = torch.cuda.memory_allocated() / 1e9 if device.type == 'cuda' else 0.0
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9 if device.type == 'cuda' else 0.0
            # MFU (Model FLOPs Utilization): fracción de TFLOPs teóricos en uso.
            # Fórmula: MFU = (6 * N_params * tok/s) / peak_FLOPs
            # H200 BF16 peak: 989 TFLOPs. Factor 6 = 2(forward)+4(backward).
            # Valor de referencia: SSM bien optimizado ~18-22%, transformer ~35-45%.
            peak_tflops = 989e12   # H200 BF16
            mfu_pct = 100.0 * (6 * n_params * toks_per_s) / peak_tflops
            tok_remain = max(args.total_tokens - total_toks, 0)
            eta_s      = tok_remain / toks_all if toks_all > 0 else float('inf')
            eta_str    = (f"{int(eta_s//3600)}h{int((eta_s%3600)//60):02d}m"
                          if eta_s < 86400 else "??")

            msg = (
                f"[{global_step:>6}/{total_steps}]  "
                f"loss={accum_loss:.4f}(lm={lm_loss_v:.4f}|aux={rout_loss_v:.4f}|z={z_loss_v:.2e})  "
                f"ema={loss_ema:.4f}  ppl={ppl:.1f}  bpb={bpb:.3f}  "
                f"lr={cur_lr:.2e}  grad={grad_norm:.3f}  "
                f"H={h_bits:.2f}b  pF={p_fast:.2f}|pH={p_hybrid:.2f}  "
                f"maxFema={max_fast_ema:.2f}  "
                f"tps={toks_per_s:>8,.0f}(avg={toks_all:,.0f})  "
                f"MFU={mfu_pct:.1f}%  "
                f"VRAM={vram_used:.1f}/{vram_total:.0f}GB  "
                f"skip={nan_skip_pct:.1f}%|{spike_adj_pct:.1f}%  "
                f"[{total_toks/1e9:.3f}B/{args.total_tokens/1e9:.1f}B  "
                f"ETA={eta_str}  t={elapsed/60:.0f}min]"
            )
            log(msg)
            log_vals.append({
                'step':             global_step,
                'loss':             accum_loss,
                'loss_lm':          lm_loss_v,
                'loss_routing':     rout_loss_v,
                'loss_z':           z_loss_v,
                'loss_balance':     bal_loss_v,
                'loss_ema':         loss_ema,
                'ppl':              ppl,
                'bpb':              bpb,
                'grad_norm':        float(grad_norm),
                'tok_per_s':        toks_per_s,
                'tok_per_s_avg':    toks_all,
                'total_toks':       total_toks,
                'elapsed_min':      elapsed / 60.0,
                'H_bits':           h_bits,
                'p_fast':           p_fast,
                'p_hybrid':         p_hybrid,
                'max_fast_ema':     max_fast_ema,
                'lr':               float(cur_lr),
                'vram_used_gb':     vram_used,
                'nan_skip_pct':     nan_skip_pct,
                'spike_adj_pct':    spike_adj_pct,
                'router_collapses': router_watch.n_collapses,
            })

        # ── Checkpoint ────────────────────────────────────────────────────────
        if global_step % args.save_every == 0 or global_step == total_steps:
            # Desactivar graph_mode temporalmente para guardar estado limpio
            for m in model.modules():
                if hasattr(m, 'graph_mode'):
                    m.graph_mode = False
            ckpt_data = {
                'step':       global_step,
                'total_toks': total_toks,
                'loss_ema':   loss_ema,
                'model':      model.state_dict(),
                'opt':        opt.state_dict(),
                'args':       vars(args),
            }
            checkpointer.save(global_step, ckpt_data)
            log(f"[ckpt] Encolado step={global_step}  loss_ema={loss_ema:.4f}  "
                f"(async — no bloquea)")
            # Restaurar graph_mode
            for m in model.modules():
                if hasattr(m, 'graph_mode'):
                    m.graph_mode = use_graphs

        if total_toks >= args.total_tokens:
            break

    # ── Fin del training ──────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_start
    eff_tps = total_toks / total_elapsed

    log(f"\n{'='*72}")
    log(f"  CHIMERA Elite — Training completo")
    log(f"  Steps          : {global_step:,}")
    log(f"  Tokens         : {total_toks/1e9:.3f}B")
    log(f"  Tiempo         : {total_elapsed/3600:.2f}h  ({total_elapsed/60:.1f}min)")
    log(f"  Throughput eff : {eff_tps:,.0f} tok/s")
    log(f"  Loss EMA final : {loss_ema:.4f}  PPL≈{math.exp(min(loss_ema,20)):.1f}")
    log(f"  NaN skips      : {grad_monitor._total_skips}  "
        f"({100*grad_monitor._total_skips/max(global_step,1):.1f}% de steps)")
    log(f"  Spike-adjusted : {spike_detect._steps_in_recovery_total} steps  "
        f"({100*spike_detect._steps_in_recovery_total/max(global_step,1):.1f}%)  "
        f"|  {spike_detect._n_spikes} eventos de spike")
    log(f"  Router collapses: {router_watch.n_collapses}")
    log(f"{'='*72}\n")

    # Esperar a que el último checkpoint se guarde
    checkpointer.join()

    # Guardar log JSON
    (ckpt_dir / 'train_elite_log.json').write_text(json.dumps(log_vals, indent=2))
    return loss_ema


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark de throughput (sin datos reales)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(args):
    """Mide tok/s con y sin CUDA Graphs para estimar tiempo total de training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    model = build_model(args.model, args.vocab or 32000, device,
                        ckpt_interval=getattr(args, 'ckpt_interval', 2))
    amp_dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32
    if args.dtype == 'bfloat16':
        model = model.bfloat16()
    model.train()

    B, S = args.batch, args.seq_len
    ids    = torch.randint(0, args.vocab or 32000, (B, S), device=device)
    labels = torch.randint(0, args.vocab or 32000, (B, S), device=device)
    n_params = sum(p.numel() for p in model.parameters())

    def measure(tag, n_reps=20, use_graph=False):
        g_step = None
        if use_graph:
            # Mini-warmup para benchmark
            for _ in range(10):
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    _, loss, _ = model(ids, labels=labels)
                loss.backward()
                model.zero_grad(set_to_none=False)
            for m in model.modules():
                if hasattr(m, 'graph_mode'):
                    m.graph_mode = True
            g_step = CUDAGraphElite(model, amp_dtype=amp_dtype)
            g_step.capture(ids, labels)
            model.zero_grad(set_to_none=False)

        # Warmup
        for _ in range(5):
            model.zero_grad(set_to_none=False)
            if g_step:
                g_step.step(ids, labels)
            else:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    _, loss, _ = model(ids, labels=labels)
                loss.backward()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_reps):
            model.zero_grad(set_to_none=False)
            if g_step:
                g_step.step(ids, labels)
            else:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    _, loss, _ = model(ids, labels=labels)
                loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ms  = elapsed / n_reps * 1000
        tps = B * S / (ms / 1000)
        return ms, tps

    print(f"\n{'='*65}")
    print(f"  CHIMERA {args.model} Elite Benchmark")
    print(f"  GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"  {n_params/1e6:.2f}M params  |  B={B} S={S}  |  {args.dtype}")
    print(f"{'='*65}")
    print(f"  {'Configuración':<30s}  {'ms/step':>10}  {'tok/s':>12}  speedup")
    print(f"  {'-'*65}")

    ms_base, tps_base = measure('Sin CUDA Graph  (BF16)', use_graph=False)
    print(f"  {'Sin CUDA Graph (BF16)':<30s}  {ms_base:>8.1f}ms  {tps_base:>11,.0f}  1.00×")

    try:
        ms_g, tps_g = measure('Con CUDA Graph', use_graph=True)
        speedup = tps_g / tps_base
        print(f"  {'Con CUDA Graph':<30s}  {ms_g:>8.1f}ms  {tps_g:>11,.0f}  {speedup:.2f}×")
    except Exception as e:
        print(f"  CUDA Graph: FALLÓ ({type(e).__name__}: {e})")
        tps_g, ms_g = tps_base, ms_base

    est_tps = tps_g
    tokens_3B   = 3e9
    hours_3B    = tokens_3B / est_tps / 3600
    tokens_chin = 1.59e9
    hours_chin  = tokens_chin / est_tps / 3600

    print(f"\n  {'─'*65}")
    print(f"  Throughput estimado H200 (con CUDA Graph): {est_tps:,.0f} tok/s")
    print(f"  Chinchilla 125M (1.59B tok):  {hours_chin:.2f}h")
    print(f"  3B tokens:                    {hours_3B:.2f}h")
    print(f"  {'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='train_h200_elite.py — CHIMERA Elite Trainer para H200 140GB'
    )
    # Data
    p.add_argument('--data_dir',    default=None)
    p.add_argument('--hbm_dataset', action='store_true', default=True,
                   help='Cargar dataset en HBM GPU (si VRAM suficiente)')
    p.add_argument('--no_hbm',      dest='hbm_dataset', action='store_false')

    # Model
    p.add_argument('--model', default='125M', choices=['tiny', '125M', '350M'])
    p.add_argument('--vocab', type=int, default=None)

    # Training
    # H200 óptimo: batch=128 S=2048 (262k tok/step) con HBM dataset.
    # Roofline: batch=128 aumenta SM occupancy ~4× vs batch=32 en SSM scan
    # (el scan es BW-bound; más warps activos esconden latencias HBM3e).
    # Con grad_ckpt ckpt_interval=2 + chunked CE: ~35-40 GB VRAM en 125M.
    p.add_argument('--batch',        type=int,   default=128,
                   help='Batch size. H200 rec: 128 (máxima SM occupancy en SSM scan)')
    p.add_argument('--seq_len',      type=int,   default=2048)
    p.add_argument('--grad_accum',   type=int,   default=1)
    p.add_argument('--lr',           type=float, default=3e-4)
    p.add_argument('--muon_lr',      type=float, default=0.02)
    p.add_argument('--adamw_lr',     type=float, default=3e-4)
    p.add_argument('--wd',           type=float, default=0.1)
    p.add_argument('--grad_clip',    type=float, default=1.0)
    p.add_argument('--aux_weight',   type=float, default=0.01)
    p.add_argument('--total_tokens', type=float, default=3e9)
    p.add_argument('--dtype',        default='bfloat16', choices=['float32', 'bfloat16'])
    p.add_argument('--muon',         action='store_true',  default=True)
    p.add_argument('--no_muon',      dest='muon', action='store_false')

    # CUDA Graph (manual) vs torch.compile (automático, preferible en H200)
    p.add_argument('--cuda_graphs',    action='store_true',  default=True,
                   help='CUDA Graphs manuales. Se deshabilita automáticamente con --compile')
    p.add_argument('--no_cuda_graphs', dest='cuda_graphs', action='store_false')
    p.add_argument('--compile',        action='store_true', default=True,
                   help='torch.compile (Triton/Inductor). Activado por defecto. '
                        'Compilación tarda ~2-4 min la primera vez, luego 20-40%% más rápido.')
    p.add_argument('--no_compile',     dest='compile', action='store_false',
                   help='Desactivar torch.compile.')

    # Checkpoint / logging
    p.add_argument('--ckpt_dir',   default='./ckpt_elite')
    p.add_argument('--log_every',  type=int, default=10)
    p.add_argument('--save_every', type=int, default=500)
    p.add_argument('--ckpt_interval', type=int, default=2,
                   help='Gradient checkpointing interval (2=cada 2 capas, 999=desactivado)')
    p.add_argument('--no_resume',  action='store_true')
    p.add_argument('--seed',       type=int, default=42)

    # Utilities
    p.add_argument('--benchmark', action='store_true')

    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.benchmark:
        benchmark(args)
        return

    if args.data_dir is None:
        print('ERROR: --data_dir requerido. Usar --benchmark para solo throughput.')
        print('  python train_h200_elite.py --benchmark --model 125M --batch 32')
        import sys; sys.exit(1)

    train(args)


if __name__ == '__main__':
    main()
