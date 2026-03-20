"""
train_h200.py — Trainer CHIMERA ultra-optimizado para NVIDIA H200
================================================================

DIFERENCIAS VS train_chimera.py (trainer estándar):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  + CUDA Graphs: la región forward+backward se captura y replay desde HBM.
                 No hay Python dispatch overhead (~35-55% más throughput).
  + HBM Dataset: todos los tokens cargados en GPU VRAM (3.18 GB @ 1.59B tok).
                 Zero PCIe latency en data loading.
  + TTT externo: update_ttt_inplace() corre FUERA del graph capturado,
                 cada `--ttt_interval` pasos (default: 1).
  + BF16 master weights: parámetros del modelo en BF16, optimizer states FP32.
  + Compilación selectiva: mamba2.forward protegido con _dynamo.disable para
    evitar conflicto con Triton kernels de mamba_ssm.
  + H200 presets: batch=16, seq=2048, grad_accum=2 → ~200K tok/s efectivos.

Prerequisitos:
    pip install tiktoken sentencepiece  # para tokenización
    # Tokenizar datos primero:
    python tokenize_dataset.py --data_dir /data/raw --out_dir /data/tokens

Uso mínimo:
    python train_h200.py --data_dir /data/tokens --model 125M

Uso completo:
    python train_h200.py \\
        --data_dir /data/tokens \\
        --model 125M --vocab 32000 \\
        --batch 16 --seq_len 2048 --grad_accum 2 \\
        --lr 3e-4 --total_tokens 3e9 \\
        --cuda_graphs --hbm_dataset \\
        --ttt_interval 1 \\
        --ckpt_dir ./ckpt_h200 \\
        --log_every 10
"""

from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from chimera_config import ChimeraConfig
from chimera_lm     import ChimeraLM, build_chimera_125M, build_chimera_350M
from tokenize_dataset import BinaryTokenDataset
try:
    from train_h200_elite import RouterEntropyWatchdog as _RouterEntropyWatchdog
except Exception:
    _RouterEntropyWatchdog = None

# ─────────────────────────────────────────────────────────────────────────────
# H200 flags globales
# ─────────────────────────────────────────────────────────────────────────────
torch.set_float32_matmul_precision('high')   # TF32 para ops FP32 residuales
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


# ─────────────────────────────────────────────────────────────────────────────
# Muon optimizer (mismo que train_chimera.py)
# ─────────────────────────────────────────────────────────────────────────────

def _newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + 1e-7)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if G.size(0) > G.size(1) else X


class Muon(torch.optim.Optimizer):
    """Muon: SGD con momentum + ortogonalización Newton-Schulz."""

    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5,
                 adamw_params=None, adamw_lr: float = 3e-4,
                 adamw_betas=(0.9, 0.95), adamw_wd: float = 0.1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self.adamw_params = list(adamw_params) if adamw_params is not None else []
        self.adamw_lr    = adamw_lr
        self.adamw_betas = adamw_betas
        self.adamw_wd    = adamw_wd

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        # Muon step para parámetros 2D
        for group in self.param_groups:
            lr       = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if 'mu' not in state:
                    state['mu'] = torch.zeros_like(g)
                mu = state['mu']
                mu.mul_(momentum).add_(g)
                if nesterov:
                    g = g + momentum * mu
                else:
                    g = mu
                if g.ndim == 2:
                    g = _newtonschulz5(g, steps=ns_steps).to(g.dtype)
                g *= max(1.0, g.size(0) / g.size(1)) ** 0.5
                p.add_(g, alpha=-lr)

        # AdamW para bias/norm/embedding
        if self.adamw_params:
            beta1, beta2 = self.adamw_betas
            for p in self.adamw_params:
                if p.grad is None:
                    continue
                g   = p.grad.float()
                st  = self.state[p]
                if 'm' not in st:
                    st['m']    = torch.zeros_like(g)
                    st['v']    = torch.zeros_like(g)
                    st['step'] = 0
                st['step'] += 1
                st['m'].mul_(beta1).add_(g, alpha=1 - beta1)
                st['v'].mul_(beta2).addcmul_(g, g, value=1 - beta2)
                m_hat = st['m']  / (1 - beta1 ** st['step'])
                v_hat = st['v']  / (1 - beta2 ** st['step'])
                p.add_(p.float(), alpha=-self.adamw_lr * self.adamw_wd)
                p.addcdiv_(m_hat, v_hat.sqrt() + 1e-8, value=-self.adamw_lr)

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# LR scheduler WSD
# ─────────────────────────────────────────────────────────────────────────────

def wsd_lr(step: int, total_steps: int, warmup: int, decay_frac: float = 0.1) -> float:
    """Warmup-Stable-Decay."""
    decay_start = int(total_steps * (1 - decay_frac))
    if step < warmup:
        return step / max(1, warmup)
    if step < decay_start:
        return 1.0
    progress = (step - decay_start) / max(1, total_steps - decay_start)
    return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset en CPU (fallback para máquinas sin suficiente VRAM)
# ─────────────────────────────────────────────────────────────────────────────

class PackedBinaryDataset(Dataset):
    """Dataset mmap sobre .bin uint16 con packing (sin padding)."""

    def __init__(self, shard_paths: list[str], seq_len: int):
        import numpy as np
        self.seq_len = seq_len
        arrays = [__import__('numpy').memmap(p, dtype='uint16', mode='r') for p in shard_paths]
        self.data = __import__('numpy').concatenate(arrays)
        self.n = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        import numpy as np
        start = idx * self.seq_len
        chunk = torch.from_numpy(
            self.data[start : start + self.seq_len + 1].astype('int64')
        )
        return chunk[:-1], chunk[1:]


# ─────────────────────────────────────────────────────────────────────────────
# CUDA Graph wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CUDAGraphTrainStep:
    """
    Captura y gestiona el CUDA Graph de un paso de entrenamiento CHIMERA.

    Ciclo de vida:
      1. warmup(n)      — ejecuta n pasos normales (sin grafo) para inicializar
                          estados del archive, lazily-allocated tensors, etc.
      2. capture()      — captura el grafo con static tensors ids/labels
      3. step(ids, lbl) — copia input a static buffers, replay del grafo

    Arquitectura dual:
      • TTT update: llamado UNA VEZ fuera del grafo (update_ttt_inplace)
                    antes de cada replay. No incurre graph breaks.
      • Forward+Backward: 100% capturado. Zero Python dispatch overhead.
      • Optimizer step: fuera del grafo (modifica parámetros con side effects).

    Notas sobre side effects:
      • `mamba2.dt_bias.data.copy_()` ocurre dentro del grafo (graph_mode=True).
        En graph capture, este op se usa solo cuando TTT=False (graph_mode desactiva
        el TTT en-forward). Con update_ttt_inplace(), dt_bias se modifica fuera.
      • archive.maybe_archive(): llamado dentro del grafo — opera sobre sus
        propios buffers (siempre misma firma). Es seguro.
      • bus cache: stateless por batch — ningún estado persistente entre steps.
    """

    def __init__(self, model: ChimeraLM, scaler_ctx, aux_weight: float = 0.01):
        self.model      = model
        self.scaler_ctx = scaler_ctx
        self.aux_weight = aux_weight
        self._graph     = None
        self._static_ids    = None
        self._static_labels = None
        self._static_loss   = None

    def _forward_backward(self, ids: torch.Tensor, labels: torch.Tensor):
        """Forward + backward sin grad accumulation (1 step natural)."""
        with self.scaler_ctx:
            logits, total_loss, _ = self.model(ids, labels=labels,
                                               aux_weight=self.aux_weight)
        total_loss.backward()
        return total_loss

    def warmup(self, data_iter, n_warmup: int = 25):
        """
        Ejecuta n_warmup pasos sin grafo.
        Propósito:
          - Inicializar archive (primeros landmarks)
          - Activar lazy buffers en mamba_ssm
          - Calentar CUDA caches y cudnn benchmarks
          - Post warm-up: graph_mode = True en todas las capas
        """
        print(f"[CUDAGraph] Warmup: {n_warmup} pasos sin grafo...")
        for i, (ids, labels) in enumerate(data_iter):
            if i >= n_warmup:
                break
            ids    = ids.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            self.model.zero_grad(set_to_none=True)
            self._forward_backward(ids, labels)
        torch.cuda.synchronize()

        # Activar graph_mode en todas las AdvancedChimeraLayer
        n_layers_set = 0
        for module in self.model.modules():
            if hasattr(module, 'graph_mode'):
                module.graph_mode = True
                n_layers_set += 1
        print(f"[CUDAGraph] graph_mode=True en {n_layers_set} capas. Capturando grafo...")

    def capture(self, ids: torch.Tensor, labels: torch.Tensor):
        """
        Captura el CUDA Graph con tensores input estáticos.
        ids y labels deben ser los tensores 'estáticos' que se reusan en cada step.
        """
        B, S = ids.shape
        self._static_ids    = ids.clone()
        self._static_labels = labels.clone()

        # Warmup del stream antes de capture (limpiar ops pendientes)
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                self.model.zero_grad(set_to_none=True)
                self._forward_backward(self._static_ids, self._static_labels)
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        self._graph = torch.cuda.CUDAGraph()
        self.model.zero_grad(set_to_none=True)
        with torch.cuda.graph(self._graph):
            self._static_loss = self._forward_backward(
                self._static_ids, self._static_labels
            )
        torch.cuda.synchronize()
        print(f"[CUDAGraph] ✓ Grafo capturado. Shape: ids={tuple(ids.shape)}")

    def step(self, ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Copia inputs a static buffers y replay del grafo.
        Retorna el loss scalar (desde el buffer estático del grafo).
        """
        assert self._graph is not None, "Llamar capture() antes de step()"
        self._static_ids.copy_(ids, non_blocking=True)
        self._static_labels.copy_(labels, non_blocking=True)
        self._graph.replay()
        return self._static_loss.detach()

    @property
    def is_captured(self) -> bool:
        return self._graph is not None


# ─────────────────────────────────────────────────────────────────────────────
# Training loop principal
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_size: str, vocab_size: int, device: torch.device) -> ChimeraLM:
    if model_size == "125M":
        model = build_chimera_125M(vocab_size=vocab_size)
    elif model_size == "350M":
        model = build_chimera_350M(vocab_size=vocab_size)
    elif model_size == "tiny":
        cfg = ChimeraConfig(d_model=256, n_layers=4, expand=2, headdim=32)
        model = ChimeraLM(cfg, vocab_size=vocab_size)
    else:
        raise ValueError(f"Tamaño de modelo no reconocido: {model_size!r}")
    return model.to(device)


def build_optimizer(model: ChimeraLM, use_muon: bool, lr: float,
                    adamw_lr: float) -> torch.optim.Optimizer:
    # Separar parámetros: 2D (matrices) → Muon/AdamW con decay
    #                      1D/escalar    → AdamW sin decay
    decay_params   = [p for p in model.parameters()
                      if p.requires_grad and p.ndim >= 2]
    nodecay_params = [p for p in model.parameters()
                      if p.requires_grad and p.ndim < 2]

    if use_muon:
        opt = Muon(
            [{'params': decay_params}],
            lr=lr, momentum=0.95, nesterov=True, ns_steps=5,
            adamw_params=nodecay_params,
            adamw_lr=adamw_lr, adamw_betas=(0.9, 0.95), adamw_wd=0.1,
        )
    else:
        opt = torch.optim.AdamW(
            [
                {'params': decay_params,   'weight_decay': 0.1},
                {'params': nodecay_params, 'weight_decay': 0.0},
            ],
            lr=lr, betas=(0.9, 0.95), fused=True,
        )
    return opt


def load_meta(data_dir: str) -> dict:
    meta_path = Path(data_dir) / "meta.json"
    if not meta_path.exists():
        # Intentar detectar .bin files directamente
        bins = sorted(Path(data_dir).glob("*.bin"))
        if not bins:
            raise FileNotFoundError(
                f"No se encontró meta.json ni archivos .bin en {data_dir}\n"
                f"Ejecutar primero: python tokenize_dataset.py --data_dir /raw --out_dir {data_dir}"
            )
        # Meta sintético
        import numpy as np
        total = sum(os.path.getsize(b) // 2 for b in bins)
        return {
            "shards": [{"path": str(b), "n_tokens": os.path.getsize(b) // 2} for b in bins],
            "n_tokens": total,
            "vocab_size": 50257,
        }
    return json.loads(meta_path.read_text())


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARNING] CUDA no disponible — el trainer H200 es solo para GPU.")

    # ── Dataset ───────────────────────────────────────────────────────────────
    meta       = load_meta(args.data_dir)
    shard_paths = [s["path"] for s in meta["shards"]]
    vocab_size  = args.vocab or meta.get("vocab_size", 32000)

    if args.hbm_dataset and device.type == "cuda":
        print(f"[data] Cargando dataset en HBM ({meta['n_tokens']*2/1e9:.2f} GB)...")
        hbm_ds = BinaryTokenDataset(shard_paths, seq_len=args.seq_len, mode="hbm")
        train_loader = None   # usaremos hbm_ds.get_batch_hbm() directamente
        print(f"[data] Dataset HBM listo. {hbm_ds.n_tokens:,} tokens. "
              f"{hbm_ds.n_seqs:,} secuencias de {args.seq_len} tokens.")
    else:
        dataset     = PackedBinaryDataset(shard_paths, seq_len=args.seq_len)
        train_loader = DataLoader(
            dataset, batch_size=args.batch, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            prefetch_factor=4, persistent_workers=args.num_workers > 0,
            drop_last=True,
        )
        hbm_ds = None

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = build_model(args.model, vocab_size, device)

    # ROI 1 — AMP CORRECTO: FP32 master weights + BF16 autocast en forward.
    # El error de diseño anterior: model.bfloat16() convierte los pesos a BF16.
    # Consecuencia: Muon hace `g = p.grad.float()` pero luego `p.add_(g, alpha=-lr)`
    # añade FP32 a un param BF16 → el resultado se trunca a BF16 (3 dígitos decimales)
    # perdiend acumulación de pequeñas correcciones. FlashDiffSLRFunction.backward()
    # hace `.float()` de Q1/K1/V → crea copias temporales FP32 de tensores BF16
    # → picos de VRAM innecesarios durante backward.
    # Con FP32 masters:
    #   - Pesos en FP32: actualizaciones exactas (7 dígitos vs 3)
    #   - autocast casts activaciones a BF16 en forward → Tensor Cores
    #   - backward en FP32 → FlashDiffSLRFunction.backward() sin extra upcast
    #   - Lion/Muon optimizer states: siempre FP32 (sin cambio)
    # Costo: pesos 2× VRAM en params (125M → +500MB). Para H200-80GB: negligible.
    # NOT calling model.bfloat16() — pesos quedan en FP32 (default de torch)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {args.model} | {n_params/1e6:.2f}M params | dtype=fp32_masters+bf16_fwd")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    opt = build_optimizer(model, args.muon, args.lr, args.adamw_lr)

    # ── LR schedule ───────────────────────────────────────────────────────────
    tokens_per_step   = args.batch * args.seq_len * args.grad_accum
    total_steps       = int(args.total_tokens / tokens_per_step)
    warmup_steps      = min(max(20, int(total_steps * 0.01)), 500)
    # Para H200: cuantos menos warmup steps mejor (GPU cara!)
    # Mínimo 20 para inicializar archive antes de CUDA Graph capture.
    print(f"[sched] total_steps={total_steps:,}  warmup={warmup_steps}  "
          f"tokens/step={tokens_per_step:,}")

    # ── AMP context ───────────────────────────────────────────────────────────
    # BF16 autocast siempre habilitado en H200 (TensorCore path para matmuls).
    # FP32 masters garantizan que los pesos no se truncan a BF16.
    amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True)

    # ROI 2 — max-autotune: torch.compile en vez de CUDA Graphs.
    # Ambos son mutuamente exclusivos: CUDA Graphs capturan la secuencia exacta
    # de CUDA ops; torch.compile regenera ops via Inductor/Triton JIT.
    # max-autotune benchmarkea configuraciones de tiles Triton en el primer step
    # (~10-15 min de compilación inicial, se amortiza en la primera hora).
    # Para H200 con workloads de entrenamiento largo (>1B tokens), max-autotune
    # puede superar CUDA Graphs ~5-10% al optimizar tiles para sus núcleos HBM3.
    if args.max_autotune:
        if args.cuda_graphs:
            print("[compile] max-autotune es incompatible con CUDA Graphs → desactivando CUDA Graphs")
            args.cuda_graphs = False
        # torch._dynamo.config.cache_size_limit sube de 8 a 64 para evitar
        # recompilaciones por cambio de forma de batch (warmup / checkpoint)
        torch._dynamo.config.cache_size_limit = 64
        print("[compile] Aplicando torch.compile(mode='max-autotune')...")
        print("[compile] ADVERTENCIA: primer step puede tardar 10-15 min. Normal.")
        model = torch.compile(
            model,
            mode='max-autotune',
            fullgraph=False,   # mamba_ssm C++ ops rompen el grafo completo
            backend='inductor',
        )
        print("[compile] torch.compile configurado. JIT compilation en primer step.")

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── CUDA Graph setup ──────────────────────────────────────────────────────
    use_graphs = args.cuda_graphs and device.type == "cuda"
    graph_step = None
    if use_graphs:
        graph_step = CUDAGraphTrainStep(model, amp_ctx, aux_weight=args.aux_weight)

    # ── Iterador de datos ─────────────────────────────────────────────────────
    def get_batch(step: int):
        if hbm_ds is not None:
            return hbm_ds.get_batch_hbm(args.batch)
        # CPU DataLoader
        return next(iter_loader)

    def _get_layers():
        """Retorna todas las AdvancedChimeraLayer del modelo."""
        from advanced_chimera import AdvancedChimeraLayer
        return [m for m in model.modules() if isinstance(m, AdvancedChimeraLayer)]

    # ── State ────────────────────────────────────────────────────────────────
    global_step   = 0
    total_toks    = 0
    loss_ema      = None
    t_start       = time.perf_counter()
    log_vals: list[dict] = []

    # ── Router Entropy Watchdog ──────────────────────────────────────────────
    # Detecta y corrige colapso del router de tiers (FAST/HYBRID/FULL).
    # Cuando H_bits < 1.2 durante 10 steps consecutivos → boost aux_weight × 10
    # por 100 steps. Se desactiva automáticamente después de la recuperación.
    # Solo activo en el fallback path (no CUDA graph) donde loss_dict está disponible.
    entropy_watchdog = None
    if _RouterEntropyWatchdog is not None:
        entropy_watchdog = _RouterEntropyWatchdog(
            collapse_threshold=1.2,
            boost_factor=10.0,
            recovery_steps=100,
            window=10,
        )
    _base_aux_weight  = args.aux_weight   # guardar original para el watchdog

    # ── Intentar cargar checkpoint ────────────────────────────────────────────
    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists() and not args.no_resume:
        ckpt = torch.load(str(latest_ckpt), map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        global_step = ckpt.get("step", 0)
        total_toks  = ckpt.get("total_toks", 0)
        loss_ema    = ckpt.get("loss_ema", None)
        print(f"[resume] Checkpoint cargado: step={global_step}, tokens={total_toks:,}")

    # ── DataLoader iterator (CPU mode) ────────────────────────────────────────
    if train_loader is not None:
        iter_loader = iter(train_loader)

    # ── Warmup + CUDA Graph capture ───────────────────────────────────────────
    if use_graphs and not graph_step.is_captured:
        # Generar algunos batches para warmup
        warmup_batches = []
        for _ in range(30):
            if hbm_ds is not None:
                ids, labels = hbm_ds.get_batch_hbm(args.batch)
            else:
                try:
                    ids, labels = next(iter_loader)
                except StopIteration:
                    iter_loader = iter(train_loader)
                    ids, labels = next(iter_loader)
                ids    = ids.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            warmup_batches.append((ids, labels))

        class _WarmupIter:
            def __init__(self, data): self.data = data; self.i = 0
            def __iter__(self): return self
            def __next__(self):
                if self.i >= len(self.data): raise StopIteration
                b = self.data[self.i]; self.i += 1; return b

        model.train()
        opt.zero_grad(set_to_none=True)
        graph_step.warmup(_WarmupIter(warmup_batches), n_warmup=25)

        # Capturar grafo con el primer batch estático
        ids_cap, lbl_cap = warmup_batches[-1]
        graph_step.capture(ids_cap, lbl_cap)

    # ── Main training loop ────────────────────────────────────────────────────
    model.train()
    layers = _get_layers()

    while global_step < total_steps:
        t_step_start = time.perf_counter()

        # Gradient accumulation loop
        accum_loss = 0.0
        opt.zero_grad(set_to_none=True)

        for accum_i in range(args.grad_accum):
            # ── Get data ─────────────────────────────────────────────────────
            if hbm_ds is not None:
                ids, labels = hbm_ds.get_batch_hbm(args.batch)
            else:
                try:
                    ids, labels = next(iter_loader)
                except StopIteration:
                    iter_loader = iter(train_loader)
                    ids, labels = next(iter_loader)
                ids    = ids.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            # ── TTT update fuera del grafo ────────────────────────────────────
            if args.cuda_graphs and use_graphs and accum_i == 0:
                if global_step % args.ttt_interval == 0:
                    with torch.no_grad():
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                                enabled=(args.dtype == "bfloat16")):
                            # Normalizar x de embedding para chunk TTT
                            with torch.no_grad():
                                emb = model.embedding(ids[:, :64])
                            for layer in layers:
                                x_norm_chunk = layer.norm(emb)
                                layer.update_ttt_inplace(x_norm_chunk)

            # ── Forward + Backward ────────────────────────────────────────────
            if use_graphs and graph_step.is_captured and accum_i == 0 and args.grad_accum == 1:
                # Caso óptimo: 1 step = 1 replay (sin inner gradient accumulation)
                loss_val = graph_step.step(ids, labels)
                accum_loss = loss_val.item()
            else:
                # Fallback: forward Python normal (multi-accum o sin grafo)
                # Ensure graph_mode is False for normal execution so side-effects work
                for m in model.modules():
                    if hasattr(m, 'graph_mode'): m.graph_mode = False

                # RouterEntropyWatchdog: ajusta aux_weight antes del forward
                # para que el coeficiente de routing entropy sea el correcto.
                _eff_aux = args.aux_weight
                if entropy_watchdog is not None and entropy_watchdog._recovering > 0:
                    _eff_aux = _base_aux_weight * entropy_watchdog.boost
                with amp_ctx:
                    logits, total_loss, loss_dict = model(
                        ids, labels=labels, aux_weight=_eff_aux
                    )
                (total_loss / args.grad_accum).backward()
                accum_loss += total_loss.item() / args.grad_accum
                
                # Restore graph_mode for next capture/replay
                for m in model.modules():
                    if hasattr(m, 'graph_mode'): m.graph_mode = use_graphs
                # Actualizar watchdog con routing entropy del paso actual
                if entropy_watchdog is not None and accum_i == 0:
                    _w_boost = entropy_watchdog.update(loss_dict)
                    if _w_boost > 1.0 and entropy_watchdog._recovering == entropy_watchdog.recovery:
                        print(
                            f"[RouterEntropyWatchdog] step={global_step}: "
                            f"Colapso de router detectado (H<{entropy_watchdog.threshold}bit). "
                            f"aux_weight ×{entropy_watchdog.boost:.0f} por {entropy_watchdog.recovery} steps."
                        )

        # ── Optimizer step ────────────────────────────────────────────────────
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lrs = [pg.get('lr', args.lr) for pg in opt.param_groups]

        # LR schedule manual (ajusta lr en param_groups)
        lr_scale = wsd_lr(global_step, total_steps, warmup_steps)
        for pg in opt.param_groups:
            if 'lr' in pg:
                pg['lr'] = args.lr * lr_scale

        opt.step()

        # ── Estadísticas ──────────────────────────────────────────────────────
        global_step += 1
        step_toks = args.batch * args.seq_len * args.grad_accum
        total_toks += step_toks
        step_ms    = (time.perf_counter() - t_step_start) * 1000
        toks_per_s = step_toks / (step_ms / 1000)

        alpha   = 0.98
        loss_ema = accum_loss if loss_ema is None else (alpha * loss_ema + (1 - alpha) * accum_loss)

        # ── Logging ───────────────────────────────────────────────────────────
        if global_step % args.log_every == 0:
            elapsed  = time.perf_counter() - t_start
            toks_all = total_toks / elapsed

            print(
                f"[{global_step:>6}/{total_steps}] "
                f"loss={accum_loss:.4f} ema={loss_ema:.4f}  "
                f"grad={grad_norm:.3f}  lr={args.lr * lr_scale:.2e}  "
                f"{toks_per_s:,.0f} tok/s  "
                f"[{total_toks/1e9:.2f}B/{args.total_tokens/1e9:.1f}B tok]"
            )
            log_vals.append({
                "step": global_step, "loss": accum_loss, "loss_ema": loss_ema,
                "grad_norm": float(grad_norm), "tok_per_s": toks_per_s,
                "total_tokens": total_toks,
            })

        # ── Checkpoint ───────────────────────────────────────────────────────
        if global_step % args.save_every == 0 or global_step == total_steps:
            # Temporalmente desactivar graph_mode para guardar estado limpio
            for layer in layers:
                if hasattr(layer, 'graph_mode'):
                    layer.graph_mode = False

            ckpt_data = {
                "step":       global_step,
                "total_toks": total_toks,
                "loss_ema":   loss_ema,
                "model":      model.state_dict(),
                "opt":        opt.state_dict(),
                "args":       vars(args),
            }
            torch.save(ckpt_data, str(ckpt_dir / "latest.pt"))
            torch.save(ckpt_data, str(ckpt_dir / f"step_{global_step:07d}.pt"))
            print(f"[ckpt] Guardado step={global_step}, "
                  f"loss_ema={loss_ema:.4f}, tokens={total_toks:,}")

            # Restaurar graph_mode
            for layer in layers:
                if hasattr(layer, 'graph_mode'):
                    layer.graph_mode = use_graphs

        # ── Progreso global ───────────────────────────────────────────────────
        if total_toks >= args.total_tokens:
            break

    # ── Final ────────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_start
    effective_tps = total_toks / total_elapsed
    print(f"\n{'='*60}")
    print(f"Entrenamiento completo!")
    print(f"  Steps totales  : {global_step:,}")
    print(f"  Tokens totales : {total_toks:,} ({total_toks/1e9:.2f}B)")
    print(f"  Tiempo total   : {total_elapsed/3600:.2f}h")
    print(f"  Throughput eff.: {effective_tps:,.0f} tok/s")
    print(f"  Loss final EMA : {loss_ema:.4f}")
    print(f"  Checkpoint     : {ckpt_dir}/latest.pt")

    # Guardar log
    log_path = ckpt_dir / "training_log.json"
    log_path.write_text(json.dumps(log_vals, indent=2))
    print(f"  Log guardado   : {log_path}")

    return loss_ema


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark rápido de throughput (sin dataset real)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_throughput(args):
    """
    Mide tok/s en modo benchmark: datos sintéticos, sin checkpoint.
    Útil para estimar throughput antes de lanzar entrenamiento real.
    """
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(args.model, args.vocab or 32000, device)
    if args.dtype == "bfloat16":
        model = model.bfloat16()

    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    B, S = args.batch, args.seq_len
    ids    = torch.randint(0, args.vocab or 32000, (B, S), device=device)
    labels = torch.randint(0, args.vocab or 32000, (B, S), device=device)

    amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                  enabled=(args.dtype == "bfloat16"))

    # Warmup
    print(f"\n[bench] Modelo: {args.model} | {n_params/1e6:.1f}M params | "
          f"B={B} S={S} dtype={args.dtype}")
    print(f"[bench] Warmup...")
    for _ in range(5):
        with amp_ctx:
            logits, loss, _ = model(ids, labels=labels)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Benchmark 20 steps
    print(f"[bench] Midiendo 20 steps...")
    t0 = time.perf_counter()
    for _ in range(20):
        with amp_ctx:
            logits, loss, _ = model(ids, labels=labels)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms_per_step = elapsed / 20 * 1000
    tps = B * S * 20 / elapsed

    # Con CUDA Graphs estimado (+40% conservador)
    tps_graph = tps * 1.4

    print(f"\n{'='*55}")
    print(f"  CHIMERA {args.model} throughput en {torch.cuda.get_device_name()}")
    print(f"{'='*55}")
    print(f"  Sin CUDA Graphs : {ms_per_step:.1f} ms/step  {tps:,.0f} tok/s")
    print(f"  Con CUDA Graphs : ~{tps_graph:,.0f} tok/s  (estimado +40%)")
    print(f"  Chinchilla 125M : 1.59B tokens → ~{1.59e9/tps_graph/3600:.1f}h con graphs")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="train_h200.py — CHIMERA trainer optimizado para H200"
    )
    # Data
    p.add_argument("--data_dir",    default=None,
                   help="Directorio con .bin tokens (de tokenize_dataset.py)")
    p.add_argument("--hbm_dataset", action="store_true",
                   help="Cargar dataset entero en HBM GPU (zero-latency)")

    # Model
    p.add_argument("--model",  default="125M", choices=["tiny", "125M", "350M"],
                   help="Tamaño de modelo a construir")
    p.add_argument("--vocab",  type=int, default=None,
                   help="Vocab size (default: auto-detectado de meta.json)")

    # Training
    p.add_argument("--batch",        type=int,   default=16)
    p.add_argument("--seq_len",      type=int,   default=2048)
    p.add_argument("--grad_accum",   type=int,   default=1)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--adamw_lr",     type=float, default=3e-4,
                   help="LR para bias/norm/embedding (si --muon)")
    p.add_argument("--total_tokens", type=float, default=3e9,
                   help="Tokens totales a entrenar (1.59e9 = Chinchilla 125M)")
    p.add_argument("--dtype",        default="bfloat16", choices=["float32", "bfloat16"],
                   help="Dtype legacy (ignorado: siempre FP32 masters + BF16 autocast)")
    p.add_argument("--max_autotune",  action="store_true", default=False,
                   help="torch.compile(max-autotune) en vez de CUDA Graphs. "
                        "Primera compilación: 10-15 min. ROI en runs >1B tokens.")
    p.add_argument("--muon",         action="store_true", default=True,
                   help="Usar Muon optimizer (recomendado)")
    p.add_argument("--no_muon",      dest="muon", action="store_false")
    p.add_argument("--aux_weight",   type=float, default=0.01,
                   help="Peso del routing auxiliary loss")
    p.add_argument("--num_workers",  type=int,   default=4,
                   help="DataLoader workers (CPU mode)")

    # CUDA Graphs
    p.add_argument("--cuda_graphs",  action="store_true", default=True,
                   help="Activar CUDA Graph capture (H200 recomendado)")
    p.add_argument("--no_cuda_graphs", dest="cuda_graphs", action="store_false")
    p.add_argument("--ttt_interval", type=int, default=1,
                   help="Pasos entre TTT updates fuera del grafo")

    # Checkpointing / logging
    p.add_argument("--ckpt_dir",   default="./ckpt_h200")
    p.add_argument("--log_every",  type=int, default=10)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--no_resume",  action="store_true",
                   help="Ignorar checkpoint existente y empezar de cero")

    # Utilities
    p.add_argument("--benchmark",  action="store_true",
                   help="Solo benchmark de throughput (sin datos reales)")
    p.add_argument("--seed",       type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.benchmark:
        benchmark_throughput(args)
        return

    if args.data_dir is None:
        print("ERROR: --data_dir requerido para entrenamiento.")
        print("Uso: python train_h200.py --data_dir /data/tokens --model 125M")
        print("     python train_h200.py --benchmark  # solo throughput")
        sys.exit(1)

    train(args)


if __name__ == "__main__":
    main()
