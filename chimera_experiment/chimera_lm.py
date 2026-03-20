"""
chimera_lm.py — Modelo de Lenguaje completo sobre CHIMERA
==========================================================
Encapsula:
  • ChimeraStack  : N capas AdvancedChimeraLayer con bus-cache threading
                    + gradient checkpointing selectivo por capa (cada 2 capas)
  • ChimeraLM     : Embedding → ChimeraStack → RMSNorm → LM-head
                    Weight-tying embedding↔lm_head (ahorra ~24M parámetros en 125M)

Técnicas de velocidad integradas en el modelo:
  1. Weight-tying            → reduce parámetros, mejora convergencia
  2. RMSNorm (no LayerNorm)  → sin resta de media, ~20% más rápido
  3. Residual-scale init     → 1/√(2·L) en proj de salida por capa (GPT-2)
  4. Selective grad-ckpt     → compatible con bus-cache (use_reentrant=False)

Uso:
    cfg = ChimeraConfig(d_model=768, n_layers=12)   # 125M preset
    lm  = ChimeraLM(cfg, vocab_size=32000).cuda().bfloat16()
    logits, loss = lm(input_ids, labels=labels)     # loss=None si labels=None
"""

from __future__ import annotations

import math
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from chimera_config   import ChimeraConfig
from advanced_chimera import AdvancedChimeraLayer
from chimera_losses   import ChimeraRoutingLoss


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# RMSNorm: usa la implementación fusionada de PyTorch (torch.nn.RMSNorm).
# En PyTorch ≥ 2.1, usa un kernel CUDA fused (2 passes HBM, not 4+):
#   pass1: compute RMS → pass2: scale. ~30-40% más rápido que implementación manual.
# Alias para backward compat con código que importe RMSNorm de este módulo.
RMSNorm = nn.RMSNorm


# ─────────────────────────────────────────────────────────────────────────────
# Stack de capas con bus-cache threading
# ─────────────────────────────────────────────────────────────────────────────

class ChimeraStack(nn.Module):
    """
    Stack de N AdvancedChimeraLayer con:
      • Threading del bus_cache entre capas
      • Gradient checkpointing selectivo (cada `ckpt_interval` capas)
      • Recolección de aux_dicts para pérdidas de routing y TTT

    Parámetro ckpt_interval:
      - 1   → checkpointing en todas las capas (máximo ahorro VRAM, +30% tiempo)
      - 2   → cada 2 capas   (equilibrio recomendado: ~40% ahorro VRAM, +15% tiempo)
      - 999 → sin checkpointing (default — necesario mientras AdvancedChimeraLayer
              tenga mutaciones in-place de estado: dt_momentum (Lion), archive.maybe_archive.
              Torch checkpoint re-ejecuta el forward y detecta tensores intermedios distintos
              → CheckpointError. Solución: usar grad_accum para ampliar batch efectivo.
              Una vez AdvancedChimeraLayer sea stateless, ckpt_interval=2 funcionará.)
    """    

    def __init__(self, config: ChimeraConfig, ckpt_interval: int = 999):
        super().__init__()
        self.ckpt_interval = ckpt_interval

        self.layers = nn.ModuleList([
            AdvancedChimeraLayer(
                d_model   = config.d_model,
                expand    = config.expand,
                headdim   = config.headdim,
                layer_idx = i,
                d_state   = config.d_state,
                bus_dim   = config.bus_dim,
                landmark_dim    = config.landmark_dim,
                max_landmarks   = config.max_landmarks,
                ttt_err_threshold = config.ttt_err_threshold,
                sdtm_n_heads    = config.sdtm_n_heads,
                sdtm_d_mem      = config.sdtm_d_mem,
            )
            for i in range(config.n_layers)
        ])
        self.n_layers = config.n_layers

        # Residual-scaling GPT-2 style: escala la proj de salida de Mamba2 por
        # 1/√(2·n_layers) para evitar que la varianza de residuals crezca con L.
        # Se aplica al parámetro out_proj directamente (in-place, no overhead).
        if config.residual_scale:
            scale = 1.0 / math.sqrt(2.0 * config.n_layers)
            for layer in self.layers:
                if hasattr(layer.mamba2, 'out_proj'):
                    with torch.no_grad():
                        layer.mamba2.out_proj.weight.mul_(scale)

    # ── Forward con checkpointing ──────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        collect_aux: bool = True,
        initial_bus_cache: torch.Tensor = None,
    ):
        """
        x:           [B, S, D]
        collect_aux: si True, acumula aux_dicts de cada capa para routing loss.
        initial_bus_cache: bus_cache heredado de un chunk anterior (para chunked training).

        Returns:
            x:        [B, S, D]          — salida tras todas las capas
            aux_list: list[dict] | []    — aux_dicts por capa (vacío si collect_aux=False)
        """
        bus_cache = initial_bus_cache
        aux_list  = [] if collect_aux else None

        for i, layer in enumerate(self.layers):
            use_ckpt = (
                self.training
                and self.ckpt_interval < 999
                and (i % self.ckpt_interval == 0)
            )

            if use_ckpt:
                # use_reentrant=False: soporta non-tensor args (bus_cache puede ser None)
                # y es compatible con torch.compile.
                x, bus_cache, aux = checkpoint(
                    self._layer_fn,
                    layer, x, bus_cache,
                    use_reentrant=False,
                )
            else:
                x, bus_cache, aux = layer(x, bus_cache=bus_cache, return_aux=True)

            if collect_aux and aux is not None:
                aux_list.append(aux)

        return x, aux_list if collect_aux else []

    def forward_with_carry(
        self,
        x: torch.Tensor,
        collect_aux: bool = True,
        initial_bus_cache: torch.Tensor = None,
    ):
        """
        Forward con carry de bus_cache — expone el bus_cache final para
        transmitirlo al siguiente chunk en chunked training.

        Returns:
            x:              [B, S, D]
            aux_list:       list[dict] | []
            final_bus_cache: tensor del bus (detachado para carry inter-chunk)
        """
        bus_cache = initial_bus_cache
        aux_list  = [] if collect_aux else None

        for i, layer in enumerate(self.layers):
            x, bus_cache, aux = layer(x, bus_cache=bus_cache, return_aux=True)
            if collect_aux and aux is not None:
                aux_list.append(aux)

        final_bus = bus_cache.detach() if bus_cache is not None else None
        return x, (aux_list if collect_aux else []), final_bus

    @staticmethod
    def _layer_fn(layer, x, bus_cache):
        """Wrapper estático para checkpoint (no captura nada del scope exterior)."""
        return layer(x, bus_cache=bus_cache, return_aux=True)

    # ── Inference: cache allocation + incremental step ──────────────────────

    def allocate_inference_cache(self, batch_size, dtype=None, ring_size=None):
        """Allocate per-layer caches for incremental decode."""
        return [layer.allocate_inference_cache(batch_size, dtype=dtype, ring_size=ring_size)
                for layer in self.layers]

    def step(self, x_tok, caches):
        """
        Single-token decode through all layers.
        x_tok: [B, 1, D], caches: list[dict] from allocate_inference_cache.
        Returns: (x_tok [B, 1, D], caches)
        """
        for i, layer in enumerate(self.layers):
            x_tok, caches[i] = layer.step(x_tok, caches[i])
        return x_tok, caches


# ─────────────────────────────────────────────────────────────────────────────
# ChimeraLM — Modelo de Lenguaje completo
# ─────────────────────────────────────────────────────────────────────────────

class ChimeraLM(nn.Module):
    """
    Modelo de Lenguaje basado en CHIMERA.

    Arquitectura:
        input_ids [B, S]
        → Embedding [B, S, D]
        → ChimeraStack (N capas)
        → RMSNorm
        → LM-head [B, S, vocab_size]   (weight-tied con embedding)
        → CrossEntropyLoss (si labels!=None)

    Parámetros:
        config:     ChimeraConfig con hiperparámetros del modelo
        vocab_size: tamaño del vocabulario (GPT-NeoX=50277, LLaMA=32000)
        tie_weights:True → weight-tying embedding↔lm_head (default True)
        ckpt_interval: ver ChimeraStack (default 2)
    """

    def __init__(
        self,
        config:       ChimeraConfig,
        vocab_size:   int  = 32000,
        tie_weights:  bool = True,
        ckpt_interval: int = 999,
    ):
        super().__init__()
        self.config     = config
        self.vocab_size = vocab_size
        self.d_model    = config.d_model

        # ── Embedding ─────────────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        # Inicialización estándar: std = 1/√(d_model) — más suave que default N(0,1)
        nn.init.normal_(self.embedding.weight, std=config.d_model ** -0.5)

        # ── Stack de capas CHIMERA ─────────────────────────────────────────────
        self.stack = ChimeraStack(config, ckpt_interval=ckpt_interval)

        # ── Norm final ────────────────────────────────────────────────────────
        # nn.RMSNorm: fusionado en PyTorch ≥ 2.1 (2 passes HBM, not 4+)
        self.norm_f = nn.RMSNorm(config.d_model, eps=1e-6)

        # ── LM-head (sin bias) ────────────────────────────────────────────────
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

        # ── Weight-tying ──────────────────────────────────────────────────────
        # lm_head.weight y embedding.weight comparten tensor → gradientes se suman.
        # Efecto: ~24M parámetros menos en 125M (d=768, vocab=32k).
        # Paper: "Using the Output Embedding to Improve Language Models" (Press & Wolf 2017)
        if tie_weights:
            self.lm_head.weight = self.embedding.weight

        # ── Routing loss handler ───────────────────────────────────────────────
        self.routing_loss = ChimeraRoutingLoss(
            entropy_weight      = config.routing_entropy_weight,
            supervision_weight  = config.routing_supervision_weight,
            balance_weight      = config.routing_balance_weight,
            z_loss_weight       = 1e-3,
            target_entropy_frac = config.routing_target_entropy,
            min_tier_prob       = config.routing_min_tier_prob,
        )

    # ── Count params ──────────────────────────────────────────────────────────

    def num_parameters(self, count_embedding: bool = True) -> int:
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if not count_embedding:
            # No contar el embedding dos veces si hay weight-tying
            params -= self.embedding.weight.numel()
        return params

    # ── Inference: incremental decode ──────────────────────────────────────────

    def allocate_inference_cache(self, batch_size, dtype=None, ring_size=None):
        """Allocate per-layer caches for incremental decode."""
        return self.stack.allocate_inference_cache(batch_size, dtype=dtype, ring_size=ring_size)

    # ── Training: torch.compile ────────────────────────────────────────────────

    def compile_for_training(self, mode='default', backend='inductor'):
        """
        Aplica torch.compile al stack para reducir overhead de Python dispatch.

        El profiling (test_triton_perf + profile_chimera_components) mostró que
        97% del tiempo per-layer es orquestación Python. torch.compile fusiona
        las operaciones pequeñas, elimina dispatch overhead, y aplica optimizaciones
        del backend Inductor (operator fusion, memory planning, etc.).

        Pipeline de optimización:
          1. Set graph_mode=True en todas las capas → elimina graph breaks de:
             - TTT-Lite (autograd.grad dentro de forward)
             - Archive (maybe_archive con .item() y mutations)
             - SDTM write (compute_write con .item())
             - Aux dict (.item() calls)
          2. Pre-computa archive KV cache para retrieve compilable
          3. Compila ChimeraStack con torch.compile

        Nota sobre mode:
          - 'default': Inductor code generation, sin CUDA graphs. Estable con
            custom autograd.Function y Triton kernels (graph breaks tolerados).
          - 'reduce-overhead': Inductor + CUDA graphs. Más rápido si NO hay
            graph breaks dinámicos (.item() sobre valores cambiantes).
            Requiere que FlashDiffSLR.forward() no use .item() en lam.
          - 'max-autotune': Más tiempo de compilación, máximo throughput.

        El trainer debe llamar post-forward:
          - model.post_compile_step()   # TTT + Archive + SDTM updates

        Warmup:
          Con fullgraph=False, dynamo crea ~16+ subgrafos (4 capas × ~4 graph
          breaks en mamba2/SLR/etc). Cada subgrafo se compila lazily via Inductor.
          Se necesitan ~15 training steps de warmup antes de steady-state.
          Steady-state típico: ~2.2× speedup en throughput (tok/s).

        Args:
            mode: 'default' | 'reduce-overhead' | 'max-autotune'
            backend: 'inductor' (default) | 'eager' (debug)
        Returns: self (para encadenar)
        """
        import torch._dynamo

        # Configuración de dynamo
        torch._dynamo.config.suppress_errors = True          # fallback silencioso
        torch._dynamo.config.cache_size_limit = 64           # más cache para variaciones
        torch._dynamo.config.optimize_ddp = True             # compatible con DDP

        # 1. Activar graph_mode en todas las capas
        for layer in self.stack.layers:
            layer.set_graph_mode(True)

        # 2. Pre-computar archive cache para retrieve compilable
        self.precompute_archive_caches()

        # 3. Compilar el stack completo
        self.stack = torch.compile(
            self.stack,
            mode=mode,
            fullgraph=False,     # permite graph breaks residuales
            backend=backend,
            dynamic=False,       # shapes estáticas → más optimizaciones
        )

        self._compiled = True
        return self

    def precompute_archive_caches(self):
        """Pre-computa KV cache de archive en todas las capas para retrieve compilable."""
        for layer in self.stack.layers:
            if hasattr(layer, 'archive'):
                layer.archive.precompute_retrieve_cache()

    def post_compile_step(self):
        """
        Operaciones post-forward para training con compile.
        Ejecuta las operaciones diferidas que causan graph breaks.
        Llamar DESPUÉS de loss.backward() + optimizer.step().
        """
        for layer in self.stack.layers:
            layer.update_ttt_inplace()
            layer.archive_deferred()
            layer.update_sdtm_inplace()
        # Re-computar archive caches para el próximo forward
        self.precompute_archive_caches()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,          # [B, S]
        labels:    torch.Tensor | None = None,  # [B, S]  (optional)
        aux_weight: float = 0.01,         # peso de pérdidas auxiliares vs LM-loss
    ):
        """
        Returns:
            Si labels=None:
                logits [B, S, vocab_size]
            Si labels es Tensor:
                (logits, total_loss, loss_dict)
                loss_dict = {'lm': ..., 'routing': ..., 'total': ...}

        El loss retornado es `lm_loss + aux_weight * routing_loss`.
        """
        B, S = input_ids.shape

        # Embedding
        x = self.embedding(input_ids)     # [B, S, D]

        # Stack (con checkpointing selectivo + colección de aux_dicts)
        x, aux_list = self.stack(x, collect_aux=(labels is not None))

        # Norm final
        x = self.norm_f(x)               # [B, S, D]

        # LM-head: proyección a vocab
        # Solo computar logits completos si los necesitamos todos para la loss.
        # En inference podemos usar solo el último token (optimización futura).
        logits = self.lm_head(x)         # [B, S, vocab_size]

        if labels is None:
            return logits

        # ── Cross-Entropy LM loss ──────────────────────────────────────────────
        # Shift: predecir token[i+1] desde token[i]
        # logits[:, :-1]  → [B, S-1, V]   labels[:, 1:]  → [B, S-1]
        shift_logits = logits[:, :-1].contiguous().view(-1, self.vocab_size)
        shift_labels = labels[:, 1:].contiguous().view(-1)

        # ignore_index=-100 por convención HuggingFace (tokens enmascarados)
        lm_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        # ── Routing loss (Z-loss + entropy + balance) ─────────────────────────
        routing_total = torch.tensor(0.0, device=input_ids.device)
        routing_info  = {}
        if aux_list:
            # Promediar pérdidas de routing sobre todas las capas
            layer_routing_losses = []
            for aux in aux_list:
                if aux is not None and 'router_logits' in aux:
                    rl, rinfo = self.routing_loss(aux)
                    layer_routing_losses.append(rl)
                    routing_info = rinfo   # último (para logging)
            if layer_routing_losses:
                routing_total = torch.stack(layer_routing_losses).mean()

        total_loss = lm_loss + aux_weight * routing_total

        loss_dict = {
            'lm':      lm_loss.item(),
            'routing': routing_total.item() if routing_total.requires_grad else float(routing_total),
            'total':   total_loss.item(),
            **{k: v for k, v in routing_info.items()},
        }

        return logits, total_loss, loss_dict

    # ── Generación autoregresiva (incremental) ──────────────────────────────────

    @torch.inference_mode()
    def _prefill(self, input_ids: torch.Tensor):
        """
        Prefill: forward completo del prompt capturando SSM states.
        Returns: (last_logits [B, 1, V], caches list[dict])
        """
        B, S = input_ids.shape
        x = self.embedding(input_ids)     # [B, S, D]

        # InferenceParams captura (conv_state, ssm_state) al final del forward.
        inf_params = None
        try:
            from mamba_ssm.utils.generation import InferenceParams
            inf_params = InferenceParams(
                max_seqlen=S, max_batch_size=B, seqlen_offset=0
            )
            for layer in self.stack.layers:
                conv_st, ssm_st = layer.mamba2.allocate_inference_cache(
                    B, max_seqlen=1, dtype=x.dtype
                )
                inf_params.key_value_memory_dict[layer.mamba2.layer_idx] = (
                    conv_st, ssm_st
                )
        except (ImportError, AttributeError):
            pass

        # Forward por capa (sin checkpointing, con captura de estados)
        bus_cache = None
        for layer in self.stack.layers:
            x, bus_cache, _ = layer(
                x, bus_cache=bus_cache, return_aux=True,
                inference_params=inf_params,
            )

        x = self.norm_f(x)
        logits = self.lm_head(x[:, -1:, :])   # [B, 1, V]

        # Construir caches para decode incremental
        dtype = next(self.parameters()).dtype
        caches = []
        for layer in self.stack.layers:
            cache = layer.allocate_inference_cache(B, dtype=dtype)
            if inf_params is not None:
                lidx = layer.mamba2.layer_idx
                states = inf_params.key_value_memory_dict.get(lidx)
                if states is not None:
                    cache['conv_state'].copy_(states[0].to(cache['conv_state'].dtype))
                    cache['ssm_state'].copy_(states[1].to(cache['ssm_state'].dtype))
            caches.append(cache)

        return logits, caches

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,    # [B, S_prompt]
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Generación incremental: prefill O(S) una vez + decode O(1) por token.

        Speedup vs versión anterior: ~S× (e.g., 256× para prompts de 256 tokens).
        """
        B, S = input_ids.shape

        # ── Phase 1: Prefill ─────────────────────────────────────────────────
        logits, caches = self._prefill(input_ids)

        # ── Phase 2: Decode incremental ──────────────────────────────────────
        generated = []
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]   # [B, V]

            if temperature != 1.0:
                next_logits = next_logits / temperature
            if top_k > 0:
                vals, _ = torch.topk(next_logits, top_k, dim=-1)
                threshold = vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, -1e9)

            probs    = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # [B, 1]
            generated.append(next_tok)

            # Step: embedding → stack layers → norm → lm_head
            x_tok = self.embedding(next_tok)            # [B, 1, D]
            x_tok, caches = self.stack.step(x_tok, caches)
            x_tok = self.norm_f(x_tok)
            logits = self.lm_head(x_tok)                # [B, 1, V]

        return torch.cat(generated, dim=-1)  # [B, max_new_tokens]


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de construcción
# ─────────────────────────────────────────────────────────────────────────────

def build_chimera_125M(vocab_size: int = 32000, **kwargs) -> ChimeraLM:
    """
    Construye CHIMERA-125M: d=768, L=12.
    ~125M parámetros (incluyendo embedding, sin contar weight-tying dos veces).
    """
    from chimera_config import ChimeraConfig
    cfg = ChimeraConfig(
        d_model   = 768,
        n_layers  = 12,
        expand    = 2,
        headdim   = 32,
        d_state   = 64,
        bus_dim   = 128,
        dtype     = "bfloat16",
        **kwargs,
    )
    return ChimeraLM(cfg, vocab_size=vocab_size)


def build_chimera_350M(vocab_size: int = 32000, **kwargs) -> ChimeraLM:
    cfg = ChimeraConfig(
        d_model  = 1024,
        n_layers = 24,
        expand   = 2,
        headdim  = 32,
        d_state  = 64,
        bus_dim  = 128,
        dtype    = "bfloat16",
        **kwargs,
    )
    return ChimeraLM(cfg, vocab_size=vocab_size)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test rápido
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=== ChimeraLM smoke test ===\n")
    torch.set_float32_matmul_precision('high')

    # Modelo tiny para test rápido
    cfg = ChimeraConfig(
        d_model  = 256,
        n_layers = 4,
        expand   = 2,
        headdim  = 32,
        d_state  = 64,
    )
    model = ChimeraLM(cfg, vocab_size=4096, ckpt_interval=999).cuda().bfloat16()
    total_p = model.num_parameters()
    print(f"  Parámetros totales: {total_p:,}  ({total_p/1e6:.2f}M)")
    print(f"  Weight-tying: {model.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr()}")

    B, S = 2, 512
    ids    = torch.randint(0, 4096, (B, S), device='cuda')
    labels = torch.randint(0, 4096, (B, S), device='cuda')

    # Warmup
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        _, loss, ld = model(ids, labels=labels)
    loss.backward()
    model.zero_grad()

    # Medición
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 5
    for _ in range(N):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss, ld = model(ids, labels=labels)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N * 1e3

    tps = B * S / (dt / 1e3)
    print(f"\n  Forward+Backward: {dt:.1f} ms  →  {tps:,.0f} tok/s  (BF16, grad-ckpt)")
    print(f"  LM loss: {ld['lm']:.4f}   routing: {ld['routing']:.5f}")
    print(f"  Total loss: {ld['total']:.4f}")
    print("\n  [OK] ChimeraLM listo para training.")
