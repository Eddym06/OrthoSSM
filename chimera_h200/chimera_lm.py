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
  4. Selective grad-ckpt     → compatible con bus-cache (use_reentrant=False, PyTorch ≥2.8)

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
# En PyTorch ≥ 2.1 está compilada en CUDA con 2 passes HBM (1 load + 1 store)
# frente a las 4-5 passes de la versión manual (cast+pow+mean+rsqrt+scale+cast).
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
      - 999 → sin checkpointing (usar si hay problemas de compatibilidad)

    NOTA: _skip_side_effects en AdvancedChimeraLayer evita que las mutaciones
    in-place (TTT-Lite, archive.maybe_archive) se repitan durante la
    recomputación de gradient checkpointing (use_reentrant=False).
    """    

    def __init__(self, config: ChimeraConfig, ckpt_interval: int = 2):
        super().__init__()
        self.ckpt_interval = ckpt_interval
        self.bus_dim       = config.bus_dim   # necesario para pre-alocar el ring buffer

        self.layers = nn.ModuleList([
            AdvancedChimeraLayer(
                d_model   = config.d_model,
                expand    = config.expand,
                headdim   = config.headdim,
                layer_idx = i,
                d_state   = config.d_state,
                use_spectral_vsa       = config.use_spectral_vsa,
                spectral_K             = config.spectral_K,
                spectral_K_min         = config.spectral_K_min,
                spectral_window        = config.spectral_window,
                spectral_ema_alpha     = config.spectral_ema_alpha,
                spectral_use_complex   = config.spectral_use_complex,
                spectral_n_retrieve    = config.spectral_n_retrieve,
                spectral_energy_threshold  = config.spectral_energy_threshold,
                spectral_lanczos_power_max = config.spectral_lanczos_power_max,
                spectral_disc_gamma    = config.spectral_disc_gamma,
                spectral_error_refresh = config.spectral_error_refresh,
                use_moe                = getattr(config, 'use_moe', False),
                moe_n_experts          = getattr(config, 'moe_n_experts', 8),
                moe_top_k              = getattr(config, 'moe_top_k', 2),
                moe_d_ff               = (getattr(config, 'moe_d_ff', 0) or None),
                use_cas                = config.use_cas,
                cas_n_experts          = config.cas_n_experts,
                cas_d_ff               = config.cas_d_ff if config.cas_d_ff > 0 else None,
                cas_tau_init           = config.cas_tau_init,
                cas_target_active      = config.cas_target_active,
                n_layers_total         = config.n_layers,
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
        labels: torch.Tensor | None = None,
        embedding: nn.Embedding | None = None,
    ):
        """
        x:           [B, S, D]
        collect_aux: si True, acumula aux_dicts de cada capa para routing loss.
        labels:      [B, S] — si proporcionado junto con embedding, computa SLR
                     aux loss inline y libera slr_out de cada capa inmediatamente.
                     Esto evita retener N×[B,S,D] tensores hasta el cómputo final.
        embedding:   nn.Embedding — tabla de embedding para target de SLR loss.

        Returns:
            x:            [B, S, D]          — salida tras todas las capas
            aux_list:     list[dict] | []    — aux_dicts por capa (sin slr_out si inline)
            slr_loss_sum: torch.Tensor       — pérdida SLR acumulada (0 si no aplica)
            n_slr_layers: int                — número de capas con SLR loss

        Bus ring buffer:
            Pre-alocado como [B, N_layers, bus_dim] de ceros al inicio del forward.
            Cada capa escribe su summary en el slot que le corresponde (write_idx=layer._layer_idx)
            y lee de los slots anteriores mediante causal masking en AsyncLightBus.forward_ring().
            Forma ESTÁTICA en toda la pila → compatible con CUDA Graphs y torch.compile.
        """
        B = x.shape[0]
        bus_ring = x.new_zeros(B, self.n_layers, self.bus_dim)
        aux_list = [] if collect_aux else None

        # SLR aux loss inline: computa por capa y libera slr_out inmediatamente.
        # Ahorro: evita retener N×[B,S,D] tensores hasta el final del forward.
        # Para N=23, D=768, S=2048, B=4: ~276MB liberados.
        _compute_slr_inline = (labels is not None and embedding is not None)
        slr_loss_sum = x.new_zeros(())
        n_slr_layers = 0
        if _compute_slr_inline:
            target_emb = embedding(labels[:, 1:]).detach()  # [B, S-1, D] — compute once

        # Reset recomputation counters for gradient checkpointing.
        # _fwd_count tracks how many times _layer_fn has been called per layer:
        #   count=0 → first forward (normal, with side effects)
        #   count≥1 → recomputation during backward (skip side effects)
        for layer in self.layers:
            layer._fwd_count = 0

        for i, layer in enumerate(self.layers):
            use_ckpt = (
                self.training
                and self.ckpt_interval < 999
                and (i % self.ckpt_interval == 0)
            )

            if use_ckpt:
                # Full-layer checkpoint: libera TODAS las activaciones, recomputa en backward.
                x, bus_ring, aux = checkpoint(
                    self._layer_fn,
                    layer, x, bus_ring,
                    use_reentrant=False,
                )
            else:
                # Selective Activation Checkpointing (SAC):
                # Solo el scan de Mamba2 se checkpointea internamente.
                # Las operaciones ligeras (router, bus, gating) conservan sus activaciones.
                # Trade-off: ~80% menos memoria que sin ckpt, ~20% menos recompute que full ckpt.
                if self.training and self.ckpt_interval < 999:
                    layer._selective_ckpt_mamba = True
                try:
                    x, bus_ring, aux = layer(x, bus_ring=bus_ring, return_aux=True)
                finally:
                    if self.training and self.ckpt_interval < 999:
                        layer._selective_ckpt_mamba = False

            if collect_aux and aux is not None:
                # SLR aux loss inline: compute per-layer, free slr_out immediately.
                if _compute_slr_inline and 'slr_out' in aux:
                    slr_h = aux['slr_out'][:, :-1]                # [B, S-1, D]
                    cos_loss = 1.0 - F.cosine_similarity(
                        slr_h.float(), target_emb.float(), dim=-1
                    ).mean()
                    slr_loss_sum = slr_loss_sum + cos_loss
                    n_slr_layers += 1
                    del aux['slr_out']  # free [B,S,D] immediately
                aux_list.append(aux)

        return x, aux_list if collect_aux else [], slr_loss_sum, n_slr_layers

    @staticmethod
    def _layer_fn(layer, x, bus_ring):
        """Wrapper para checkpoint — skips side effects en recomputación.

        Compile-safe: torch.compiler.is_compiling() detecta dry-runs del
        trazador de torch.compile, que incrementarían _fwd_count sin ejecutar
        un forward real → desincronización permanente del contador.

        FIX: try/finally garantiza que _skip_side_effects siempre se restaura
        a False aunque el forward lance una excepción. Sin esto, una excepción
        durante la recomputación dejaría _skip_side_effects=True permanentemente,
        silenciando TTT y archive.maybe_archive en todos los pasos subsiguientes
        (pérdida silenciosa de convergencia, imposible de diagnosticar).
        """
        # Guard: during torch.compile tracing, always skip side effects
        if torch.compiler.is_compiling():
            layer._skip_side_effects = True
            try:
                result = layer(x, bus_ring=bus_ring, return_aux=True)
            finally:
                layer._skip_side_effects = False
            return result

        count = getattr(layer, '_fwd_count', 0)
        layer._fwd_count = count + 1
        is_recompute = count > 0
        if is_recompute:
            layer._skip_side_effects = True
        try:
            result = layer(x, bus_ring=bus_ring, return_aux=True)
        finally:
            if is_recompute:
                layer._skip_side_effects = False
        return result


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
        ckpt_interval: int = 2,
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
        # Pass labels + embedding for inline SLR loss computation (frees N×[B,S,D])
        x, aux_list, slr_loss_sum, n_slr_layers = self.stack(
            x,
            collect_aux=(labels is not None),
            labels=labels,
            embedding=self.embedding,
        )

        # Norm final
        x = self.norm_f(x)               # [B, S, D]

        # LM-head: proyección a vocab
        if labels is None:
            # Inference: solo necesitamos el último token
            return self.lm_head(x)       # [B, S, vocab_size]

        # ── Cross-Entropy LM loss — chunked para no materializar [B,S,V] ──────
        # Con vocab=128k y B*S grande, el tensor de logits completo puede ser
        # decenas de GB. Procesamos en chunks de seq para evitar OOM.
        x_shifted  = x[:, :-1].contiguous()       # [B, S-1, D]
        shift_labels = labels[:, 1:].contiguous()  # [B, S-1]

        B_s, Sm1, D = x_shifted.shape
        V = self.vocab_size
        CHUNK = 256   # tokens por chunk (256 × B × V × 2B ≈ 256×64×128002×2 = 4GB máx)

        # Pad a múltiplo de CHUNK para shapes estáticos (evita recompilación
        # con torch.compile dynamic=False). Labels padded con -100 → ignorados.
        pad_len = (CHUNK - Sm1 % CHUNK) % CHUNK
        if pad_len > 0:
            x_shifted    = F.pad(x_shifted, (0, 0, 0, pad_len))        # [B, Sm1+pad, D]
            shift_labels = F.pad(shift_labels, (0, pad_len), value=-100)  # [B, Sm1+pad]
        Sm1_padded = Sm1 + pad_len

        lm_loss = x_shifted.new_zeros(())
        n_valid  = (shift_labels != -100).sum().float()

        for i in range(0, Sm1_padded, CHUNK):
            x_c  = x_shifted[:, i:i+CHUNK].reshape(-1, D)
            lbl_c = shift_labels[:, i:i+CHUNK].reshape(-1)
            logits_c = self.lm_head(x_c)           # [B*chunk, V]
            chunk_loss = F.cross_entropy(
                logits_c, lbl_c,
                ignore_index=-100,
                reduction='sum',
            )
            lm_loss = lm_loss + chunk_loss

        lm_loss = lm_loss / n_valid.clamp(min=1)

        # ── Routing loss (Z-loss + entropy + balance) ─────────────────────────
        routing_total = torch.tensor(0.0, device=input_ids.device)
        routing_info  = {}
        if aux_list:
            # Promediar pérdidas de routing sobre todas las capas
            layer_routing_losses = []
            for aux in aux_list:
                if aux is not None:
                    if 'router_logits' in aux:
                        rl, rinfo = self.routing_loss(aux)
                        layer_routing_losses.append(rl)
                        routing_info = rinfo   # último (para logging)
                    if 'moe_lb_loss' in aux and aux['moe_lb_loss'] is not None:
                        layer_routing_losses.append(aux['moe_lb_loss'] * 0.01) # moe weight scaling
                    if 'cas_aux' in aux and aux['cas_aux'] is not None:
                        # cas_aux podría ser dict o tensor. Asumimos dict con 'load_balance_loss'
                        if isinstance(aux['cas_aux'], dict) and 'load_balance_loss' in aux['cas_aux']:
                            layer_routing_losses.append(aux['cas_aux']['load_balance_loss'] * 0.01)
                        elif isinstance(aux['cas_aux'], torch.Tensor):
                            layer_routing_losses.append(aux['cas_aux'] * 0.01)
            
            if layer_routing_losses:
                routing_total = torch.stack(layer_routing_losses).mean()

        total_loss = lm_loss + aux_weight * routing_total

        # ── SLR latent-space auxiliary loss (computed inline in stack) ─────────
        # FIX V6: SLR loss se computa por capa DENTRO del stack loop y slr_out
        # se libera inmediatamente. Evita retener N×[B,S,D] tensores.
        # Para N=23, D=768, S=2048, B=4: ~276MB ahorrados.
        slr_aux_loss = slr_loss_sum / max(n_slr_layers, 1) if n_slr_layers > 0 else torch.tensor(0.0, device=input_ids.device)

        total_loss = total_loss + 0.1 * slr_aux_loss

        loss_dict = {
            'lm':      lm_loss.item(),
            'routing': routing_total.item() if routing_total.requires_grad else float(routing_total),
            'slr_aux': slr_aux_loss.item() if isinstance(slr_aux_loss, torch.Tensor) and slr_aux_loss.requires_grad else float(slr_aux_loss),
            'total':   total_loss.item(),
            **{k: v for k, v in routing_info.items()},
        }

        return None, total_loss, loss_dict

    # ── Generación autoregresiva ───────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,    # [B, S_prompt]
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Generación greedy / top-k token-by-token.

        NOTA: esta implementación hace prefill con forward completo (no step()),
        lo que es correcto pero ineficiente para prompts largos.
        Una versión optimizada usaría step() por cada token decoded.
        """
        B, S = input_ids.shape
        generated = input_ids

        for _ in range(max_new_tokens):
            logits = self(generated)          # [B, S_cur, V]
            next_logits = logits[:, -1, :]   # [B, V] — solo último token

            # Temperatura + top-k sampling
            if temperature != 1.0:
                next_logits = next_logits / temperature
            if top_k > 0:
                # Zeroing de valores fuera del top-k
                vals, _ = torch.topk(next_logits, top_k, dim=-1)
                threshold = vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, -1e9)

            probs     = F.softmax(next_logits, dim=-1)
            next_tok  = torch.multinomial(probs, num_samples=1)   # [B, 1]
            generated = torch.cat([generated, next_tok], dim=-1)

        return generated[:, S:]   # retornar solo los nuevos tokens


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de construcción
# ─────────────────────────────────────────────────────────────────────────────

def build_chimera_125M(vocab_size: int = 32000, ckpt_interval: int = 2, **kwargs) -> ChimeraLM:
    """
    Construye CHIMERA-130M: d=768, L=23.
    ~132M parámetros reales (medidos, incluyendo embedding 32k×768=24.6M).

    NOTA: El estimador ChimeraConfig.total_params_M sobreestima por ~57%
    (cuenta SLR como 4×D² cuando las proyecciones son compartidas).
    El parámetro count real se obtiene con sum(p.numel() for p in model.parameters()).
    """
    from chimera_config import ChimeraConfig
    cfg = ChimeraConfig(
        d_model   = 768,
        n_layers  = 23,
        expand    = 2,
        headdim   = 32,
        d_state   = 64,
        bus_dim   = 128,
        dtype     = "bfloat16",
        **kwargs,
    )
    return ChimeraLM(cfg, vocab_size=vocab_size, ckpt_interval=ckpt_interval)


def build_chimera_350M(vocab_size: int = 32000, ckpt_interval: int = 2, **kwargs) -> ChimeraLM:
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
    return ChimeraLM(cfg, vocab_size=vocab_size, ckpt_interval=ckpt_interval)


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
