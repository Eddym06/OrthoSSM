"""OrthoSSM V10 "Lightning" -- Unified Language Model
====================================================
Single canonical model definition.

V10 changes:
  - State per layer: (coeffs, momentum) instead of (coeffs, m1, m2)
  - Default degree=4 (was 8). Configurable via max_degree param.
  - Length routing: fast/hybrid/full paths based on sequence length
  - AsyncLightBus replaces CrossLayerMemoryBus
  - LUT-accelerated Chebyshev evaluation
  - Lion optimizer (single momentum, no m2)
  - Backward compatible with V9 state format (auto-converts)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sdpc_engine import SpectralDualPathContextEngine, build_ortho_stack
from sdpc_kernel import init_chebyshev_coefficients


class OrthoSSMLanguageModel(nn.Module):
    """
    OrthoSSM V10 "Lightning" Language Model.

    Args:
        vocab_size:     Vocabulary size (default: 131072)
        d_model:        Hidden dimension (default: 256)
        n_attn_heads:   Number of attention heads in NSA (default: 4)
        n_cheby_heads:  Number of Chebyshev spectral heads (default: 8)
        n_layers:       Number of stacked OrthoSSM layers (default: 2)
        max_degree:     Chebyshev polynomial degree per head (default: 4, was 8)
        window_size:    Sliding window size for local attention (default: 512)
        tie_weights:    Whether to tie embedding and lm_head weights (default: True)
        use_bf16:       Enable BF16 mixed precision (default: False)
        gradient_ckpt:  Enable gradient checkpointing (default: False)
        compile_model:  Apply torch.compile (default: False)
        use_lut:        Use Chebyshev LUT acceleration (default: True)
        **engine_kwargs: Extra kwargs for SpectralDualPathContextEngine
    """
    def __init__(
        self,
        vocab_size: int = 131072,
        d_model: int = 256,
        n_attn_heads: int = 4,
        n_cheby_heads: int = 8,
        n_layers: int = 2,
        max_degree: int = 4,
        window_size: int = 512,
        tie_weights: bool = True,
        use_bf16: bool = False,
        gradient_ckpt: bool = False,
        compile_model: bool = False,
        use_lut: bool = True,
        **engine_kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_cheby_heads = n_cheby_heads
        self.max_degree = max_degree
        self.gradient_ckpt = gradient_ckpt
        self.compile_model = compile_model

        # ── Token embedding ──
        self.embedding = nn.Embedding(vocab_size, d_model)

        # ── V10 layer stack with AsyncLightBus ──
        self.layers, self.light_bus = build_ortho_stack(
            d_model=d_model,
            n_attn_heads=n_attn_heads,
            n_cheby_heads=n_cheby_heads,
            n_layers=n_layers,
            max_degree=max_degree,
            window_size=window_size,
            use_bf16=use_bf16,
            use_lut=use_lut,
            gradient_ckpt=gradient_ckpt,  # C3: engine-level granular checkpointing
            **engine_kwargs,
        )

        # ── Final norm + LM head ──
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()

        if compile_model and hasattr(torch, 'compile'):
            # C4: torch.compile with reduce-overhead mode for inter-kernel fusion
            for i, layer in enumerate(self.layers):
                self.layers[i] = torch.compile(layer, mode='reduce-overhead', fullgraph=False)

    def _init_weights(self, std: float = 0.02):
        """Initialize weights. Residual projections scaled by 1/sqrt(2*n_layers)."""
        residual_scale = 1.0 / (2 * self.n_layers) ** 0.5
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                if 'out_proj' in name:
                    nn.init.normal_(p, mean=0.0, std=std * residual_scale)
                else:
                    nn.init.normal_(p, mean=0.0, std=std)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def fresh_states(self, batch: int, device: torch.device):
        """
        Generate fresh state tuples for all layers.

        V10: Returns (coeffs, momentum) per layer (no m2).
        V9 compat: add m2=zeros if needed.
        """
        head_dim = self.d_model // self.n_cheby_heads
        states = []
        for _ in range(self.n_layers):
            coeffs = init_chebyshev_coefficients(
                batch, self.n_cheby_heads, self.max_degree, head_dim, device
            )
            momentum = torch.zeros_like(coeffs)
            states.append((coeffs, momentum))
        return states

    def forward(self, input_ids, states=None, return_state=False):
        """
        Args:
            input_ids:    [B, S] token ids
            states:       List of (coeffs, momentum) per layer, or None
            return_state: If True, return (logits, new_states)

        Returns:
            logits: [B, S, vocab_size]
            new_states: (if return_state) list of detached (coeffs, momentum)
        """
        B, S = input_ids.shape
        x = self.embedding(input_ids)

        # Clear AsyncLightBus for new forward pass
        self.light_bus.clear()

        new_states = []
        for i, layer in enumerate(self.layers):
            layer_state = states[i] if states is not None else None

            if self.gradient_ckpt and self.training:
                # C3: engine-level granular checkpointing handles heavy ops;
                # model-level checkpoint wraps the rest (norms, projections)
                x, st = checkpoint(
                    self._layer_forward, layer, x, layer_state,
                    use_reentrant=False,
                )
            else:
                x, st = layer(x, cheby_state=layer_state, return_state=True)

            new_states.append(st)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        # E5: Snapshot bus state for gradient checkpointing compatibility
        self.light_bus.snapshot()

        if return_state:
            return logits, new_states
        return logits

    @staticmethod
    def _layer_forward(layer, x, state):
        return layer(x, cheby_state=state, return_state=True)

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self):
        n_params = self.count_parameters()
        return (
            f"OrthoSSMLanguageModel(\n"
            f"  d_model={self.d_model}, n_layers={self.n_layers},\n"
            f"  n_cheby_heads={self.n_cheby_heads}, max_degree={self.max_degree},\n"
            f"  version='V10 Lightning',\n"
            f"  params={n_params:,} ({n_params/1e6:.1f}M)\n"
            f")"
        )
