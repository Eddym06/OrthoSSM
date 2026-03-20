"""
chimera_config.py — Configuración production-ready de CHIMERA
=============================================================
Centraliza todos los hiperparámetros del stack en un dataclass serializable
compatible con HuggingFace PretrainedConfig y json/yaml estándar.

Uso:
    cfg = ChimeraConfig(d_model=1024, n_layers=24)
    model = ChimeraStack.from_config(cfg)
    cfg.save("config.json")
    cfg2 = ChimeraConfig.load("config.json")

Invariantes garantizados en __post_init__:
    - n_heads (Mamba2) = (d_model * expand) / headdim  debe ser entero ≥ 1
    - ttt_rank ≤ d_model // 4  (mantiene overhead < 1%)
    - bus_dim ≤ d_model
    - VRAM estimado (prefill) disponible en .vram_estimate_mb()
"""

from __future__ import annotations
import dataclasses, json, math, os
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─── Estimador de VRAM (regla de pulgar, no remplaza nvitop) ───────────────────
def _estimate_vram_mb(d_model: int, n_layers: int, max_seq_len: int,
                      expand: int, d_state: int, bus_dim: int,
                      dtype_bytes: int = 4) -> dict:
    """
    Devuelve un dict con estimación de VRAM en MB por componente.
    Supone batch_size=1, FP32 a menos que dtype_bytes se especifique.
    """
    d_inner  = d_model * expand
    n_heads  = d_inner // 32                       # headdim fijo 32 por defecto

    # Parámetros del modelo (pesos)
    params_mamba2_per_layer  = (
        d_model * d_inner * 2   +   # in/out proj
        d_inner * d_state * n_heads  # ssm matrices
    )
    params_slr_per_layer     = d_model * d_model * 4        # Q1,Q2,K1,K2
    params_archive_per_layer = d_model * 128 * 8            # compress + attn heads
    params_bus_per_layer     = (d_model * bus_dim) * 3      # publish, gather_q, modulate
    params_router_per_layer  = d_model * 32 + 32 * 3        # MLP
    params_ttt_per_layer     = d_model * 4 * 2              # ttt_U + ttt_V (rank=4)
    params_total_per_layer   = (params_mamba2_per_layer + params_slr_per_layer +
                                params_archive_per_layer + params_bus_per_layer +
                                params_router_per_layer + params_ttt_per_layer)
    params_total = params_total_per_layer * n_layers
    weights_mb   = params_total * dtype_bytes / 1e6

    # Activaciones (prefill, B=1)
    # Mamba2 SSD scan: O(S * d_inner) en HBM con FlashSSM
    act_per_layer = max_seq_len * d_inner * dtype_bytes + max_seq_len * bus_dim * dtype_bytes
    act_mb        = act_per_layer * n_layers / 1e6

    # SSM states por capa (O(1) en inferencia, O(L) en training)
    ssm_state_per_layer = n_heads * 32 * d_state * dtype_bytes
    ssm_mb = ssm_state_per_layer * n_layers / 1e6

    # Landmarks (max 64 por capa, 128-dim)
    lm_mb = n_layers * 64 * 128 * dtype_bytes / 1e6

    total_mb = weights_mb + act_mb + ssm_mb + lm_mb
    return {
        "weights_mb":    round(weights_mb,  1),
        "activations_mb": round(act_mb,      1),
        "ssm_state_mb":  round(ssm_mb,      1),
        "landmarks_mb":  round(lm_mb,       1),
        "total_mb":      round(total_mb,    1),
        "total_gb":      round(total_mb / 1024, 3),
    }


# ─── ChimeraConfig ─────────────────────────────────────────────────────────────

@dataclass
class ChimeraConfig:
    """
    Configuración canónica de CHIMERA.

    Organización por sub-sistema:
      - Modelo base (Mamba2 SSD)
      - Router de complejidad
      - TTT-Lite (dt_bias adaptativo)
      - TTT-Full (low-rank U/V)
      - SLR + SGR
      - AsyncLightBus
      - NativeLandmarkArchive
      - Training
      - Inferencia
      - Metadatos
    """

    # ── Arquitectura base ────────────────────────────────────────────────────
    d_model:          int   = 256        # dimensión de embedding
    n_layers:         int   = 4          # número de capas CHIMERA
    expand:           int   = 2          # factor de expansión Mamba2 (d_inner = d_model * expand)
    headdim:          int   = 32         # dimension por cabeza SSM
    d_state:          int   = 64         # dimensión de estado SSM

    # ── Router ──────────────────────────────────────────────────────────────
    n_tiers:              int   = 3      # FAST / HYBRID / FULL
    router_hidden:        int   = 32     # hidden del MLP del router
    slr_threshold:        float = 0.30   # umbral dinámico base para activar SLR
    arch_threshold:       float = 0.50   # umbral dinámico base para archive retrieval

    # ── TTT-Lite ─────────────────────────────────────────────────────────────
    ttt_mini_chunk:   int   = 64         # longitud del mini-chunk para cómputo de gradiente
    ttt_lr:           float = 1e-3       # learning rate Lion para dt_bias
    ttt_beta:         float = 0.9        # momentum Lion
    ttt_threshold:    float = 0.3        # umbral de active_prob para activar TTT

    # ── TTT-Full (low-rank) ──────────────────────────────────────────────────
    ttt_rank:         int   = 4          # rango de la corrección U @ V
    ttt_scale_init:   float = -4.0       # sigmoid(-4) ≈ 0.018 → warm-up suave

    # ── SLR / SGR ────────────────────────────────────────────────────────────
    slr_window_size:  int   = 1024         # ventana de atención diferencial SLR
    sgr_top_k_frac:   float = 0.125      # fracción top-K de SGR (12.5%)

    # ── AsyncLightBus ─────────────────────────────────────────────────────────
    bus_dim:          int   = 128        # dimensión del bus

    # ── NativeLandmarkArchive ─────────────────────────────────────────────────
    landmark_dim:         int   = 128    # dimensión de landmarks comprimidos
    max_landmarks:        int   = 64     # máximo de landmarks por capa
    ttt_err_threshold:    float = 0.3    # umbral de error TTT para archivar

    # ── SpectralVSAArchive v2 (ChebyHolo) ─────────────────────────────────────
    use_spectral_vsa:        bool  = False   # True → usa SpectralVSAArchive, False → NativeLandmarkArchive
    spectral_K:              int   = 32      # K_max: máximo de coeficientes Chebyshev
    spectral_K_min:          int   = 4       # K_min: mínimo de coeficientes activos (dynamic K)
    spectral_window:         int   = 256     # tamaño de ventana circular del buffer
    spectral_ema_alpha:      float = 0.9     # EMA decay para binding en V_mem
    spectral_use_complex:    bool  = True    # True → roles DFT complejos, False → bipolares
    spectral_n_retrieve:     int   = 8       # bandas de frecuencia a recuperar
    spectral_energy_threshold: float = 0.95  # umbral de energía acumulada para dynamic K
    spectral_lanczos_power_max: float = 3.0  # potencia máxima Lanczos (anti-Runge/Gibbs); 3.0 evita over-damping
    spectral_disc_gamma:     float = 3.0     # sensibilidad del detector de discontinuidades
    spectral_error_refresh:  float = 0.5     # ratio error/V_mem para refresh completo

    # ── Mixture-of-Experts (MoE) ─────────────────────────────────────────────
    # Sparse Top-K MoE FFN — activar con use_moe=True para capacidad paramétrica
    # con FLOPs constantes. Si use_cas=True simultáneamente, CAS tiene prioridad.
    use_moe:              bool  = False    # True → activa MoE FFN
    moe_n_experts:        int   = 8       # número de expertos MoE
    moe_top_k:            int   = 2       # top-K expertos por token
    moe_d_ff:             int   = 0       # d_ff por experto (0 → auto: d_model * 2)

    # ── Chimera Autonomous Swarm (CAS) ──────────────────────────────────────
    # AoE × LExI × TTT-Coupled — expertos autónomos con:
    #   - Micro-probing: cada experto decide si procesar cada token (sin router softmax)
    #   - Depth threshold: umbral aprendible condicionado por profundidad de capa
    #   - TTT coupling: pérdida proxy modula umbral en tiempo real (respuesta inmunológica)
    # use_cas=True reemplaza funcionalmente a use_moe (CAS tiene prioridad)
    use_cas:              bool  = False   # True → activa CAS, False → sin CAS o usa MoE
    cas_n_experts:        int   = 8       # número de expertos autónomos
    cas_d_ff:             int   = 0       # d_ff por experto (0 → auto: d_model * 2)
    cas_tau_init:         float = 0.3     # umbral inicial de activación (~30% expertos)
    cas_target_active:    float = 0.25    # fracción objetivo de pares (token, expert) activos
    cas_budget_weight:    float = 0.01    # peso de la pérdida de budget de activación
    cas_diversity_weight: float = 0.005   # peso de la pérdida de diversidad (entropía)
    cas_balance_weight:   float = 0.005   # peso de la pérdida de balance entre expertos

    # ── Pérdidas auxiliares ──────────────────────────────────────────────────
    routing_entropy_weight:    float = 0.05  # peso hinge de entropía
    routing_supervision_weight: float = 0.10  # peso KL de supervisión TTT
    routing_balance_weight:    float = 0.02  # peso de load balance
    routing_target_entropy:    float = 0.70  # fracción de H_max deseada
    # NOTA: min_tier_prob DEBE ser > min_prob_floor (0.05) del GatedComplexityPredictor.
    # Si ambos son iguales, el floor garantiza mean_p >= 0.05 siempre →
    # F.relu(0.05 - mean_p) = 0 permanentemente → la balance loss nunca se activa.
    routing_min_tier_prob:     float = 0.10  # prob mínima por tier (> floor=0.05)
    ttt_pred_weight:           float = 0.05  # peso pérdida predictiva TTT

    # ── Training ─────────────────────────────────────────────────────────────
    dtype:            str   = "float32"       # "float32" | "bfloat16"
    use_tf32:         bool  = True            # torch.set_float32_matmul_precision('high')
    grad_clip:        float = 1.0             # gradient clipping global L2-norm
    weight_decay:     float = 0.1
    lr:               float = 3e-4
    lr_min:           float = 3e-5           # cosine decay floor
    warmup_steps:     int   = 500
    max_seq_len:      int   = 2048
    # Inicialización residual a escala: GPT-style (1/√(2·n_layers))
    # Referencia: GPT-2 paper §2.3
    residual_scale:   bool  = True

    # ── Inferencia ────────────────────────────────────────────────────────────
    # TTT-Lite se desactiva en decode (single-token, no mini-chunk)
    # TTT-Full sigue activo (corrección estática aprendida)
    decode_ttt_active: bool = False

    # ── Metadatos ─────────────────────────────────────────────────────────────
    version:          str = "chimera-1.0"
    architecture:     str = "CHIMERA-SSM-Hybrid"

    # ── Computed properties (no son parámetros directos) ─────────────────────

    def __post_init__(self):
        """Valida restricciones de integridad."""
        d_inner = self.d_model * self.expand
        if d_inner % self.headdim != 0:
            raise ValueError(
                f"d_inner={d_inner} (d_model*expand) debe ser divisible por "
                f"headdim={self.headdim}. Ajusta d_model, expand o headdim."
            )
        n_heads = d_inner // self.headdim
        if n_heads < 1:
            raise ValueError(f"n_heads={n_heads} < 1. Reduce headdim o aumenta d_model/expand.")

        if self.ttt_rank > self.d_model // 4:
            raise ValueError(
                f"ttt_rank={self.ttt_rank} > d_model//4={self.d_model//4}. "
                "Overhead TTT-Full > 1%, reduce ttt_rank."
            )
        if self.bus_dim > self.d_model:
            raise ValueError(f"bus_dim={self.bus_dim} > d_model={self.d_model}.")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError(f"dtype={self.dtype!r} inválido. Usar 'float32' o 'bfloat16'.")
        if not (0 < self.sgr_top_k_frac <= 1.0):
            raise ValueError(f"sgr_top_k_frac debe estar en (0, 1].")

    # ── Properties derivados ──────────────────────────────────────────────────

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand

    @property
    def n_heads(self) -> int:
        return self.d_inner // self.headdim

    @property
    def total_params_estimate(self) -> int:
        """Estimación de parámetros totales (M)."""
        per_layer = (
            self.d_model * self.d_inner * 2 +       # Mamba2 proj
            self.d_inner * self.d_state * self.n_heads +  # SSM
            self.d_model * self.d_model * 4 +       # SLR Q1,Q2,K1,K2
            self.d_model * self.bus_dim * 3 +        # Bus
            self.d_model * self.landmark_dim * 8 +   # Archive
            self.d_model * self.ttt_rank * 2 +       # TTT-Full U/V
            self.d_model * self.router_hidden + self.router_hidden * self.n_tiers
        )
        return per_layer * self.n_layers

    @property
    def total_params_M(self) -> float:
        return self.total_params_estimate / 1e6

    # ── Serialización ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        """Guarda configuración como JSON canonico."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ChimeraConfig":
        """Carga desde JSON."""
        with open(path) as f:
            data = json.load(f)
        # Eliminar campos desconocidos para compatibilidad hacia atrás
        known = {f.name for f in dataclasses.fields(cls)}
        data  = {k: v for k, v in data.items() if k in known}
        return cls(**data)

    @classmethod
    def tiny(cls) -> "ChimeraConfig":
        """Configuración mínima para pruebas unitarias rápidas."""
        return cls(d_model=64, n_layers=2, expand=2, headdim=32, d_state=16,
                   bus_dim=32, landmark_dim=32, max_landmarks=16, ttt_rank=2,
                   router_hidden=16)

    @classmethod
    def small_125M(cls) -> "ChimeraConfig":
        """~132M params reales — escala GPT-2 medium (n_layers=23 medido empíricamente).
        AVISO: total_params_M del estimador no es fiable (sobreestima SLR ~57%).
        Usar sum(p.numel() for p in model.parameters()) para el conteo exacto."""
        return cls(d_model=768, n_layers=23, expand=2, headdim=32, d_state=64,
                   bus_dim=128, landmark_dim=128, max_landmarks=64, ttt_rank=4,
                   router_hidden=64, max_seq_len=2048, lr=6e-4)

    @classmethod
    def medium_350M(cls) -> "ChimeraConfig":
        """~350M params — escala GPT-2 medium."""
        return cls(d_model=1024, n_layers=24, expand=2, headdim=32, d_state=64,
                   bus_dim=256, landmark_dim=256, max_landmarks=128, ttt_rank=8,
                   router_hidden=128, max_seq_len=4096, lr=3e-4)

    @classmethod
    def large_1B(cls) -> "ChimeraConfig":
        """~1B params."""
        return cls(d_model=2048, n_layers=24, expand=2, headdim=64, d_state=128,
                   bus_dim=256, landmark_dim=256, max_landmarks=128, ttt_rank=8,
                   router_hidden=256, max_seq_len=8192, lr=2e-4,
                   dtype="bfloat16")

    @classmethod
    def xlarge_3B(cls) -> "ChimeraConfig":
        """~3B params."""
        return cls(d_model=2560, n_layers=32, expand=2, headdim=64, d_state=128,
                   bus_dim=512, landmark_dim=512, max_landmarks=256, ttt_rank=16,
                   router_hidden=512, max_seq_len=16384, lr=1e-4,
                   dtype="bfloat16")

    # ── Estimación de VRAM ────────────────────────────────────────────────────

    def vram_estimate(self, max_seq_len: Optional[int] = None) -> dict:
        """Estimación de VRAM en MB/GB para la configuración actual."""
        dtype_bytes = 2 if self.dtype == "bfloat16" else 4
        S = max_seq_len or self.max_seq_len
        vram = _estimate_vram_mb(
            d_model=self.d_model, n_layers=self.n_layers,
            max_seq_len=S, expand=self.expand, d_state=self.d_state,
            bus_dim=self.bus_dim, dtype_bytes=dtype_bytes
        )
        if self.use_spectral_vsa:
            # SpectralVSA replaces NativeLandmarkArchive: recompute archive VRAM.
            # Main state (FP32 always): V_mem±comps (D×8 B) + buf (W×D×4 B)
            #   + c_now/c_past/roles_real/roles_imag/err_corr×2 (6×K×D×4 B)
            old_lm = self.n_layers * 64 * 128 * dtype_bytes / 1e6
            new_lm = self.n_layers * (
                self.d_model * 8                              # V_mem complex + Kahan comps
                + self.spectral_window * self.d_model * 4    # circular buffer (FP32)
                + 6 * self.spectral_K * self.d_model * 4     # coeffs, roles, error_corr
            ) / 1e6
            vram["landmarks_mb"] = round(new_lm, 1)
            vram["total_mb"]     = round(vram["total_mb"] - old_lm + new_lm, 1)
            vram["total_gb"]     = round(vram["total_mb"] / 1024, 3)
        return vram

    def __repr__(self) -> str:
        vram = self.vram_estimate()
        return (f"ChimeraConfig(\n"
                f"  arch={self.version}  d={self.d_model}  L={self.n_layers}\n"
                f"  expand={self.expand}  headdim={self.headdim}  d_state={self.d_state}\n"
                f"  n_heads={self.n_heads}  d_inner={self.d_inner}\n"
                f"  ~params={self.total_params_M:.1f}M  dtype={self.dtype}\n"
                f"  VRAM(S={self.max_seq_len}): {vram['total_gb']:.2f} GB\n"
                f"    weights={vram['weights_mb']} MB  acts={vram['activations_mb']} MB\n"
                f")")


# ─── Preset table ──────────────────────────────────────────────────────────────

CHIMERA_PRESETS = {
    "tiny":       ChimeraConfig.tiny,
    "125M":       ChimeraConfig.small_125M,
    "350M":       ChimeraConfig.medium_350M,
    "1B":         ChimeraConfig.large_1B,
    "3B":         ChimeraConfig.xlarge_3B,
}


# ─── ChimeraStack skeleton ────────────────────────────────────────────────────

class ChimeraStack:
    """
    Wrapper de producción que instancia un stack de N AdvancedChimeraLayer
    desde un ChimeraConfig.

    Incluye:
    - Inicialización residual a escala (GPT-style)
    - Gradient clipping integrado
    - Logging de routing stats por batch
    """

    @classmethod
    def from_config(cls, cfg: ChimeraConfig):
        """Crea el stack. Requiere torch y advanced_chimera importados."""
        import torch.nn as nn
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from advanced_chimera import AdvancedChimeraLayer

        layers = nn.ModuleList([
            AdvancedChimeraLayer(
                d_model=cfg.d_model,
                expand=cfg.expand,
                headdim=cfg.headdim,
                layer_idx=i,
                d_state=cfg.d_state,
                use_spectral_vsa       = cfg.use_spectral_vsa,
                spectral_K             = cfg.spectral_K,
                spectral_K_min         = cfg.spectral_K_min,
                spectral_window        = cfg.spectral_window,
                spectral_ema_alpha     = cfg.spectral_ema_alpha,
                spectral_use_complex   = cfg.spectral_use_complex,
                spectral_n_retrieve    = cfg.spectral_n_retrieve,
                spectral_energy_threshold  = cfg.spectral_energy_threshold,
                spectral_lanczos_power_max = cfg.spectral_lanczos_power_max,
                spectral_disc_gamma    = cfg.spectral_disc_gamma,
                spectral_error_refresh = cfg.spectral_error_refresh,
                use_moe                = cfg.use_moe,
                moe_n_experts          = cfg.moe_n_experts,
                moe_top_k              = cfg.moe_top_k,
                moe_d_ff               = cfg.moe_d_ff if cfg.moe_d_ff > 0 else None,
                use_cas                = cfg.use_cas,
                cas_n_experts          = cfg.cas_n_experts,
                cas_d_ff               = cfg.cas_d_ff if cfg.cas_d_ff > 0 else None,
                cas_tau_init           = cfg.cas_tau_init,
                cas_target_active      = cfg.cas_target_active,
                n_layers_total         = cfg.n_layers,
            )
            for i in range(cfg.n_layers)
        ])

        # Residual scaling a escala (GPT-2 §2.3)
        if cfg.residual_scale:
            import torch, math
            scale = 1.0 / math.sqrt(2 * cfg.n_layers)
            for layer in layers:
                # Escalar la proyección de salida de Mamba2 (out_proj)
                if hasattr(layer.mamba2, 'out_proj'):
                    with torch.no_grad():
                        layer.mamba2.out_proj.weight.mul_(scale)

        return layers


# ─── CLI de inspección ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    preset = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    factory = CHIMERA_PRESETS.get(preset)
    if factory is None:
        print(f"Uso: python chimera_config.py [tiny|125M|350M|1B|3B]")
        print(f"Presets disponibles: {list(CHIMERA_PRESETS)}")
        sys.exit(1)

    cfg = factory()
    print(cfg)
    print(f"\nVRAM estimado (prefill S={cfg.max_seq_len}):")
    vram = cfg.vram_estimate()
    for k, v in vram.items():
        print(f"  {k:20s} = {v}")

    # Round-trip JSON
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    cfg.save(path)
    cfg2 = ChimeraConfig.load(path)
    os.unlink(path)
    assert cfg.to_dict() == cfg2.to_dict(), "Round-trip JSON falló"
    print(f"\nRound-trip JSON: ✓")
    print(f"Parámetros estimados: {cfg.total_params_M:.1f}M")
