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
                      max_landmarks: int = 64, landmark_dim: int = 128,
                      sdtm_n_heads: int = 1, sdtm_d_mem: int = 0,
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
    params_archive_per_layer = d_model * landmark_dim * 8   # compress + attn heads
    params_bus_per_layer     = (d_model * bus_dim) * 3      # publish, gather_q, modulate
    params_router_per_layer  = d_model * 32 + 32 * 3        # MLP
    params_ttt_per_layer     = d_model * 4 * 2              # ttt_U + ttt_V (rank=4)
    d_mem = sdtm_d_mem if sdtm_d_mem > 0 else max(64, d_model // 4)
    d_mem_total = d_mem * sdtm_n_heads
    params_sdtm_per_layer    = (d_model * d_mem_total * 2 +  # W_enc + W_dec
                                d_model * sdtm_n_heads + sdtm_n_heads)  # gate_proj
    params_total_per_layer   = (params_mamba2_per_layer + params_slr_per_layer +
                                params_archive_per_layer + params_bus_per_layer +
                                params_router_per_layer + params_ttt_per_layer +
                                params_sdtm_per_layer)
    params_total = params_total_per_layer * n_layers
    weights_mb   = params_total * dtype_bytes / 1e6

    # Activaciones (prefill, B=1)
    act_per_layer = max_seq_len * d_inner * dtype_bytes + max_seq_len * bus_dim * dtype_bytes
    act_mb        = act_per_layer * n_layers / 1e6

    # SSM states por capa (O(1) en inferencia, O(L) en training)
    ssm_state_per_layer = n_heads * 32 * d_state * dtype_bytes
    ssm_mb = ssm_state_per_layer * n_layers / 1e6

    # Landmarks (max por capa)
    lm_mb = n_layers * max_landmarks * landmark_dim * dtype_bytes / 1e6

    # SDTM state per layer: n_heads × (M_fast + M_slow + momentum + 2×kahan + usage_ema)
    sdtm_state = sdtm_n_heads * d_mem * d_mem * 5 + sdtm_n_heads * d_mem  # elements
    sdtm_mb = n_layers * sdtm_state * 4 / 1e6  # always FP32 for Lion buffers

    total_mb = weights_mb + act_mb + ssm_mb + lm_mb + sdtm_mb
    return {
        "weights_mb":    round(weights_mb,  1),
        "activations_mb": round(act_mb,      1),
        "ssm_state_mb":  round(ssm_mb,      1),
        "landmarks_mb":  round(lm_mb,       1),
        "sdtm_mb":       round(sdtm_mb,     1),
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
    d_state:          int   = 128        # dimensión de estado SSM (v1.3: 64→128 — 2× capacidad literal)

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
    slr_window_size:  int   = 64         # ventana de atención diferencial SLR
    sgr_top_k_frac:   float = 0.125      # fracción top-K de SGR (12.5%)

    # ── AsyncLightBus ─────────────────────────────────────────────────────────
    bus_dim:          int   = 256        # dimensión del bus (v1.3: 128→256 — 2× ancho de banda)

    # ── NativeLandmarkArchive ───────────────────────────────────────────────
    landmark_dim:         int   = 128    # dimensión de landmarks comprimidos
    max_landmarks:        int   = 512    # máximo de landmarks por capa (v1.3: 64→512 — 8× snapshots)
    ttt_err_threshold:    float = 0.3    # umbral de error TTT para archivar

    # ── Pérdidas auxiliares ──────────────────────────────────────────────────
    routing_entropy_weight:    float = 0.05  # peso hinge de entropía
    routing_supervision_weight: float = 0.10  # peso KL de supervisión TTT
    routing_balance_weight:    float = 0.02  # peso de load balance
    routing_target_entropy:    float = 0.70  # fracción de H_max deseada
    routing_min_tier_prob:     float = 0.05  # prob mínima por tier
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

    # ── Chunked Training (contexto infinito) ────────────────────────────────
    # Procesa secuencias largas en chunks con carry de estado SSM/bus/archive.
    # 'auto' → estima chunk_size basado en VRAM disponible dinámicamente.
    chunked_training:     bool       = False     # activar chunked TBPTT
    chunk_size:           int | str  = 'auto'    # tokens por chunk (int o 'auto')
    chunk_gc_interval:    int        = 4         # GC cada N chunks

    # ── Curriculum Learning (adaptativo) ─────────────────────────────────────
    # El curriculum se adapta a las probabilidades del router en tiempo real.
    curriculum_enabled:   bool  = False    # activar curriculum adaptativo
    curriculum_ema_alpha: float = 0.05     # alpha para EMA de routing stats

    # ── Serving (Paged State Manager) ────────────────────────────────────────
    max_serving_sessions: int | str = 'auto'   # max sesiones concurrentes
    serving_ring_size:    int       = 16       # ring buffer por sesión

    # ── SDTM (Surprise-Driven Dual-Timescale Memory) ─────────────────────────
    # Memoria asociativa dinámica O(1) VRAM, Multi-Head.
    # v1.3: n_heads=4 — cada cabeza se especializa en patrones distintos.
    sdtm_enabled:                bool  = True       # habilitar SDTM
    sdtm_n_heads:                int   = 4          # cabezas de memoria (v1.3: 1→4)
    sdtm_d_mem:                  int   = 0          # 0 = auto (max(64, d_model//4)) per head
    sdtm_lr:                     float = 5e-4       # learning rate Lion para M_fast
    sdtm_beta:                   float = 0.9        # momentum Lion
    sdtm_consolidation_interval: int   = 2048       # tokens entre consolidaciones
    sdtm_consolidation_rate:     float = 0.3        # shrink de M_fast tras consolidar
    sdtm_surprise_top_k:         int   = 16         # tokens sorprendentes por write

    # ── Inferencia ────────────────────────────────────────────────────────────
    # TTT-Lite se desactiva en decode (single-token, no mini-chunk)
    # TTT-Full sigue activo (corrección estática aprendida)
    decode_ttt_active: bool = False

    # ── Metadatos ─────────────────────────────────────────────────────────────
    version:          str = "chimera-1.3"
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
        d_mem = self.sdtm_d_mem if self.sdtm_d_mem > 0 else max(64, self.d_model // 4)
        d_mem_total = d_mem * self.sdtm_n_heads
        per_layer = (
            self.d_model * self.d_inner * 2 +       # Mamba2 proj
            self.d_inner * self.d_state * self.n_heads +  # SSM
            self.d_model * self.d_model * 4 +       # SLR Q1,Q2,K1,K2
            self.d_model * self.bus_dim * 3 +        # Bus
            self.d_model * self.landmark_dim * 8 +   # Archive
            self.d_model * self.ttt_rank * 2 +       # TTT-Full U/V
            self.d_model * self.router_hidden + self.router_hidden * self.n_tiers +
            self.d_model * d_mem_total * 2 +         # SDTM W_enc + W_dec
            self.d_model * self.sdtm_n_heads + self.sdtm_n_heads  # SDTM gate_proj
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
                   router_hidden=16, sdtm_n_heads=1)

    @classmethod
    def small_125M(cls) -> "ChimeraConfig":
        """~125M params — escala GPT-2 small. Primera prueba real."""
        return cls(d_model=768, n_layers=12, expand=2, headdim=32, d_state=128,
                   bus_dim=256, landmark_dim=128, max_landmarks=512, ttt_rank=4,
                   router_hidden=64, max_seq_len=2048, lr=6e-4,
                   sdtm_n_heads=4)

    @classmethod
    def medium_350M(cls) -> "ChimeraConfig":
        """~350M params — escala GPT-2 medium."""
        return cls(d_model=1024, n_layers=24, expand=2, headdim=32, d_state=128,
                   bus_dim=512, landmark_dim=256, max_landmarks=1024, ttt_rank=8,
                   router_hidden=128, max_seq_len=4096, lr=3e-4,
                   sdtm_n_heads=4)

    @classmethod
    def large_1B(cls) -> "ChimeraConfig":
        """~1B params."""
        return cls(d_model=2048, n_layers=24, expand=2, headdim=64, d_state=128,
                   bus_dim=512, landmark_dim=256, max_landmarks=1024, ttt_rank=8,
                   router_hidden=256, max_seq_len=8192, lr=2e-4,
                   dtype="bfloat16", sdtm_n_heads=4)

    @classmethod
    def xlarge_3B(cls) -> "ChimeraConfig":
        """~3B params."""
        return cls(d_model=2560, n_layers=32, expand=2, headdim=64, d_state=256,
                   bus_dim=512, landmark_dim=512, max_landmarks=2048, ttt_rank=16,
                   router_hidden=512, max_seq_len=16384, lr=1e-4,
                   dtype="bfloat16", sdtm_n_heads=4)

    # ── Estimación de VRAM ────────────────────────────────────────────────────

    def vram_estimate(self, max_seq_len: Optional[int] = None) -> dict:
        """Estimación de VRAM en MB/GB para la configuración actual."""
        dtype_bytes = 2 if self.dtype == "bfloat16" else 4
        S = max_seq_len or self.max_seq_len
        return _estimate_vram_mb(
            d_model=self.d_model, n_layers=self.n_layers,
            max_seq_len=S, expand=self.expand, d_state=self.d_state,
            bus_dim=self.bus_dim, dtype_bytes=dtype_bytes,
            max_landmarks=self.max_landmarks, landmark_dim=self.landmark_dim,
            sdtm_n_heads=self.sdtm_n_heads, sdtm_d_mem=self.sdtm_d_mem,
        )

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
                d_state=cfg.d_state,
                bus_dim=cfg.bus_dim,
                landmark_dim=cfg.landmark_dim,
                max_landmarks=cfg.max_landmarks,
                ttt_err_threshold=cfg.ttt_err_threshold,
                sdtm_n_heads=cfg.sdtm_n_heads,
                sdtm_d_mem=cfg.sdtm_d_mem,
            )
            for _ in range(cfg.n_layers)
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
