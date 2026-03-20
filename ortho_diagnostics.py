"""
OrthoSSM Diagnostic Mode — Zero-Overhead Subsystem Dashboard
=============================================================
Recopila métricas de todos los subsistemas de OrthoSSM en un único
objeto centralizado. En producción: overhead = cero (guarda tras un
`if DIAG.enabled` que el intérprete cortocircuita inmediatamente).

Métricas recopiladas:
  1. LUT interpolation errors  — error abs. de interpolación LUT vs Clenshaw exacto
  2. Head orthogonality         — cosine-sim entre cabezas de coeficientes
  3. Recall similarity dist.   — distribución de max_sim en LandmarkArchive
  4. EMA momentum actual        — ema_momentum_t por step
  5. Bus staleness counts       — steps sin publicar por capa en AsyncLightBus
  6. Padding waste %            — S real vs S padded (potencia de 2 más cercana)
  7. Rounding bias statistics   — sesgo y varianza del stochastic rounding BF16

Uso rápido:
    from ortho_diagnostics import DIAG
    DIAG.enable()  # activar
    # … forward passes …
    DIAG.print_dashboard()
    DIAG.reset()
"""
import math
import threading
from collections import defaultdict, deque
from typing import Optional

import torch


# ============================================================================
# RING BUFFER — estadísticas deslizantes sin alloc infinito
# ============================================================================

class _RingBuffer:
    """Buffer circular de tamaño fijo para estadísticas recientes."""
    def __init__(self, maxlen: int = 256):
        self._buf   = deque(maxlen=maxlen)
        self._lock  = threading.Lock()

    def push(self, val: float):
        with self._lock:
            self._buf.append(val)

    def push_many(self, vals):
        with self._lock:
            self._buf.extend(vals)

    def stats(self) -> dict:
        with self._lock:
            data = list(self._buf)
        if not data:
            return {"n": 0, "mean": float("nan"), "std": float("nan"),
                    "min": float("nan"), "max": float("nan")}
        n   = len(data)
        mu  = sum(data) / n
        if n > 1:
            var = sum((v - mu) ** 2 for v in data) / (n - 1)
            sd  = math.sqrt(var)
        else:
            sd = 0.0
        return {
            "n":   n,
            "mean": mu,
            "std":  sd,
            "min":  min(data),
            "max":  max(data),
        }

    def __len__(self):
        return len(self._buf)


# ============================================================================
# ORTHO DIAGNOSTICS COLLECTOR
# ============================================================================

class OrthoSSMDiagnostics:
    """
    Dashboard de diagnóstico para OrthoSSM V10.

    Uso típico:
        from ortho_diagnostics import DIAG
        DIAG.enable()
        … # ejecutar forwards …
        DIAG.print_dashboard()
        DIAG.reset()

    Overhead en producción (enabled=False): un bool check por métrica.
    """

    def __init__(self):
        self.enabled: bool = False
        self._step:   int  = 0
        self._lock    = threading.Lock()

        # Metric 1: LUT errors [abs error vs exact Clenshaw]
        self.lut_abs_err           = _RingBuffer(512)

        # Metric 2: Head orthogonality [max |cosine_sim| between head pairs]
        self.head_ortho_max_cosim  = _RingBuffer(256)
        self.head_ortho_mean_cosim = _RingBuffer(256)

        # Metric 3: Recall similarity distribution
        self.recall_max_sim        = _RingBuffer(1024)
        self.recall_inject_count   = _RingBuffer(256)

        # Metric 4: EMA momentum actual (per step)
        self.ema_momentum_actual   = _RingBuffer(1024)

        # Metric 5: Bus staleness (per layer, last N readings)
        self.bus_staleness: dict[int, _RingBuffer] = defaultdict(
            lambda: _RingBuffer(256)
        )

        # Metric 6: Padding waste %
        self.padding_waste_pct     = _RingBuffer(1024)

        # Metric 7: Rounding bias and abs error
        self.round_bias            = _RingBuffer(512)   # E[result - original]
        self.round_abs_err         = _RingBuffer(512)   # E[|result - original|]

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def enable(self):
        """Activar diagnóstico (overhead mínimo)."""
        self.enabled = True

    def disable(self):
        self.enabled = False

    def reset(self):
        """Limpiar todos los buffers y reiniciar contador de steps."""
        self.lut_abs_err.reset() if hasattr(self.lut_abs_err, 'reset') else None
        for attr_name in [
            'lut_abs_err', 'head_ortho_max_cosim', 'head_ortho_mean_cosim',
            'recall_max_sim', 'recall_inject_count', 'ema_momentum_actual',
            'padding_waste_pct', 'round_bias', 'round_abs_err',
        ]:
            setattr(self, attr_name, _RingBuffer(512))
        self.bus_staleness = defaultdict(lambda: _RingBuffer(256))
        with self._lock:
            self._step = 0

    # ------------------------------------------------------------------
    # Metric 1: LUT interpolation error
    # ------------------------------------------------------------------

    def record_lut_error(self, lut_out: torch.Tensor, exact_out: torch.Tensor):
        """
        Llamar con una muestra de salidas LUT vs Clenshaw exacto.
        lut_out, exact_out: tensores de cualquier forma.
        """
        if not self.enabled:
            return
        with torch.no_grad():
            err = (lut_out.float() - exact_out.float()).abs().mean().item()
        self.lut_abs_err.push(err)

    def check_lut_error_sample(self, coeffs: torch.Tensor,
                                degree: int, n_samples: int = 64,
                                device: Optional[torch.device] = None):
        """
        Auto-evalúa el error LUT tomando `n_samples` puntos aleatorios.
        Compara salida LUT (usando la tabla global) vs Clenshaw exacto.
        Se puede llamar periódicamente sin necesidad de interceptar kernels.
        """
        if not self.enabled:
            return
        try:
            from sdpc_kernel import get_chebyshev_lut
        except ImportError:
            return
        _device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        with torch.no_grad():
            x = torch.linspace(-1.0, 1.0, n_samples, device=_device)
            # Exact Clenshaw for degree=4
            if degree >= 1:
                T0 = torch.ones_like(x)
            if degree >= 2:
                T1 = x
            if degree >= 3:
                T2 = 2 * x * T1 - T0
            if degree >= 4:
                T3 = 2 * x * T2 - T1
            # Use first batch/head coefficients, averaged over head_dim
            c = coeffs[0, 0, :degree, :].mean(dim=-1)  # [degree]
            exact = sum(c[k].item() * locals().get(f'T{k}', torch.zeros_like(x)) for k in range(min(degree, 4)))
            # LUT approximation
            lut = get_chebyshev_lut(_device)  # [max_degree, table_size]
            abs_x = x.abs()
            x_sign = x.sign()
            fidx = abs_x * (lut.shape[1] - 1)
            idx0 = fidx.long().clamp(0, lut.shape[1] - 2)
            frac = fidx - idx0.float()
            lut_approx = torch.zeros_like(x)
            for k in range(min(degree, 4)):
                t0_a = lut[k, idx0]
                t0_b = lut[k, idx0 + 1]
                T_lut = t0_a + frac * (t0_b - t0_a)
                parity = x_sign if (k % 2 == 1) else torch.ones_like(x_sign)
                lut_approx = lut_approx + c[k].item() * T_lut * parity
            err = (lut_approx - exact).abs().mean().item()
        self.lut_abs_err.push(err)

    # ------------------------------------------------------------------
    # Metric 2: Head orthogonality
    # ------------------------------------------------------------------

    def record_head_orthogonality(self, coeffs: torch.Tensor):
        """
        coeffs: [B, n_heads, degree, head_dim]
        Calcula cosine-similarity entre pares de cabezas.
        """
        if not self.enabled:
            return
        with torch.no_grad():
            B, nH, deg, hD = coeffs.shape
            flat = coeffs[0].reshape(nH, -1).float()  # [nH, deg*hD]
            norm = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            flat_n = flat / norm
            cosim = (flat_n @ flat_n.T).abs()          # [nH, nH]
            # Remove diagonal
            mask = ~torch.eye(nH, dtype=torch.bool, device=cosim.device)
            off_diag = cosim[mask]
            self.head_ortho_max_cosim.push(off_diag.max().item())
            self.head_ortho_mean_cosim.push(off_diag.mean().item())

    # ------------------------------------------------------------------
    # Metric 3: Recall similarity distribution
    # ------------------------------------------------------------------

    def record_recall_similarity(self, max_sim: torch.Tensor,
                                  inject_count: int = 0):
        """
        max_sim: [B] tensor de max cosine similarities con landmarks.
        inject_count: cuántos ítems del batch se inyectaron.
        """
        if not self.enabled:
            return
        with torch.no_grad():
            for v in max_sim.float().tolist():
                self.recall_max_sim.push(v)
        self.recall_inject_count.push(float(inject_count))

    # ------------------------------------------------------------------
    # Metric 4: EMA momentum
    # ------------------------------------------------------------------

    def record_ema_momentum(self, ema_momentum: float):
        if not self.enabled:
            return
        self.ema_momentum_actual.push(ema_momentum)

    # ------------------------------------------------------------------
    # Metric 5: Bus staleness
    # ------------------------------------------------------------------

    def record_bus_staleness(self, layer_idx: int, staleness: int):
        """
        staleness: número de steps desde el último publish de esta capa.
        """
        if not self.enabled:
            return
        self.bus_staleness[layer_idx].push(float(staleness))

    # ------------------------------------------------------------------
    # Metric 6: Padding waste
    # ------------------------------------------------------------------

    def record_sequence_length(self, S_actual: int, S_padded: Optional[int] = None):
        """
        S_padded: si None, se calcula como la siguiente potencia de 2.
        """
        if not self.enabled:
            return
        if S_padded is None:
            S_padded = 2 ** math.ceil(math.log2(max(S_actual, 1)))
        waste_pct = (1.0 - S_actual / S_padded) * 100.0 if S_padded > 0 else 0.0
        self.padding_waste_pct.push(waste_pct)

    # ------------------------------------------------------------------
    # Metric 7: Stochastic rounding bias
    # ------------------------------------------------------------------

    def record_rounding_stats(self, original: torch.Tensor,
                               rounded: torch.Tensor):
        """
        original: tensor FP32 original
        rounded:  tensor BF16 redondeado (cast to float para comparar)
        """
        if not self.enabled:
            return
        with torch.no_grad():
            diff = (rounded.float() - original.float())
            self.round_bias.push(diff.mean().item())
            self.round_abs_err.push(diff.abs().mean().item())

    # ------------------------------------------------------------------
    # Step counter
    # ------------------------------------------------------------------

    def step(self):
        if not self.enabled:
            return
        with self._lock:
            self._step += 1

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def _section(self, title: str, width: int = 72) -> str:
        bar = "═" * width
        return f"\n╔{bar}╗\n║  {title:<{width - 2}}║\n╚{bar}╝"

    def _row(self, label: str, stats: dict, width: int = 60) -> str:
        if stats["n"] == 0:
            return f"  {label:<28}  [sin datos]"
        return (
            f"  {label:<28}  "
            f"n={stats['n']:<6}  "
            f"μ={stats['mean']:+.5f}  "
            f"σ={stats['std']:.5f}  "
            f"[{stats['min']:+.5f}, {stats['max']:+.5f}]"
        )

    def print_dashboard(self):
        """Imprime el dashboard completo en stdout."""
        print(self.get_report())

    def get_report(self) -> str:
        with self._lock:
            step = self._step

        lines = []
        lines.append(self._section(
            f"OrthoSSM V10 — Diagnostic Dashboard  (step={step})"
        ))

        # ── 1. LUT errors ──
        lines.append("\n  ① LUT Interpolation Errors  (vs exact Clenshaw, ↓ mejor)")
        lines.append(self._row("  abs_error", self.lut_abs_err.stats()))

        # ── 2. Head orthogonality ──
        lines.append("\n  ② Head Orthogonality  (cosine-sim entre pares; ↓ mejor → más ortogonal)")
        lines.append(self._row("  max |cosim|",  self.head_ortho_max_cosim.stats()))
        lines.append(self._row("  mean |cosim|", self.head_ortho_mean_cosim.stats()))

        # ── 3. Recall similarity ──
        lines.append("\n  ③ Recall Similarity Distribution  (↑ → landmarks más relevantes)")
        lines.append(self._row("  max_sim",       self.recall_max_sim.stats()))
        lines.append(self._row("  inject_count",  self.recall_inject_count.stats()))

        # ── 4. EMA momentum ──
        lines.append("\n  ④ EMA Momentum Actual  (annealing 0.9 → 0.7 durante entrenamiento)")
        lines.append(self._row("  ema_momentum", self.ema_momentum_actual.stats()))

        # ── 5. Bus staleness ──
        lines.append("\n  ⑤ AsyncLightBus Staleness  (steps sin publicar por capa; ↓ mejor)")
        if self.bus_staleness:
            for layer_idx in sorted(self.bus_staleness.keys()):
                s = self.bus_staleness[layer_idx].stats()
                lines.append(self._row(f"  layer {layer_idx}", s))
        else:
            lines.append("    [sin datos de bus]")

        # ── 6. Padding waste ──
        lines.append("\n  ⑥ Padding Waste %  (0% = sin desperdicio)")
        lines.append(self._row("  waste_%", self.padding_waste_pct.stats()))

        # ── 7. Rounding bias ──
        lines.append("\n  ⑦ Stochastic Rounding BF16  (sesgo ideal ≈ 0.0)")
        lines.append(self._row("  bias (E[r-x])",   self.round_bias.stats()))
        lines.append(self._row("  abs_err (E|r-x|)", self.round_abs_err.stats()))

        lines.append("\n" + "─" * 74 + "\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Context manager: activa diagnóstico solo en un bloque
    # ------------------------------------------------------------------

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *_):
        self.disable()


# ============================================================================
# GLOBAL SINGLETON
# ============================================================================

DIAG = OrthoSSMDiagnostics()
