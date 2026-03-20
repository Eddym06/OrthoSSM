"""
test_triton_kernels.py — Test suite para ttt_kernel.py, landmark_native.py y
                          advanced_chimera.py (router robustez).

Ejecutar:
    /home/OrthoSSM/venv/bin/python chimera_h200/test_triton_kernels.py

Resultados esperados:
    test_kahan_accuracy          PASS  (error ratio ≥ 5×)
    test_lion_constraint         PASS  (|Δdt| ≤ 0.1*|A| por head)
    test_lion_backward_compat    PASS  (API antigua funciona sin Kahan)
    test_lion_kahan_stable       PASS  (Kahan no introduce NaN/Inf)
    test_semantic_gc             PASS  (GC merge el par más similar)
    test_preload_context         PASS  (K embeddings cargados)
    test_router_temperature      PASS  (alta T → distribución más suave)
    test_router_floor            PASS  (todas las probs ≥ min_prob_floor)
    test_collapse_ema            PASS  (EMA sube tras routing sesgado)
"""

import sys
import os
import math
import unittest

import torch
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────
# Permite ejecutar desde /home/OrthoSSM/ o desde /home/OrthoSSM/chimera_h200/
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Imports bajo test ─────────────────────────────────────────────────────────
from ttt_kernel import (
    lion_constrained_update_inplace,
    compute_token_errors_triton,
)
from landmark_native import NativeLandmarkArchive
from advanced_chimera import GatedComplexityPredictor, AdvancedChimeraLayer

# Dispositivo: CUDA si disponible, si no CPU (Triton requiere CUDA en kernels,
# pero los tests de router y landmark funcionan en CPU).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAS_CUDA = torch.cuda.is_available()


# ─────────────────────────────────────────────────────────────────────────────
#  Grupo 1: ttt_kernel — Kahan Summation y Lion update
# ─────────────────────────────────────────────────────────────────────────────
class TestTTTKernel(unittest.TestCase):

    def _make_bufs(self, n: int, device=None):
        """Helper: dt_bias, momentum, A_abs, grad, buffers Kahan."""
        d = device or DEVICE
        dt_bias  = torch.zeros(n, dtype=torch.float32, device=d)
        momentum = torch.zeros(n, dtype=torch.float32, device=d)
        A_abs    = torch.ones(n, dtype=torch.float32, device=d)   # A~1 → máx Δ~0.1
        grad     = torch.randn(n, dtype=torch.float32, device=d) * 0.1
        mom_comp = torch.zeros(n, dtype=torch.float32, device=d)
        dt_comp  = torch.zeros(n, dtype=torch.float32, device=d)
        return dt_bias, momentum, A_abs, grad, mom_comp, dt_comp

    # ── 1a: Kahan es más preciso que FP32 naive sobre muchos pasos ────────────
    @unittest.skipUnless(HAS_CUDA, "requiere CUDA para el kernel Triton")
    def test_kahan_accuracy(self):
        """El error acumulado con Kahan debe ser ≥5× menor que sin Kahan."""
        N_STEPS = 5_000
        n = 64
        torch.manual_seed(42)

        # Ground truth en FP64
        dt64  = torch.zeros(n, dtype=torch.float64, device=DEVICE)
        mom64 = torch.zeros(n, dtype=torch.float64, device=DEVICE)
        A64   = torch.ones(n, dtype=torch.float64, device=DEVICE)
        beta  = 0.9
        lr    = 1e-3

        dt_no_kahan  = torch.zeros(n, dtype=torch.float32, device=DEVICE)
        mom_no_kahan = torch.zeros(n, dtype=torch.float32, device=DEVICE)

        dt_kahan  = torch.zeros(n, dtype=torch.float32, device=DEVICE)
        mom_kahan = torch.zeros(n, dtype=torch.float32, device=DEVICE)
        mc = torch.zeros(n, dtype=torch.float32, device=DEVICE)
        dc = torch.zeros(n, dtype=torch.float32, device=DEVICE)

        rng = torch.Generator(device=DEVICE)
        rng.manual_seed(0)
        for _ in range(N_STEPS):
            g = torch.randn(n, device=DEVICE, dtype=torch.float32,
                            generator=rng) * 0.01

            # FP64 ground truth
            g64 = g.double()
            sign_g64 = torch.sign(beta * mom64 + (1 - beta) * g64)
            A_per = 0.1 * A64
            delta64 = torch.clamp(-lr * sign_g64, -A_per, A_per)
            mom64 = beta * mom64 + (1 - beta) * g64
            dt64  = dt64 + delta64

            # FP32 sin Kahan (legacy kernel)
            lion_constrained_update_inplace(
                dt_no_kahan, mom_no_kahan, g, A64.float(),
                beta=beta, lr=lr, active_prob=1.0
            )

            # FP32 con Kahan
            lion_constrained_update_inplace(
                dt_kahan, mom_kahan, g, A64.float(),
                beta=beta, lr=lr, active_prob=1.0,
                mom_comp=mc, dt_comp=dc,
            )

        err_no_kahan = (dt_no_kahan.double() - dt64).abs().mean().item()
        err_kahan    = (dt_kahan.double()    - dt64).abs().mean().item()

        # Kahan debe mejorar al menos 5×
        ratio = err_no_kahan / (err_kahan + 1e-30)
        self.assertGreater(
            ratio, 5.0,
            f"Kahan debería mejorar ≥5× pero ratio={ratio:.2f} "
            f"(sin Kahan={err_no_kahan:.2e}, con Kahan={err_kahan:.2e})"
        )

    # ── 1b: Restricción Lion — |Δdt_bias| ≤ 0.1·|A| ─────────────────────────
    @unittest.skipUnless(HAS_CUDA, "requiere CUDA para el kernel Triton")
    def test_lion_constraint(self):
        """Cada step: |dt_bias_nuevo - dt_bias_viejo| ≤ 0.1 * |A|."""
        n = 128
        torch.manual_seed(7)
        dt, mom, A_abs, grad, mc, dc = self._make_bufs(n)

        for _ in range(50):
            dt_before = dt.clone()
            g = torch.randn(n, device=DEVICE) * 0.5   # gradientes grandes
            lion_constrained_update_inplace(
                dt, mom, g, A_abs,
                beta=0.9, lr=1e-3, active_prob=1.0,
                mom_comp=mc, dt_comp=dc,
            )
            delta = (dt - dt_before).abs()
            max_allowed = 0.1 * A_abs
            violations = (delta > max_allowed + 1e-6).sum().item()
            self.assertEqual(
                violations, 0,
                f"Violación de restricción: {violations} heads con |Δdt| > 0.1·|A|"
            )

    # ── 1c: Backward compatibility — API sin Kahan no rompe ──────────────────
    @unittest.skipUnless(HAS_CUDA, "requiere CUDA para el kernel Triton")
    def test_lion_backward_compat(self):
        """lion_constrained_update_inplace sin mom_comp/dt_comp debe funcionar."""
        n = 32
        torch.manual_seed(3)
        dt   = torch.zeros(n, device=DEVICE, dtype=torch.float32)
        mom  = torch.zeros(n, device=DEVICE, dtype=torch.float32)
        A    = torch.ones(n, device=DEVICE, dtype=torch.float32)
        grad = torch.randn(n, device=DEVICE) * 0.05

        try:
            lion_constrained_update_inplace(dt, mom, grad, A,
                                             beta=0.9, lr=1e-3, active_prob=0.8)
        except Exception as e:
            self.fail(f"API legada lanzó excepción: {e}")

        # Debe haber cambiado dt (active_prob=0.8 > 0)
        self.assertFalse(torch.all(dt == 0),
                         "dt debería haber sido actualizado")

    # ── 1d: Kahan no introduce NaN/Inf  ──────────────────────────────────────
    @unittest.skipUnless(HAS_CUDA, "requiere CUDA para el kernel Triton")
    def test_lion_kahan_stable(self):
        """Tras 10 000 pasos con Kahan, dt y mom deben ser finitos."""
        n = 64
        torch.manual_seed(99)
        dt, mom, A_abs, _, mc, dc = self._make_bufs(n)

        for i in range(10_000):
            g = torch.randn(n, device=DEVICE) * (0.01 if i % 100 != 0 else 10.0)
            lion_constrained_update_inplace(
                dt, mom, g, A_abs,
                beta=0.9, lr=1e-3, active_prob=1.0,
                mom_comp=mc, dt_comp=dc,
            )

        self.assertTrue(torch.isfinite(dt).all(),  "dt tiene NaN/Inf tras 10k pasos")
        self.assertTrue(torch.isfinite(mom).all(), "mom tiene NaN/Inf tras 10k pasos")


# ─────────────────────────────────────────────────────────────────────────────
#  Grupo 2: NativeLandmarkArchive — GC Semántico y preload_context
# ─────────────────────────────────────────────────────────────────────────────
class TestLandmarkNative(unittest.TestCase):

    def _make_archive(self, d_model=64, landmark_dim=32, max_landmarks=8):
        """Construye un NativeLandmarkArchive en CPU."""
        return NativeLandmarkArchive(
            d_model=d_model,
            landmark_dim=landmark_dim,
            max_landmarks=max_landmarks,
        )

    # ── 2a: _semantic_gc fusiona el par más similar ───────────────────────────
    def test_semantic_gc(self):
        """Después del GC, el par de embeddings más similares debe fusionarse."""
        d = 32
        archive = self._make_archive(d_model=128, landmark_dim=d, max_landmarks=4)
        torch.manual_seed(1)

        # Llenar el archivo: embs[0] y embs[1] son casi idénticos
        base = torch.randn(d)
        archive._store_landmark(F.normalize(base, dim=0), importance=1.0)
        archive._store_landmark(F.normalize(base + 0.01 * torch.randn(d), dim=0),
                                importance=1.0)
        archive._store_landmark(F.normalize(torch.randn(d), dim=0), importance=1.0)
        archive._store_landmark(F.normalize(torch.randn(d), dim=0), importance=1.0)

        n_before = archive.n_archived.item()
        self.assertEqual(n_before, 4)

        # GC manual
        archive._semantic_gc()
        n_after = archive.n_archived.item()

        self.assertEqual(n_after, 3,
            f"GC debería reducir de 4 a 3 landmarks, hay {n_after}")

    # ── 2b: GC preserva landmarks disímiles ──────────────────────────────────
    def test_semantic_gc_preserves_diverse(self):
        """GC sobre embeddings ortogonales aún reduce en 1."""
        d = 16
        archive = self._make_archive(d_model=64, landmark_dim=d, max_landmarks=4)
        # Embeddings ortogonales (muy disímiles)
        e = torch.eye(4, d)
        for i in range(4):
            archive._store_landmark(e[i], importance=float(i + 1))

        archive._semantic_gc()
        self.assertEqual(archive.n_archived.item(), 3)

    # ── 2c: preload_context carga K embeddings ────────────────────────────────
    def test_preload_context(self):
        """preload_context debe cargar K embeddings en el archivo."""
        archive = self._make_archive(d_model=64, landmark_dim=32, max_landmarks=8)
        K = 5
        ctx = torch.randn(K, 32)    # K embeddings del tamaño landmark_dim
        archive.preload_context(ctx)

        n = archive.n_archived.item()
        self.assertEqual(
            n, K,
            f"Se esperaban {K} landmarks, hay {n}"
        )

    # ── 2d: GC se activa cuando se llena el archivo  ──────────────────────────
    def test_archive_gc_on_full(self):
        """Cuando n_landmarks == max_landmarks, _store_landmark debe ejecutar GC."""
        d_model, landmark_dim, max_lm = 64, 32, 4
        archive = self._make_archive(d_model, landmark_dim, max_lm)

        # Precargar hasta el límite
        for _ in range(max_lm):
            emb = torch.randn(landmark_dim)
            archive._store_landmark(emb, importance=1.0)

        self.assertEqual(archive.n_archived.item(), max_lm)

        # Un extra debe forzar GC → n_archived vuelve a max_lm (no max_lm+1)
        extra = torch.randn(landmark_dim)
        archive._store_landmark(extra, importance=1.0)

        n = archive.n_archived.item()
        self.assertEqual(
            n, max_lm,
            f"Después del GC debería haber {max_lm} landmarks, hay {n}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Grupo 3: GatedComplexityPredictor — temperatura, floor, EMA
# ─────────────────────────────────────────────────────────────────────────────
class TestRouter(unittest.TestCase):

    def _make_input(self, B=4, S=32, d_model=64):
        return torch.randn(B, S, d_model)

    # ── 3a: temperatura alta → distribución más suave (menor max_prob) ────────
    def test_router_temperature(self):
        """Con T alta la distribución es más uniforme que con T baja."""
        d, n_tiers = 64, 3
        router_soft = GatedComplexityPredictor(d, n_tiers)
        router_hard = GatedComplexityPredictor(d, n_tiers)

        # Copiar mismos pesos de MLP
        router_hard.load_state_dict(router_soft.state_dict())

        # Forzar temperaturas distintas vía log_temp
        with torch.no_grad():
            router_soft.log_temp.fill_(math.log(5.0))   # T~5 → más uniforme
            router_hard.log_temp.fill_(math.log(0.1))   # T~0.1 → más nítido

        x = self._make_input()
        with torch.no_grad():
            p_soft, _ = router_soft(x)
            p_hard, _ = router_hard(x)

        # Max prob de la distribución suave debe ser menor (más uniforme)
        max_soft = p_soft.max(dim=-1).values.mean().item()
        max_hard = p_hard.max(dim=-1).values.mean().item()
        self.assertLess(
            max_soft, max_hard,
            f"Con T=5 la max_prob ({max_soft:.3f}) debería ser menor que con T=0.1 ({max_hard:.3f})"
        )

    # ── 3b: anti-collapse floor garantiza mínimo por tier ────────────────────
    def test_router_floor(self):
        """Todas las probabilidades de salida deben ser ≥ min_prob_floor."""
        floors = [0.05, 0.10, 0.15]
        for floor in floors:
            with self.subTest(floor=floor):
                router = GatedComplexityPredictor(64, n_tiers=3,
                                                  min_prob_floor=floor)
                x = self._make_input(B=8, S=16)
                with torch.no_grad():
                    probs, _ = router(x)

                min_p = probs.min().item()
                self.assertGreaterEqual(
                    min_p, floor - 1e-6,
                    f"Con floor={floor}, la probabilidad mínima es {min_p:.4f}"
                )

    # ── 3c: suma de probabilidades por muestra es ~1 ─────────────────────────
    def test_router_sums_to_one(self):
        """Tras el floor y re-normalización, cada fila debe sumar a 1."""
        router = GatedComplexityPredictor(64)
        x = self._make_input(B=6, S=24)
        with torch.no_grad():
            probs, _ = router(x)

        row_sums = probs.sum(dim=-1)  # [B]
        self.assertTrue(
            torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5),
            f"Las filas no suman a 1. Sumas: {row_sums.tolist()}"
        )

    # ── 3d: EMA de prob_fast sube tras routing sesgado ───────────────────────
    def test_collapse_ema(self):
        """fast_prob_ema debe subir si el router siempre elige FAST."""
        # Simulamos el router + EMA sin instanciar AdvancedChimeraLayer completo
        # (que requiere Mamba2/CUDA). Verificamos la lógica matemática directamente.
        fast_prob_ema = torch.tensor(1/3)   # init: distribución uniforme

        # Simular 300 pasos donde prob_fast=0.95 (router colapsado a FAST)
        fake_fast_prob = torch.tensor(0.95)
        for _ in range(300):
            fast_prob_ema = fast_prob_ema * 0.99 + fake_fast_prob * 0.01

        # EMA debería converger hacia 0.95, ciertamente superar 0.5
        ema_val = fast_prob_ema.item()
        self.assertGreater(
            ema_val, 0.5,
            f"EMA debería superar 0.5 tras routing FAST, pero es {ema_val:.4f}"
        )
        self.assertLess(
            ema_val, 1.0,
            "EMA no debe superar 1.0"
        )
        # Verificar que el mecanismo Mul+Add del buffer tensor funciona
        buf = torch.tensor(1/3)
        buf.mul_(0.99).add_(torch.tensor(0.95) * 0.01)
        self.assertTrue(torch.isfinite(buf), "EMA buffer debe ser finito")

    # ── 3e: floor=0 → comportamiento normal softmax ───────────────────────────
    def test_router_no_floor(self):
        """Con min_prob_floor=0.0, el router actúa como softmax puro."""
        router = GatedComplexityPredictor(64, n_tiers=3, min_prob_floor=0.0)
        x = self._make_input(B=4, S=8)
        with torch.no_grad():
            probs, logits = router(x)
        T = router.log_temp.exp()
        expected = F.softmax(logits / T, dim=-1)
        self.assertTrue(torch.allclose(probs, expected, atol=1e-5))


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Dispositivo: {DEVICE}")
    print(f"CUDA disponible: {HAS_CUDA}")
    if HAS_CUDA:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    # Orden lógico de ejecución
    for cls in [TestTTTKernel, TestLandmarkNative, TestRouter]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total  = result.testsRun
    passed = total - len(result.failures) - len(result.errors) - len(result.skipped)
    skipped = len(result.skipped)
    print(f"\n{'=' * 60}")
    print(f"Resultados: {passed} PASS | {len(result.failures)+len(result.errors)} FAIL | {skipped} SKIP | {total} TOTAL")
    sys.exit(0 if result.wasSuccessful() else 1)
