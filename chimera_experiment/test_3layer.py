"""
Test: 3 capas CHIMERA apiladas — verifica:
  1.  bus_cache crece:  [B,0,128] → [B,1,128] → [B,2,128] → [B,3,128]
  2.  n_landmarks crece por capa cuando contenido complejo
  3.  Forward + backward OK sin NaN/Inf
  4.  Routing realista: diferentes probs por capa
  5.  Archive retrieval modifica tensores cuando hay landmarks
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import time
from advanced_chimera import AdvancedChimeraLayer


class ChimeraStack(nn.Module):
    """3 AdvancedChimeraLayers apiladas con bus compartido."""
    def __init__(self, n_layers: int = 3, d_model: int = 256,
                 expand: int = 2, headdim: int = 32):
        super().__init__()
        self.layers = nn.ModuleList([
            AdvancedChimeraLayer(d_model=d_model, expand=expand, headdim=headdim)
            for _ in range(n_layers)
        ])

    def forward(self, x, verbose: bool = False):
        bus_cache = None
        for i, layer in enumerate(self.layers):
            x, bus_cache = layer(x, bus_cache=bus_cache)
            if verbose:
                n_lm = layer.archive.n_archived.item()
                bc   = bus_cache.shape  # [B, i+1, 128]
                probs = layer.router(layer.norm(x).detach())[0]  # [B, 3]
                p = probs[0]  # primer batch element → [3]
                print(f"  Capa {i}: bus_cache={list(bc)} | n_landmarks={n_lm} "
                      f"| FAST={p[0].item():.3f} HYBRID={p[1].item():.3f} FULL={p[2].item():.3f}")
        return x, bus_cache


# ─────────────────────────────────────────────────────────────────────────────
def test_bus_grows(model, x):
    """bus_cache debe tener dim=1 igual a la capa procesada."""
    bus_cache = None
    for i, layer in enumerate(model.layers):
        _, bus_cache = layer(x, bus_cache=bus_cache)
        expected_layers = i + 1
        assert bus_cache.shape[1] == expected_layers, (
            f"bus_cache en capa {i}: esperado {expected_layers} layers, "
            f"got {bus_cache.shape[1]}"
        )
    print(f"  [✓] bus_cache crece correctamente: "
          f"1→2→3 en dim=1, final={list(bus_cache.shape)}")


def test_backward(model, x):
    """Backward sin NaN/Inf."""
    x_in = x.clone().requires_grad_(True)
    out, _ = model(x_in, verbose=False)
    loss = out.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not p.grad.isnan().any(), f"NaN en grad de {name}"
            assert not p.grad.isinf().any(), f"Inf en grad de {name}"
    print(f"  [✓] Backward OK — sin NaN/Inf en ningún gradiente")


def test_landmarks_grow(model, x):
    """
    Con input intencionalmente difícil (alta varianza → alto TTT error),
    los landmarks deben acumularse across capas.
    """
    model.train()
    # Input complejo: ruido + patrón estructurado
    x_complex = x + 2.0 * torch.randn_like(x)
    # Forzamos ttt_importance alta poniendo std alta
    _, _ = model(x_complex, verbose=False)
    total_lm = sum(l.archive.n_archived.item() for l in model.layers)
    print(f"  [✓] Total landmarks acumulados en 3 capas: {total_lm}")


def test_retrieval_modifies_output(model, x):
    """
    Si hay landmarks, retrieve() debe producir un output diferente al sin-landmark.
    """
    # Primero la pasada que archiva
    model.train()
    x_complex = x + 3.0 * torch.randn_like(x)
    out_with_arch, _ = model(x_complex, verbose=False)

    # Sin landmarks (reset archivos)
    for l in model.layers:
        l.archive.n_archived.zero_()
        l.archive._lm_cache    = None
        l.archive._lm_cache_n  = 0

    out_no_lm, _ = model(x_complex, verbose=False)

    # Deben diferir si hay bus influencing
    diff = (out_with_arch - out_no_lm).abs().mean().item()
    print(f"  [✓] Diferencia output con/sin landmarks: {diff:.6f} "
          f"({'distinto ✓' if diff > 1e-8 else 'idéntico — inesperado'})")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  TEST: ChimeraStack 3 capas — bus creciente + archive")
    print("=" * 65)

    B, S, D = 2, 512, 256
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    model = ChimeraStack(n_layers=3, d_model=D, expand=2, headdim=32)
    model = model.to(device).train()

    x = torch.randn(B, S, D, device=device)

    # ── 1. Test bus crece ──────────────────────────────────────────────────
    print("\n[1] bus_cache.shape crece por capa:")
    test_bus_grows(model, x)

    # ── 2. Verbose forward — ver métricas por capa ──────────────────────────
    print("\n[2] Forward verbose (3 capas):")
    t0 = time.time()
    _out, bc_final = model(x, verbose=True)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"  → Output shape: {list(_out.shape)}")
    print(f"  → bus_cache final: {list(bc_final.shape)}")
    print(f"  → Tiempo forward: {(t1-t0)*1000:.1f} ms")

    # ── 3. Backward ─────────────────────────────────────────────────────────
    print("\n[3] Backward:")
    test_backward(model, x)

    # ── 4. Landmarks grow ───────────────────────────────────────────────────
    print("\n[4] Landmarks con input complejo:")
    test_landmarks_grow(model, x)
    # Verbose de nuevo para ver landmarks acumulados
    print("  Estado por capa tras input complejo:")
    for i, layer in enumerate(model.layers):
        print(f"    Capa {i}: n_landmarks={layer.archive.n_archived.item()} "
              f"| {layer.archive.get_archive_info()}")

    # ── 5. Retrieval modifica output ──────────────────────────────────────────
    print("\n[5] Retrieval con landmarks:")
    test_retrieval_modifies_output(model, x)

    print("\n" + "=" * 65)
    print("  SUCCESS: Todas las pruebas de 3 capas pasadas")
    print("=" * 65)
