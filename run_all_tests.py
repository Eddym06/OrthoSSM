#!/usr/bin/env python3
"""
run_all_tests.py — Chimera Battery Test Runner
================================================
Ejecuta todos los tests importantes en orden de profundidad,
captura stdout+stderr, y guarda un informe consolidado en:
    /home/OrthoSSM/chimera_test_results.txt

Tests incluidos (orden ascendente de profundidad):
  [1] test_triton_kernels.py  — Kahan, Lion, GC, Router (13 asserts)
  [2] test_3layer.py          — Stack integration: bus, archive, backward
  [3] test_chimera_full.py    — TTT kernel, scheduler, losses, step()
  [4] precision_tests.py      — Mega-kernel EMA precision vs Triton ref
  [5] deep_analysis.py        — Throughput/VRAM/FD-gradients/stress
  [6] test_v10.py             — Legacy V10 OrthoSSM kernel (T1-T17)
"""

import subprocess
import sys
import os
import time
from datetime import datetime

VENV_PY = "/home/OrthoSSM/venv/bin/python"
CHIMERA  = "/home/OrthoSSM/chimera_experiment"
ROOT     = "/home/OrthoSSM"
OUT_FILE = "/home/OrthoSSM/chimera_test_results.txt"

# (archivo, cwd, timeout_s, descripción, profundidad)
TESTS = [
    (
        f"{CHIMERA}/test_triton_kernels.py",
        CHIMERA,
        120,
        "test_triton_kernels — Kahan precision, Lion constraint, Semantic GC, Router robustez (13 unit-tests con asserts)",
        "ALTA",
    ),
    (
        f"{CHIMERA}/test_3layer.py",
        CHIMERA,
        120,
        "test_3layer — Stack 3 capas: bus_cache crece, landmarks acumulan, backward sin NaN/Inf, retrieval modifica output",
        "ALTA",
    ),
    (
        f"{CHIMERA}/test_chimera_full.py",
        CHIMERA,
        180,
        "test_chimera_full — TTT Triton vs Python ref, WarmupScheduler 3 fases, ChimeraLosses entropy, step() autoregressive",
        "ALTA",
    ),
    (
        f"{ROOT}/precision_tests.py",
        ROOT,
        120,
        "precision_tests — Mega-kernel EMA vs kernel Triton directo, SLR Flash vs PyTorch fallback, BF16 guard",
        "ALTA",
    ),
    (
        f"{CHIMERA}/deep_analysis.py",
        CHIMERA,
        300,
        "deep_analysis — T1 Throughput/VRAM, T2 Estabilidad numérica, T3 Stress 4K tokens, T4 Latencia componentes, T5 Chunked error, T6 Gradientes FD, T7 Proyección 1M tokens, T8 step() autoregresivo",
        "MUY ALTA",
    ),
    (
        f"{ROOT}/test_v10.py",
        ROOT,
        180,
        "test_v10 — Legacy V10 OrthoSSM (sdpc_kernel, sdpc_engine, model.py): T1-T17 incluyendo LUT, gradientes, SLR",
        "MEDIA",
    ),
]


def run_test(script_path, cwd, timeout, label, depth):
    print(f"\n{'─'*72}")
    print(f"  EJECUTANDO: {os.path.basename(script_path)}")
    print(f"  Profundidad: {depth}")
    print(f"  Descripción: {label}")
    print(f"  Timeout: {timeout}s")
    print(f"{'─'*72}")
    sys.stdout.flush()

    t0 = time.perf_counter()
    result = subprocess.run(
        [VENV_PY, script_path],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0

    stdout = result.stdout
    stderr = result.stdout  # not needed separately
    combined = result.stdout
    if result.stderr.strip():
        combined += "\n--- STDERR ---\n" + result.stderr

    # Detectar pass/fail
    rc = result.returncode
    status = "PASS" if rc == 0 else "FAIL"

    # Contar tests individuales si los hay
    lines = result.stdout.split("\n")
    pass_count = sum(1 for l in lines if "✓" in l or "PASS" in l or "ok" in l.lower() and "---" not in l)
    fail_count = sum(1 for l in lines if "✗" in l or "FAIL" in l or "ERROR" in l)

    print(combined)
    print(f"\n  ──► STATUS: {status} | RC={rc} | Tiempo: {elapsed:.1f}s")
    sys.stdout.flush()

    return {
        "script": os.path.basename(script_path),
        "label": label,
        "depth": depth,
        "status": status,
        "rc": rc,
        "elapsed_s": elapsed,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "output": combined,
    }


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 72)
    print(f"  CHIMERA BATTERY TEST — {timestamp}")
    print(f"  GPU: ", end="")
    sys.stdout.flush()

    # Quick GPU check
    import torch
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CPU ONLY")
    print(f"  PyTorch: {torch.__version__}")
    print("=" * 72)

    all_results = []
    total_t0 = time.perf_counter()

    for (script, cwd, timeout, label, depth) in TESTS:
        try:
            r = run_test(script, cwd, timeout, label, depth)
        except subprocess.TimeoutExpired:
            print(f"\n  [TIMEOUT] {script} superó {timeout}s")
            r = {
                "script": os.path.basename(script),
                "label": label,
                "depth": depth,
                "status": "TIMEOUT",
                "rc": -1,
                "elapsed_s": timeout,
                "pass_count": 0,
                "fail_count": 1,
                "output": f"TIMEOUT after {timeout}s",
            }
        except Exception as e:
            print(f"\n  [EXCEPTION] {e}")
            r = {
                "script": os.path.basename(script),
                "label": label,
                "depth": depth,
                "status": "EXCEPTION",
                "rc": -2,
                "elapsed_s": 0,
                "pass_count": 0,
                "fail_count": 1,
                "output": str(e),
            }
        all_results.append(r)

    total_elapsed = time.perf_counter() - total_t0

    # ── Resumen ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RESUMEN FINAL")
    print("=" * 72)
    n_pass = sum(1 for r in all_results if r["status"] == "PASS")
    n_fail = sum(1 for r in all_results if r["status"] != "PASS")
    for r in all_results:
        sym = "✓" if r["status"] == "PASS" else "✗"
        print(f"  {sym} {r['script']:<30} {r['status']:<8} {r['elapsed_s']:6.1f}s  [{r['depth']}]")
    print(f"\n  TOTAL: {n_pass}/{len(all_results)} PASS | Tiempo total: {total_elapsed:.1f}s")

    # ── Guardar TXT ───────────────────────────────────────────────────────────
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"  CHIMERA BATTERY TEST RESULTS\n")
        f.write(f"  Fecha: {timestamp}\n")
        if torch.cuda.is_available():
            f.write(f"  GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
        f.write(f"  PyTorch: {torch.__version__}\n")
        f.write("=" * 80 + "\n\n")

        # Resumen ejecutivo
        f.write("RESUMEN EJECUTIVO\n")
        f.write("─" * 80 + "\n")
        for r in all_results:
            sym = "✓" if r["status"] == "PASS" else "✗"
            f.write(f"  {sym}  {r['script']:<30} {r['status']:<8}  {r['elapsed_s']:6.1f}s  [{r['depth']}]\n")
        f.write(f"\n  TOTAL: {n_pass}/{len(all_results)} PASS  |  Tiempo total: {total_elapsed:.1f}s\n")
        f.write("\n\n")

        # Detalle por test
        for r in all_results:
            f.write("=" * 80 + "\n")
            f.write(f"TEST: {r['script']}\n")
            f.write(f"Descripción: {r['label']}\n")
            f.write(f"Profundidad: {r['depth']}\n")
            f.write(f"Status: {r['status']}  |  RC: {r['rc']}  |  Tiempo: {r['elapsed_s']:.1f}s\n")
            f.write("─" * 80 + "\n")
            f.write(r["output"])
            f.write("\n\n")

    print(f"\n  Resultados guardados en: {OUT_FILE}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
