"""
download_marin.py — Descargador streaming de marin-community/fineweb-edu-pretokenized-10B
==========================================================================================

Estrategia:
  • Streaming desde HuggingFace → RAM → disco /workspace/
  • NUNCA descarga los 10B completos: corta en 2.6B tokens
  • Almacena en uint32 (vocab Marin = 128K > 65535, requiere uint32)
  • Escribe meta.json compatible con train_h200_elite.py
  • Opcionalmente transfiere a HBM (GPU) al final

Uso:
    python3 download_marin.py
    python3 download_marin.py --tokens 2.6e9 --out_dir /workspace --pin_hbm

Salida:
    /workspace/marin_tokens/tokens_00.bin    ← hasta 1B tokens por shard (uint32)
    /workspace/marin_tokens/tokens_01.bin    ← segundo shard si > 1B
    /workspace/marin_tokens/tokens_02.bin    ← tercer shard si > 2B
    /workspace/marin_tokens/meta.json        ← stats + dtype para el trainer
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ─── Constantes ──────────────────────────────────────────────────────────────
DATASET_ID   = "marin-community/fineweb-edu-pretokenized-10B"
TARGET_TOKS  = 2_600_000_000   # 2.6B tokens (ajustable con --tokens)
SHARD_SIZE   = 1_000_000_000   # 1B tokens por shard
BUFFER_SIZE  =     1_000_000   # flush al disco cada 1M tokens
DTYPE        = np.uint32       # 4 bytes/tok — necesario para vocab 128K
BYTES_PER    = 4

# Nombres de columna que pueden tener los tokens (se auto-detecta)
_TOKEN_COLUMNS = ["token_ids", "input_ids", "tokens", "ids", "text_tokens"]


# ─── Detección de columna de tokens ──────────────────────────────────────────

def _detect_token_column(sample: dict, verbose: bool = True) -> str:
    """Detecta autom\u00e1ticamente en qu\u00e9 columna est\u00e1n los tokens del dataset."""
    for col in _TOKEN_COLUMNS:
        if col in sample:
            val = sample[col]
            if isinstance(val, (list, tuple)) and len(val) > 0:
                if isinstance(val[0], int):
                    if verbose:
                        print(f"[info] Columna de tokens detectada: '{col}'  "
                              f"(primer valor: {val[0]}, longitud muestra: {len(val)})")
                    return col
    # Última opción: mostrar todas las columnas disponibles y fallar claramente
    print(f"[ERROR] No se encontró columna de tokens.")
    print(f"        Columnas disponibles: {list(sample.keys())}")
    print(f"        Columnas buscadas:    {_TOKEN_COLUMNS}")
    print(f"        Añade el nombre correcto a _TOKEN_COLUMNS en el script.")
    sys.exit(1)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _human(n: int, unit: str = "tok") -> str:
    if n >= 1e9:
        return f"{n/1e9:.3f}B {unit}"
    if n >= 1e6:
        return f"{n/1e6:.2f}M {unit}"
    if n >= 1e3:
        return f"{n/1e3:.1f}K {unit}"
    return f"{n} {unit}"


def _eta(done: int, total: int, elapsed: float) -> str:
    if done == 0:
        return "??:??"
    rate = done / elapsed
    remaining = (total - done) / rate
    h, r = divmod(int(remaining), 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _bar(done: int, total: int, width: int = 30) -> str:
    frac = min(done / total, 1.0)
    filled = int(frac * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


# ─── Pipeline principal ──────────────────────────────────────────────────────

def download_marin(
    target_tokens: int,
    out_dir: str,
    shard_size: int = SHARD_SIZE,
    pin_hbm: bool = False,
    hf_token: str | None = None,
    verbose: bool = True,
):
    """
    Descarga target_tokens tokens del dataset Marin en modo streaming.

    No descarga el dataset completo (10B) — corta exactamente en target_tokens.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*68}")
        print(f"  CHIMERA — Descargador Marin Dataset (streaming)")
        print(f"  Dataset  : {DATASET_ID}")
        print(f"  Target   : {_human(target_tokens)} tokens ({target_tokens * BYTES_PER / 1e9:.2f} GB en uint32)")
        print(f"  Salida   : {out_path}/")
        print(f"  dtype    : uint32 (vocab Marin = 128K > 65535)")
        print(f"{'='*68}\n")

    # ── 1. Cargar dataset en modo streaming ──────────────────────────────────
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' no está instalado.")
        print("        Ejecutar: pip install datasets")
        sys.exit(1)

    load_kwargs: dict = dict(split="train", streaming=True, trust_remote_code=True)
    if hf_token:
        load_kwargs["token"] = hf_token

    print(f"[download] Conectando con HuggingFace ({DATASET_ID})...")
    t_load = time.perf_counter()
    try:
        ds = load_dataset(DATASET_ID, **load_kwargs)
    except Exception as e:
        print(f"[ERROR] load_dataset falló: {e}")
        print("  Verifica:")
        print("  1. Conexión a internet activa")
        print("  2. pip install datasets>=2.14.0")
        print("  3. Si el dataset es privado: --hf_token TU_TOKEN")
        sys.exit(1)

    print(f"[download] Dataset conectado en {time.perf_counter()-t_load:.1f}s\n")

    # ── 2. Detectar columna de tokens (de la primera muestra) ────────────────
    col_name: str | None = None

    # ── 3. Abrir primer shard ────────────────────────────────────────────────
    shard_idx   = 0
    shard_toks  = 0
    shard_paths = []
    shard_sizes = []

    def _open_shard(idx: int):
        p = out_path / f"tokens_{idx:02d}.bin"
        return open(p, "wb"), str(p)

    f_out, shard_path = _open_shard(shard_idx)

    buffer:    list[int] = []
    total_toks = 0
    t_start    = time.perf_counter()
    n_samples  = 0
    last_print = 0.0

    # ── 4. Iterar samples ─────────────────────────────────────────────────────
    try:
        for sample in ds:
            n_samples += 1

            # Detectar columna en la primera muestra
            if col_name is None:
                col_name = _detect_token_column(sample, verbose=verbose)

            tokens: list[int] = sample[col_name]

            # Añadir al buffer
            buffer.extend(tokens)

            # Flush al disco cada BUFFER_SIZE tokens
            while len(buffer) >= BUFFER_SIZE:
                # ¿Cuántos tokens necesitamos aún?
                remaining = target_tokens - total_toks
                can_write = min(BUFFER_SIZE, remaining)

                chunk = buffer[:can_write]
                buffer = buffer[can_write:]

                # Distribuir en shards
                written_in_chunk = 0
                while written_in_chunk < len(chunk):
                    space_in_shard = shard_size - shard_toks
                    piece = chunk[written_in_chunk: written_in_chunk + space_in_shard]
                    arr = np.array(piece, dtype=DTYPE)
                    f_out.write(arr.tobytes())
                    shard_toks     += len(piece)
                    total_toks     += len(piece)
                    written_in_chunk += len(piece)

                    # Cerrar shard lleno y abrir el siguiente
                    if shard_toks >= shard_size:
                        f_out.close()
                        shard_paths.append(shard_path)
                        shard_sizes.append(shard_toks)
                        shard_idx   += 1
                        shard_toks   = 0
                        f_out, shard_path = _open_shard(shard_idx)

                # ── Print progreso ────────────────────────────────────────
                now = time.perf_counter()
                if now - last_print >= 2.0:
                    elapsed = now - t_start
                    tps     = total_toks / elapsed if elapsed > 0 else 0
                    pct     = 100.0 * total_toks / target_tokens
                    eta_str = _eta(total_toks, target_tokens, elapsed)
                    bar     = _bar(total_toks, target_tokens)
                    gb_done = total_toks * BYTES_PER / 1e9

                    print(
                        f"\r{bar} {pct:5.1f}%  "
                        f"{_human(total_toks)}  "
                        f"{tps/1e6:.2f}M tok/s  "
                        f"{gb_done:.2f}GB  "
                        f"ETA {eta_str}  "
                        f"muestras={n_samples:,}",
                        end="", flush=True
                    )
                    last_print = now

                if total_toks >= target_tokens:
                    break

            if total_toks >= target_tokens:
                break

        # ── Flush buffer residual ─────────────────────────────────────────────
        if total_toks < target_tokens and buffer:
            remaining = target_tokens - total_toks
            chunk = buffer[:remaining]
            written_in_chunk = 0
            while written_in_chunk < len(chunk):
                space_in_shard = shard_size - shard_toks
                piece = chunk[written_in_chunk: written_in_chunk + space_in_shard]
                arr = np.array(piece, dtype=DTYPE)
                f_out.write(arr.tobytes())
                shard_toks     += len(piece)
                total_toks     += len(piece)
                written_in_chunk += len(piece)
                if shard_toks >= shard_size:
                    f_out.close()
                    shard_paths.append(shard_path)
                    shard_sizes.append(shard_toks)
                    shard_idx   += 1
                    shard_toks   = 0
                    f_out, shard_path = _open_shard(shard_idx)

    finally:
        # Cerrar últim shard siempre, incluso si hubo excepción
        if shard_toks > 0:
            f_out.close()
            shard_paths.append(shard_path)
            shard_sizes.append(shard_toks)
        elif shard_toks == 0 and shard_idx > 0:
            f_out.close()   # shard vacío
        else:
            f_out.close()
            if shard_toks > 0:
                shard_paths.append(shard_path)
                shard_sizes.append(shard_toks)

    # ── 5. Meta.json ──────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start

    # Estimar vocab_size real (escanear primer shard — rápido con memmap)
    if shard_paths:
        sample_arr   = np.fromfile(shard_paths[0], dtype=DTYPE)
        vocab_seen   = int(sample_arr.max()) + 1 if len(sample_arr) > 0 else 128000
    else:
        vocab_seen = 128000

    meta = {
        "n_tokens":      total_toks,
        "vocab_size":    vocab_seen,
        "dtype":         "uint32",
        "bytes_per_token": BYTES_PER,
        "shards":        [{"path": p, "n_tokens": s} for p, s in zip(shard_paths, shard_sizes)],
        "tokenizer":     "huggingface",
        "hf_model":      "stanford-crfm/marin-tokenizer",
        "dataset":       DATASET_ID,
        "total_bytes":   total_toks * BYTES_PER,
        "size_gb":       round(total_toks * BYTES_PER / 1e9, 3),
        "download_time_s": round(elapsed, 1),
        "avg_tok_per_s":   round(total_toks / elapsed),
    }
    meta_path = out_path / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n\n{'='*68}")
    print(f"  [OK] Descarga completada")
    print(f"  Tokens       : {_human(total_toks)}")
    print(f"  Tamaño disco : {total_toks * BYTES_PER / 1e9:.2f} GB (uint32)")
    print(f"  Shards       : {len(shard_paths)}")
    print(f"  Tiempo       : {elapsed/60:.1f} min  ({total_toks/elapsed/1e6:.2f} M tok/s)")
    print(f"  Vocab visto  : {vocab_seen:,}")
    print(f"  Meta.json    : {meta_path}")
    print(f"{'='*68}\n")
    for p, s in zip(shard_paths, shard_sizes):
        print(f"  {Path(p).name}  →  {_human(s)}  ({s*BYTES_PER/1e9:.2f} GB)")

    # ── 6. Pin a HBM (opcional) ──────────────────────────────────────────────
    if pin_hbm:
        _pin_to_hbm(shard_paths, out_path, total_toks, verbose=verbose)

    return meta


# ─── Pin a HBM ───────────────────────────────────────────────────────────────

def _pin_to_hbm(
    shard_paths: list[str],
    out_path: Path,
    total_toks: int,
    verbose: bool = True,
):
    """
    Carga todos los shards en GPU HBM y guarda tokens_hbm.pt.
    Para H200: 2.6B × 4B (uint32→int32) = 10.4 GB — cabe fácil en 141 GB HBM.
    El trainer lo lee con torch.load(map_location='cuda').
    """
    try:
        import torch
    except ImportError:
        print("[pin_hbm] PyTorch no disponible — omitiendo.")
        return

    if not torch.cuda.is_available():
        print("[pin_hbm] CUDA no disponible — omitiendo pin_hbm.")
        print("[pin_hbm] Usa --pin_hbm solo en RunPod con GPU activa.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    needed_gb = total_toks * 4 / 1e9  # int32 = 4 bytes

    if needed_gb > vram_gb * 0.5:
        print(f"[pin_hbm] ADVERTENCIA: {needed_gb:.1f} GB necesarios, "
              f"{vram_gb:.0f} GB disponibles en {gpu_name}.")
        print(f"[pin_hbm] El modelo también necesita VRAM — considera no usar --pin_hbm "
              f"si no hay espacio suficiente.")

    print(f"\n[pin_hbm] Cargando {_human(total_toks)} tokens a {gpu_name} ({vram_gb:.0f} GB VRAM)...")
    print(f"[pin_hbm] Espacio requerido: {needed_gb:.2f} GB (int32 en GPU)")

    tensors = []
    for p in shard_paths:
        t0 = time.perf_counter()
        # uint32 → int32: safe porque los IDs de Marin (128K) caben en int32
        arr = np.fromfile(p, dtype=np.uint32).astype(np.int32)
        t   = torch.from_numpy(arr).cuda()
        tensors.append(t)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  {Path(p).name}  →  GPU  "
              f"({t.nbytes/1e9:.2f} GB)  "
              f"{elapsed_ms:.0f}ms")

    hbm_path = out_path / "tokens_hbm.pt"
    print(f"[pin_hbm] Guardando tokens_hbm.pt...")
    torch.save(tensors, str(hbm_path))
    total_gb = sum(t.nbytes for t in tensors) / 1e9
    print(f"[pin_hbm] ✓ {hbm_path}  ({total_gb:.2f} GB en GPU)")

    # VRAM usada
    used_gb = torch.cuda.memory_allocated() / 1e9
    print(f"[pin_hbm] VRAM usada ahora: {used_gb:.2f} / {vram_gb:.0f} GB")


# ─── Verificación de disco ────────────────────────────────────────────────────

def _check_disk_space(out_dir: str, needed_gb: float) -> bool:
    import shutil
    total, used, free = shutil.disk_usage(out_dir)
    free_gb = free / 1e9
    print(f"[disk] /workspace: {free_gb:.1f} GB libres — necesarios: {needed_gb:.1f} GB")
    if free_gb < needed_gb * 1.05:
        print(f"[disk] ADVERTENCIA: Espacio ajustado. "
              f"Reduce --tokens si el disco se llena.")
        return False
    return True


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Descarga 2.6B tokens de Marin en modo streaming a /workspace/"
    )
    parser.add_argument(
        "--tokens",   type=float, default=2.6e9,
        help="Número de tokens a descargar (default: 2.6B = 2.6e9)"
    )
    parser.add_argument(
        "--out_dir",  default="/workspace/marin_tokens",
        help="Directorio de salida (default: /workspace/marin_tokens)"
    )
    parser.add_argument(
        "--shard_size", type=int, default=1_000_000_000,
        help="Tokens máximos por shard (default: 1B)"
    )
    parser.add_argument(
        "--pin_hbm",  action="store_true",
        help="Cargar tokens a GPU HBM al terminar (requiere CUDA, ocupa ~10.4 GB)"
    )
    parser.add_argument(
        "--hf_token", default=None,
        help="HuggingFace token si el dataset es privado (normalmente no necesario)"
    )
    args = parser.parse_args()

    target = int(args.tokens)

    # Verificar espacio en disco
    needed_gb = target * BYTES_PER / 1e9
    _check_disk_space(args.out_dir, needed_gb)

    meta = download_marin(
        target_tokens = target,
        out_dir       = args.out_dir,
        shard_size    = args.shard_size,
        pin_hbm       = args.pin_hbm,
        hf_token      = args.hf_token,
        verbose       = True,
    )

    print(f"\n{'='*68}")
    print(f"  PRÓXIMOS PASOS:")
    print(f"{'='*68}")
    print(f"\n  1) Cargar a HBM y entrenar (en RunPod):")
    print(f"     python3 train_h200_elite.py \\")
    print(f"         --data_dir {args.out_dir} \\")
    print(f"         --model 125M \\")
    print(f"         --vocab {meta['vocab_size']} \\")
    print(f"         --batch 128 \\")
    print(f"         --compile \\")
    print(f"         --hbm_dataset \\")
    print(f"         --total_tokens {args.tokens:.0e} \\")
    print(f"         --ckpt_dir /workspace/ckpt_chimera \\")
    print(f"         --log_every 10")
    print(f"\n  2) Al terminar, exportar modelo a tu PC local:")
    print(f"     scp runpod:/workspace/ckpt_chimera/latest.pt ~/chimera_model.pt")
    print(f"\n  3) Test de inferencia local (RTX 4050):")
    print(f"     python3 inference_test.py --ckpt ~/chimera_model.pt")
    print(f"\n{'='*68}\n")


if __name__ == "__main__":
    main()
