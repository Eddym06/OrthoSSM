"""
download_marin_v2.py — Descarga directa de shards zarr v3 (sharding_indexed + blosc/lz4)
===========================================================================================

El dataset marin-community/fineweb-edu-pretokenized-10B usa Zarr v3 con:
  • sharding_indexed: cada shard file = 134M int64 tokens (~287 MB comprimido)
  • inner chunks: 262K tokens comprimidos con blosc/lz4
  • 75 shards totales (10B tokens) — solo descargamos los primeros ~20 (2.6B)

Salida:
    /workspace/marin_tokens/tokens_00.bin   (uint32, ~1B tokens = 4 GB)
    /workspace/marin_tokens/tokens_01.bin
    /workspace/marin_tokens/tokens_02.bin
    /workspace/marin_tokens/meta.json

Uso:
    python3.10 download_marin_v2.py
    python3.10 download_marin_v2.py --tokens 2.6e9 --out_dir /workspace/marin_tokens
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import blosc
import numpy as np

# ─── Constantes zarr del dataset ─────────────────────────────────────────────
DATASET_BASE = "datasets/marin-community/fineweb-edu-pretokenized-10B"
DATA_PATH    = f"{DATASET_BASE}/train/input_ids/data"

OUTER_CHUNK  = 134_217_728   # tokens por shard file
INNER_CHUNK  =     262_144   # tokens por inner chunk (sub-shard)
N_INNER      = OUTER_CHUNK // INNER_CHUNK   # 512 inner chunks por shard
INDEX_ENTRY  = 16            # bytes por entrada del índice (2 × uint64)
CRC_BYTES    = 4
INDEX_SIZE   = N_INNER * INDEX_ENTRY + CRC_BYTES   # 8,196 bytes al final del shard

# ─── Constantes de salida ─────────────────────────────────────────────────────
OUTPUT_DTYPE    = np.uint32   # IDs Marin ≤ 128K, caben en uint32
BYTES_PER       = 4
OUT_SHARD_SIZE  = 1_000_000_000   # tokens por archivo .bin de salida


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _human(n: int, unit: str = "tok") -> str:
    if n >= 1e9:  return f"{n/1e9:.3f}B {unit}"
    if n >= 1e6:  return f"{n/1e6:.2f}M {unit}"
    if n >= 1e3:  return f"{n/1e3:.1f}K {unit}"
    return f"{n} {unit}"


def _eta(done: int, total: int, elapsed: float) -> str:
    if done == 0: return "??:??"
    rate = done / elapsed
    remaining = (total - done) / rate
    h, r = divmod(int(remaining), 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _bar(done: int, total: int, width: int = 26) -> str:
    frac = min(done / total, 1.0)
    n = int(frac * width)
    return "[" + "#" * n + "." * (width - n) + "]"


# ─── Decodificador de shard zarr v3 sharding_indexed ─────────────────────────

def _decode_shard(shard_bytes: bytes, max_tokens: int | None = None) -> np.ndarray:
    """
    Decodifica un shard file zarr v3 (formato sharding_indexed + blosc/lz4).

    Estructura del shard:
        [inner_chunk_0 (blosc)] [inner_chunk_1 (blosc)] ... [inner_chunk_511 (blosc)]
        [índice: 512 × (offset_u64 + len_u64)] [CRC32C (4 bytes)]

    Devuelve array int64 con hasta max_tokens tokens.
    """
    # El índice está al FINAL del archivo
    index_data = shard_bytes[-INDEX_SIZE:-CRC_BYTES]   # saltar CRC32C

    parts = []
    total = 0
    limit = max_tokens if max_tokens is not None else OUTER_CHUNK

    for i in range(N_INNER):
        if total >= limit:
            break

        offset, length = struct.unpack_from('<QQ', index_data, i * INDEX_ENTRY)

        if offset == 0xFFFFFFFFFFFFFFFF or length == 0:
            # Inner chunk ausente → fill_value = 0
            n = min(INNER_CHUNK, limit - total)
            parts.append(np.zeros(n, dtype=np.int64))
            total += n
            continue

        compressed = shard_bytes[offset: offset + length]
        raw = blosc.decompress(compressed)
        arr = np.frombuffer(raw, dtype='<i8')   # int64 little-endian

        # Truncar si es el último shard parcial
        if total + len(arr) > limit:
            arr = arr[:limit - total]

        parts.append(arr)
        total += len(arr)

    if not parts:
        return np.array([], dtype=np.int64)
    return np.concatenate(parts)


# ─── Descargador principal ────────────────────────────────────────────────────

def download_marin(
    target_tokens: int,
    out_dir: str,
    hf_token: str | None = None,
) -> dict:
    """
    Descarga target_tokens tokens de Marin a out_dir como archivos .bin (uint32).
    """
    from huggingface_hub import HfFileSystem

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fs = HfFileSystem(token=hf_token)

    # ── 1. Listar shards disponibles ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CHIMERA — Descargador Marin (Zarr v3 directo)")
    print(f"  Dataset : marin-community/fineweb-edu-pretokenized-10B")
    print(f"  Target  : {_human(target_tokens)} ({target_tokens * BYTES_PER / 1e9:.1f} GB uint32)")
    print(f"  Salida  : {out_path}/")
    print(f"{'='*60}\n")

    print(f"[info] Listando shards en {DATA_PATH}/c ...")
    items = fs.ls(f"{DATA_PATH}/c", detail=True)

    shard_files = sorted(
        [
            (int(x['name'].split('/')[-1]), x['name'], x.get('size', 0))
            for x in items
            if x['name'].split('/')[-1].isdigit()
        ],
        key=lambda x: x[0],
    )

    n_needed     = (target_tokens + OUTER_CHUNK - 1) // OUTER_CHUNK
    shards_to_dl = shard_files[:n_needed]
    compressed_gb = sum(sz for _, _, sz in shards_to_dl) / 1e9

    print(f"[info] Shards totales  : {len(shard_files)} ({len(shard_files)*OUTER_CHUNK/1e9:.0f}B tokens)")
    print(f"[info] Shards a bajar  : {len(shards_to_dl)}  (índices {shards_to_dl[0][0]}-{shards_to_dl[-1][0]})")
    print(f"[info] Descarga total  : {compressed_gb:.2f} GB (comprimido)")
    print(f"[info] Salida total    : {target_tokens * BYTES_PER / 1e9:.2f} GB (uint32)")
    print()

    # ── 2. Archivos de salida ─────────────────────────────────────────────────
    out_sid    = 0
    out_stoks  = 0
    f_out_path = out_path / f"tokens_{out_sid:02d}.bin"
    f_out      = open(f_out_path, "wb")
    out_paths  = [str(f_out_path)]
    out_sizes  = [0]

    total_written = 0
    t_start = time.perf_counter()

    # ── 3. Iterar shards ──────────────────────────────────────────────────────
    for dl_idx, (shard_num, shard_path_hf, shard_bytes_compressed) in enumerate(shards_to_dl):

        remaining = target_tokens - total_written
        max_tok   = min(OUTER_CHUNK, remaining)

        elapsed = max(time.perf_counter() - t_start, 1e-9)
        tps     = total_written / elapsed
        bar     = _bar(total_written, target_tokens)
        eta     = _eta(total_written, target_tokens, elapsed)

        print(
            f"\r{bar} {100.*total_written/target_tokens:5.1f}%  "
            f"shard {dl_idx+1}/{len(shards_to_dl)} [idx={shard_num}]  "
            f"{_human(total_written)}  "
            f"{tps/1e6:.2f}Mt/s  "
            f"ETA {eta}  "
            f"Descargando...        ",
            end="", flush=True
        )

        # Descargar shard comprimido (~287 MB)
        t0 = time.perf_counter()
        raw_shard = fs.cat(shard_path_hf)
        dl_time   = time.perf_counter() - t0
        dl_mb     = len(raw_shard) / 1e6

        print(
            f"\r{bar} {100.*total_written/target_tokens:5.1f}%  "
            f"shard {dl_idx+1}/{len(shards_to_dl)} [idx={shard_num}]  "
            f"{_human(total_written)}  "
            f"{tps/1e6:.2f}Mt/s  "
            f"ETA {eta}  "
            f"{dl_mb:.0f}MB en {dl_time:.1f}s → decomprimiendo...  ",
            end="", flush=True
        )

        # Decodificar (sharding_indexed + blosc)
        arr_int64 = _decode_shard(raw_shard, max_tokens=max_tok)
        del raw_shard   # liberar RAM inmediatamente

        # int64 → uint32 (IDs Marin ≤ 128K, seguros en uint32)
        arr_u32 = arr_int64.astype(np.uint32)
        del arr_int64

        # Escribir en archivos de salida (shards de 1B tokens)
        pos = 0
        while pos < len(arr_u32):
            space = OUT_SHARD_SIZE - out_stoks
            piece = arr_u32[pos: pos + space]
            f_out.write(piece.tobytes())
            out_sizes[-1] += len(piece)
            out_stoks     += len(piece)
            total_written += len(piece)
            pos           += len(piece)

            if out_stoks >= OUT_SHARD_SIZE and total_written < target_tokens:
                f_out.close()
                out_sid   += 1
                out_stoks  = 0
                f_out_path = out_path / f"tokens_{out_sid:02d}.bin"
                f_out      = open(f_out_path, "wb")
                out_paths.append(str(f_out_path))
                out_sizes.append(0)

        del arr_u32

    f_out.close()
    elapsed_total = time.perf_counter() - t_start

    # ── 4. Meta.json ──────────────────────────────────────────────────────────
    meta = {
        "n_tokens":        total_written,
        "vocab_size":      128002,
        "dtype":           "uint32",
        "bytes_per_token": BYTES_PER,
        "shards": [
            {"path": p, "n_tokens": s}
            for p, s in zip(out_paths, out_sizes)
        ],
        "tokenizer":       "huggingface",
        "hf_model":        "stanford-crfm/marin-tokenizer",
        "dataset":         "marin-community/fineweb-edu-pretokenized-10B",
        "total_bytes":     total_written * BYTES_PER,
        "size_gb":         round(total_written * BYTES_PER / 1e9, 3),
        "download_time_s": round(elapsed_total, 1),
        "avg_tok_per_s":   round(total_written / elapsed_total),
    }
    (out_path / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n\n{'='*60}")
    print(f"  [OK] Descarga y decodificación completada")
    print(f"  Tokens      : {_human(total_written)}")
    print(f"  Disco       : {total_written * BYTES_PER / 1e9:.2f} GB (uint32)")
    print(f"  Tiempo      : {elapsed_total/60:.1f} min  ({total_written/elapsed_total/1e6:.2f} Mt/s)")
    print(f"  Shards salida: {len(out_paths)}")
    print(f"  Meta.json   : {out_path}/meta.json")
    print(f"{'='*60}\n")
    for p, s in zip(out_paths, out_sizes):
        print(f"  {Path(p).name}  ->  {_human(s)}  ({s * BYTES_PER / 1e9:.2f} GB)")
    print()
    return meta


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Descarga 2.6B tokens de Marin (Zarr v3 directo, sin datasets library)"
    )
    parser.add_argument("--tokens",   type=float, default=2.6e9,
                        help="Tokens a descargar (default: 2.6B)")
    parser.add_argument("--out_dir",  default="/workspace/marin_tokens",
                        help="Directorio de salida (default: /workspace/marin_tokens)")
    parser.add_argument("--hf_token", default=None,
                        help="Token HuggingFace (no necesario para dataset público)")
    args = parser.parse_args()

    download_marin(int(args.tokens), args.out_dir, args.hf_token)


if __name__ == "__main__":
    main()
