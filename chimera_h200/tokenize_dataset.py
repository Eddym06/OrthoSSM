"""
tokenize_dataset.py — tokenización binaria ultra-rápida para chimera_h200
=========================================================================

Pipeline CPU-only, paralelizado. Convierte directorios de texto plano
a un único archivo binario int16 listo para loadear directo a HBM (H200).

Características:
  • Tokenizador: tiktoken (cl100k_base) o sentencepiece custom
  • Paralelismo: multiprocessing.Pool → escala a todos los núcleos
  • Salida: tokens.bin — int16 raw, sin header, sin padding
  • Carga HBM: flag --pin_hbm genera tokens_cuda.pt (torch.save) listo
    para torch.load(map_location='cuda') en el tráiner → zero PCIe latency
  • Compatible con vocab ≤ 65_535 (int16 suficiente para GPT-4o / Llama 3)

Uso básico:
    python tokenize_dataset.py \\
        --data_dir /data/raw_text \\
        --out_dir  /data/tokens \\
        --tokenizer cl100k_base \\
        --workers 16

Uso con SentencePiece custom:
    python tokenize_dataset.py \\
        --data_dir /data/raw_text \\
        --out_dir  /data/tokens \\
        --tokenizer sentencepiece \\
        --sp_model /path/to/tokenizer.model \\
        --workers 16

Carga HBM (requiere GPU al momento de tokenizar — opcional):
    python tokenize_dataset.py ... --pin_hbm
"""

from __future__ import annotations

import argparse, math, os, struct, sys, time
from pathlib import Path
import multiprocessing as mp
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Inicialización del tokenizador (en worker, para evitar pickling de objetos C++)
# ─────────────────────────────────────────────────────────────────────────────

_tok_cache   = {}   # por PID — cada worker tiene su propia instancia
_dtype_cache = {}   # por PID — np.uint16 o np.uint32 según vocab


def _get_storage_dtype(tokenizer_name: str, sp_model_path: str | None, hf_model: str | None) -> type:
    """
    Determina el dtype de almacenamiento óptimo para el vocabulario dado.
    Regla: vocab ≤ 65535 → uint16 (2 bytes/tok); vocab > 65535 → uint32 (4 bytes/tok).
    """
    if tokenizer_name == "cl100k_base":
        return np.uint32   # 100,277 vocab > uint16_max
    elif tokenizer_name == "o200k_base":
        return np.uint32   # 200,019 vocab > uint16_max
    elif tokenizer_name == "sentencepiece" and sp_model_path is not None:
        try:
            import sentencepiece as spm
            tok = spm.SentencePieceProcessor()
            tok.Load(sp_model_path)
            return np.uint32 if tok.get_piece_size() > 65535 else np.uint16
        except Exception:
            return np.uint16
    elif tokenizer_name == "huggingface" and hf_model is not None:
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(hf_model, use_fast=True)
        return np.uint32 if hf_tok.vocab_size > 65535 else np.uint16
    return np.uint16


def _init_tokenizer(tokenizer_name: str, sp_model_path: str | None, hf_model: str | None = None):
    """Llamado una vez por worker process."""
    global _tok_cache, _dtype_cache
    pid = os.getpid()
    if pid in _tok_cache:
        return
    if tokenizer_name == "cl100k_base":
        try:
            import tiktoken
            _tok_cache[pid] = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            raise ImportError("tiktoken no instalado. Ejecutar: pip install tiktoken")
        _dtype_cache[pid] = np.uint32   # 100,277 vocab
    elif tokenizer_name == "o200k_base":
        try:
            import tiktoken
            _tok_cache[pid] = tiktoken.get_encoding("o200k_base")
        except ImportError:
            raise ImportError("tiktoken no instalado. Ejecutar: pip install tiktoken")
        _dtype_cache[pid] = np.uint32   # 200,019 vocab
    elif tokenizer_name == "sentencepiece":
        if sp_model_path is None:
            raise ValueError("--sp_model requerido para tokenizer=sentencepiece")
        try:
            import sentencepiece as spm
            tok = spm.SentencePieceProcessor()
            tok.Load(sp_model_path)
            _tok_cache[pid] = tok
            _dtype_cache[pid] = np.uint32 if tok.get_piece_size() > 65535 else np.uint16
        except ImportError:
            raise ImportError("sentencepiece no instalado. Ejecutar: pip install sentencepiece")
    elif tokenizer_name == "huggingface":
        if hf_model is None:
            raise ValueError("--hf_model requerido para tokenizer=huggingface (ej. stanford-crfm/marin-tokenizer)")
        try:
            from transformers import AutoTokenizer
            hf_tok = AutoTokenizer.from_pretrained(hf_model, use_fast=True)
            _tok_cache[pid] = hf_tok
            _dtype_cache[pid] = np.uint32 if hf_tok.vocab_size > 65535 else np.uint16
        except ImportError:
            raise ImportError("transformers no instalado. Ejecutar: pip install transformers")
    else:
        raise ValueError(f"Tokenizador desconocido: {tokenizer_name!r}")


def _tokenize_one(args: tuple) -> tuple[str, int]:
    """
    Worker: tokeniza un archivo de texto y escribe uint16/uint32 en un .tmp binario.
    Retorna (tmp_path, n_tokens).
    """
    filepath, tmp_dir, tokenizer_name, sp_model_path, hf_model, storage_dtype = args
    _init_tokenizer(tokenizer_name, sp_model_path, hf_model)

    pid = os.getpid()
    tok = _tok_cache[pid]
    dtype = _dtype_cache.get(pid, np.dtype(storage_dtype))

    # Leer texto
    try:
        text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return (None, 0)

    if not text.strip():
        return (None, 0)

    # Tokenizar
    if tokenizer_name in ("cl100k_base", "o200k_base"):
        ids = tok.encode_ordinary(text)          # list[int] — sin special tokens
    elif tokenizer_name == "huggingface":
        ids = tok.encode(text, add_special_tokens=False)  # list[int] sin BOS/EOS
    else:
        ids = tok.encode(text, out_type=int)     # SentencePiece

    if not ids:
        return (None, 0)

    arr = np.array(ids, dtype=dtype)
    tmp_path = str(Path(tmp_dir) / f"{os.getpid()}_{hash(filepath)}.tmp")
    arr.tofile(tmp_path)
    return (tmp_path, len(ids))


# ─────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def _discover_files(data_dir: str, extensions: list[str]) -> list[str]:
    exts = set(extensions)
    files = []
    for root, dirs, fnames in os.walk(data_dir):
        # Saltar directorios ocultos / __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for fn in fnames:
            if any(fn.endswith(e) for e in exts):
                files.append(os.path.join(root, fn))
    return sorted(files)


def _human(n: int) -> str:
    for unit, thresh in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if n >= thresh:
            return f"{n/thresh:.2f}{unit}"
    return str(n)


def tokenize_dataset(
    data_dir: str,
    out_dir: str,
    tokenizer: str = "cl100k_base",
    sp_model: str | None = None,
    hf_model: str | None = None,
    workers: int | None = None,
    extensions: list[str] | None = None,
    shard_size: int = 1_000_000_000,   # 1B tokens por shard
    pin_hbm: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Tokeniza data_dir y escribe el resultado en out_dir.

    Salida:
        {out_dir}/tokens_00.bin  — uint16 o uint32 raw, n_tokens enteros
        {out_dir}/tokens_01.bin  — (si hay más de shard_size tokens)
        {out_dir}/meta.json      — stats: n_tokens, vocab_size, dtype, shard_sizes

    Si pin_hbm=True además:
        {out_dir}/tokens_hbm.pt  — torch.save([tensor_shard0, tensor_shard1, ...])
                                    dtype=torch.int32 (CUDA no tiene uint16/uint32 nativo)

    Nota: Marin tokenizer (vocab=128K) usa uint32; cl100k_base (100,277) también.
    """
    if extensions is None:
        extensions = [".txt", ".md", ".jsonl", ".json", ".csv"]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tmp_dir  = out_path / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    files = _discover_files(data_dir, extensions)
    if not files:
        raise FileNotFoundError(f"Sin archivos con extensiones {extensions} en {data_dir}")

    # Determinar dtype de almacenamiento antes de lanzar workers
    storage_dtype = _get_storage_dtype(tokenizer, sp_model, hf_model)
    storage_dtype_str = storage_dtype.__name__ if hasattr(storage_dtype, '__name__') else storage_dtype().dtype.name
    bytes_per_tok = np.dtype(storage_dtype).itemsize

    if verbose:
        print(f"[tokenize] Encontrados {len(files):,} archivos en {data_dir}")
        print(f"[tokenize] Tokenizador: {tokenizer} | Workers: {workers or mp.cpu_count()} | dtype={storage_dtype_str} ({bytes_per_tok}B/tok)")

    t0 = time.perf_counter()
    n_workers = workers or mp.cpu_count()

    # Preparar args por archivo (incluye hf_model y dtype)
    worker_args = [(f, str(tmp_dir), tokenizer, sp_model, hf_model, storage_dtype_str) for f in files]

    # Pool
    tmp_files  = []
    total_toks = 0
    failed     = 0

    with mp.Pool(
        processes=n_workers,
        initializer=_init_tokenizer,
        initargs=(tokenizer, sp_model, hf_model),
    ) as pool:
        for i, (tmp_path, ntoks) in enumerate(
            pool.imap_unordered(_tokenize_one, worker_args, chunksize=32)
        ):
            if tmp_path is not None:
                tmp_files.append(tmp_path)
                total_toks += ntoks
            else:
                failed += 1

            if verbose and (i + 1) % max(1, len(files) // 20) == 0:
                elapsed = time.perf_counter() - t0
                spd = total_toks / elapsed if elapsed > 0 else 0
                print(f"  [{i+1:>6}/{len(files)}] {_human(total_toks)} tokens  "
                      f"{spd/1e6:.1f} M tok/s", flush=True)

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"\n[tokenize] {_human(total_toks)} tokens en {elapsed:.1f}s "
              f"({total_toks/elapsed/1e6:.1f} M tok/s) | Fallidos: {failed}")

    # ── Concatenar tmps en shards ─────────────────────────────────────────────
    shard_paths = []
    shard_sizes = []
    current_shard_idx = 0
    current_shard_toks = 0
    current_shard_file = None
    current_shard_path = None

    def _open_shard(idx):
        p = out_path / f"tokens_{idx:02d}.bin"
        return open(p, "wb"), str(p)

    current_shard_file, current_shard_path = _open_shard(current_shard_idx)

    for tmp in sorted(tmp_files):           # orden reproducible
        arr = np.fromfile(tmp, dtype=storage_dtype)
        os.remove(tmp)                       # liberar disco inmediatamente

        written = 0
        while written < len(arr):
            remaining_in_shard = shard_size - current_shard_toks
            chunk = arr[written : written + remaining_in_shard]
            chunk.tofile(current_shard_file)
            current_shard_toks += len(chunk)
            written += len(chunk)

            if current_shard_toks >= shard_size:
                current_shard_file.close()
                shard_paths.append(current_shard_path)
                shard_sizes.append(current_shard_toks)
                current_shard_idx += 1
                current_shard_toks = 0
                current_shard_file, current_shard_path = _open_shard(current_shard_idx)

    if current_shard_toks > 0:
        current_shard_file.close()
        shard_paths.append(current_shard_path)
        shard_sizes.append(current_shard_toks)
    else:
        current_shard_file.close()

    # Limpiar tmp dir
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    # ── Meta ─────────────────────────────────────────────────────────────────
    # Estimar vocab_size: escanear primer shard con el dtype correcto
    sample = np.fromfile(shard_paths[0], dtype=storage_dtype)
    vocab_seen = int(sample.max()) + 1 if len(sample) > 0 else 0

    import json
    meta = {
        "n_tokens":   total_toks,
        "vocab_size": vocab_seen,
        "dtype":      storage_dtype_str,
        "bytes_per_token": bytes_per_tok,
        "shards":     [{"path": p, "n_tokens": s} for p, s in zip(shard_paths, shard_sizes)],
        "tokenizer":  tokenizer,
        "hf_model":   hf_model,
        "total_bytes": total_toks * bytes_per_tok,
        "size_gb":    round(total_toks * bytes_per_tok / 1e9, 3),
    }
    meta_path = out_path / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    if verbose:
        print(f"\n[tokenize] Escrito en {out_dir}/")
        for p, s in zip(shard_paths, shard_sizes):
            print(f"  {Path(p).name}  {_human(s)} tokens  ({s*2/1e9:.3f} GB)")
        print(f"  meta.json → vocab_seen={vocab_seen}, total={_human(total_toks)}")

    # ── HBM pinning (opcional) ────────────────────────────────────────────────
    if pin_hbm:
        _pin_to_hbm(shard_paths, out_path, verbose, storage_dtype=storage_dtype)

    return meta


def _pin_to_hbm(shard_paths: list[str], out_path: Path, verbose: bool, storage_dtype: type = np.uint16):
    """
    Carga todos los shards en GPU como int32 (CUDA no tiene uint16/uint32 nativo)
    y los guarda como tokens_hbm.pt.

    En H200 (141 GB HBM): 2.6B tokens × 4 bytes = 10.4 GB — trivial.
    En el trainer: data = torch.load('tokens_hbm.pt', map_location='cuda')
    """
    import torch
    if not torch.cuda.is_available():
        print("[pin_hbm] Sin CUDA disponible — omitiendo pin_hbm")
        return

    tensors = []
    total_bytes = 0
    for p in shard_paths:
        arr = np.fromfile(p, dtype=storage_dtype).astype(np.int32)
        t   = torch.from_numpy(arr).cuda()
        tensors.append(t)
        total_bytes += t.nbytes
        if verbose:
            print(f"  [pin_hbm] {Path(p).name} → GPU  ({t.nbytes/1e9:.3f} GB)")

    hbm_path = out_path / "tokens_hbm.pt"
    torch.save(tensors, str(hbm_path))
    if verbose:
        print(f"  [pin_hbm] Guardado: {hbm_path}  total={total_bytes/1e9:.3f} GB en GPU")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset compatible con train_h200.py
# ─────────────────────────────────────────────────────────────────────────────

class BinaryTokenDataset:
    """
    Dataset extremadamente rápido que accede a tokens binarios uint16 o uint32.

    Modos:
      1. MMAP (modo CPU): numpy.memmap sobre .bin — zero-copy, OS page cache
      2. HBM  (modo GPU): toda la data en CUDA tensor — zero-latency

    En HBM mode, __getitem__ hace slicing de CUDA tensor → no hay DataLoader,
    el trainer usa get_batch_hbm() directamente.

    El dtype (uint16/uint32) se detecta automáticamente desde meta.json si se usa
    from_meta(), o se puede pasar explícitamente con token_dtype.
    """

    def __init__(self, shard_paths: list[str], seq_len: int, mode: str = "mmap",
                 token_dtype: str = "uint16"):
        self.seq_len    = seq_len
        self.mode       = mode
        self.token_dtype = np.dtype(token_dtype)

        if mode == "mmap":
            self.data = np.concatenate([
                np.memmap(p, dtype=self.token_dtype, mode="r")
                for p in shard_paths
            ])
            self.n_tokens = len(self.data)

        elif mode == "hbm":
            import torch
            shards = []
            for p in shard_paths:
                arr = np.fromfile(p, dtype=self.token_dtype).astype(np.int32)
                shards.append(torch.from_numpy(arr).cuda())
            self.data_gpu = torch.cat(shards, dim=0)
            self.n_tokens = self.data_gpu.shape[0]
        else:
            raise ValueError(f"mode debe ser 'mmap' o 'hbm', no {mode!r}")

        # Número de secuencias completas disponibles
        self.n_seqs = (self.n_tokens - 1) // seq_len

    def get_batch_hbm(self, batch_size: int, offset: int | None = None) -> tuple:
        """
        Retorna (input_ids, labels) directamente en CUDA.
        Offset None = aleatorio.
        Solo disponible en HBM mode.
        """
        import torch
        assert self.mode == "hbm", "get_batch_hbm() solo en mode='hbm'"
        S = self.seq_len
        N = self.n_tokens - S - 1

        if offset is None:
            starts = torch.randint(0, N, (batch_size,), device='cuda')
        else:
            starts = torch.arange(offset, offset + batch_size, device='cuda').clamp(max=N)

        # Stack sin copias: advanced indexing sobre tensor HBM
        idx = starts.unsqueeze(1) + torch.arange(S + 1, device='cuda').unsqueeze(0)
        flat = self.data_gpu[idx.view(-1)].view(batch_size, S + 1)
        return flat[:, :S].long(), flat[:, 1:].long()

    def __len__(self):
        return self.n_seqs

    @classmethod
    def from_meta(cls, meta_path: str, seq_len: int, mode: str = "mmap"):
        import json
        meta = json.loads(Path(meta_path).read_text())
        shard_paths = [s["path"] for s in meta["shards"]]
        token_dtype = meta.get("dtype", "uint16")
        return cls(shard_paths, seq_len, mode, token_dtype=token_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tokeniza un directorio de texto a binario uint16 para chimera_h200"
    )
    parser.add_argument("--data_dir",   required=True, help="Directorio con archivos de texto")
    parser.add_argument("--out_dir",    required=True, help="Directorio de salida para .bin")
    parser.add_argument("--tokenizer",  default="cl100k_base",
                        choices=["cl100k_base", "o200k_base", "sentencepiece", "huggingface"],
                        help="Tokenizador: cl100k_base, o200k_base, sentencepiece, huggingface")
    parser.add_argument("--sp_model",   default=None,
                        help="Ruta al modelo SentencePiece (.model)")
    parser.add_argument("--hf_model",   default=None,
                        help="HuggingFace model ID (ej. stanford-crfm/marin-tokenizer). Requiere --tokenizer huggingface")
    parser.add_argument("--workers",    type=int, default=None,
                        help="Procesos paralelos (default: todos los cores)")
    parser.add_argument("--shard_size", type=int, default=1_000_000_000,
                        help="Tokens máximos por shard (default: 1B)")
    parser.add_argument("--extensions", nargs="+",
                        default=[".txt", ".md", ".jsonl", ".json", ".csv"],
                        help="Extensiones de archivo a procesar")
    parser.add_argument("--pin_hbm",    action="store_true",
                        help="Cargar a GPU y guardar tokens_hbm.pt (requiere CUDA)")
    args = parser.parse_args()

    if args.tokenizer == "huggingface" and not args.hf_model:
        parser.error("--hf_model es obligatorio cuando --tokenizer=huggingface")

    meta = tokenize_dataset(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        tokenizer=args.tokenizer,
        sp_model=args.sp_model,
        hf_model=args.hf_model,
        workers=args.workers,
        extensions=args.extensions,
        shard_size=args.shard_size,
        pin_hbm=args.pin_hbm,
        verbose=True,
    )

    print("\n[tokenize] ✓ Completado.")
    print(f"  Total tokens : {meta['n_tokens']:,}")
    print(f"  Total bytes  : {meta['total_bytes']/1e9:.3f} GB")
    print(f"  Vocab visto  : {meta['vocab_size']:,}")
    print(f"  Dtype        : {meta['dtype']}  ({meta['bytes_per_token']}B/tok)")
    print(f"  Shards       : {len(meta['shards'])}")   
    print(f"\nPróximo paso:")
    vocab_arg = f"--vocab {meta['vocab_size']}" if meta['vocab_size'] != 32000 else ""
    print(f"  python train_h200_elite.py --data_dir {args.out_dir} --model 125M {vocab_arg}")
    print(f"  # O con HBM dataset (zero-latency en H200):")
    print(f"  python train_h200_elite.py --data_dir {args.out_dir} --model 125M {vocab_arg} --hbm_dataset")
if __name__ == "__main__":
    main()
