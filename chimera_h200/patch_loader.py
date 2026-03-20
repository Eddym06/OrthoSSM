import re

with open('train_h200_elite.py', 'r') as f:
    text = f.read()

# Replace _random_starts and _prefetch_to_slot with sequential pointer logic

new_loader = """    def __init__(
        self,
        shard_paths: list[str],
        seq_len: int,
        batch_size: int,
        device: torch.device,
        hbm: bool = False,
        n_prefetch: int = 2,
        token_dtype: str = "uint16",
    ):
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.device     = device
        self.hbm        = hbm
        self._step      = 0
        self._rng       = torch.Generator(device=self.device)
        self._rng.manual_seed(0)
        self._token_dtype = np.dtype(token_dtype)  # uint16 o uint32
        self._ptr       = 0

        if hbm:
            # Cargar todo en GPU — zero latency
            shards = []
            for p in shard_paths:
                arr = np.fromfile(p, dtype=self._token_dtype).astype(np.int64)
                shards.append(torch.from_numpy(arr).to(device))
            self._data_gpu  = torch.cat(shards)
            self._n_toks    = self._data_gpu.numel()
            self._stream_d  = None
        else:
            # mmap → pinned CPU → async H2D en stream_data
            arrays = [np.memmap(p, dtype=self._token_dtype, mode='r') for p in shard_paths]
            self._mmap      = np.concatenate(arrays)
            self._n_toks    = len(self._mmap)
            # El stream de datos solo existe en CUDA; en CPU el copy es síncrono
            self._stream_d  = torch.cuda.Stream() if device.type == 'cuda' else None
            use_pin = device.type == 'cuda'   # pinned memory solo útil con CUDA
            self._pinned    = [
                torch.empty(batch_size * (seq_len + 1), dtype=torch.int64).pin_memory()
                if use_pin else
                torch.empty(batch_size * (seq_len + 1), dtype=torch.int64)
                for _ in range(n_prefetch)
            ]
            self._gpu_bufs  = [
                torch.empty(batch_size, seq_len + 1, dtype=torch.int64, device=device)
                for _ in range(n_prefetch)
            ]
            self._ready     = [False] * n_prefetch
            self._slot      = 0
            self._n_prefetch = n_prefetch
            # Iniciar primer prefetch
            self._prefetch_to_slot(0)

    def _prefetch_to_slot(self, slot: int):
        \"\"\"Carga un batch secuencial directo CPU → pinned → GPU async.\"\"\"
        B = self.batch_size
        T = self.seq_len + 1
        chunk_size = B * T
        
        if self._ptr + chunk_size > self._n_toks:
            self._ptr = 0 # loop dataset
            
        # Zero-copy slicing from mmap, cast to int64, into pinned buffer
        raw = self._mmap[self._ptr : self._ptr + chunk_size].astype(np.int64)
        buf = self._pinned[slot]
        buf.copy_(torch.from_numpy(raw))
        
        self._ptr += chunk_size
        
        if self._stream_d is not None:
            with torch.cuda.stream(self._stream_d):
                self._gpu_bufs[slot].copy_(buf.view(B, T), non_blocking=True)
        else:
            self._gpu_bufs[slot].copy_(buf.view(B, T))
            
        self._ready[slot] = True"""

# Replace in file using regex
text = re.sub(r"    def __init__\(.*?self\._ready\[slot\] = True", new_loader, text, flags=re.DOTALL)

with open('train_h200_elite.py', 'w') as f:
    f.write(text)

