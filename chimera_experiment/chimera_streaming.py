"""
chimera_streaming.py — D3: Streaming Inference Mode + Chunked Prefill con carry

Implementa:
  StreamingDecoder:   buffer N tokens → forward chunk → TTT update cada N pasos.
                      Más rápido que step() puro para generación larga porque:
                      (a) el kernel Mamba2 hace scan paralelo sobre N tokens
                      (b) TTT-Lite se actualiza cada N tokens con información real
                      (c) SLR/archive operan sobre N tokens → mejor calidad retrieval

  chunked_prefill:    Procesa contexto largo en trozos de chunk_size tokens,
                      pasando bus_cache entre chunks (D5 carry).

  StreamingLMDecoder: Wrapper completo: embedding → streaming → lm_head → tokens.

Uso rápido:
    from chimera_streaming import StreamingDecoder, chunked_prefill
    decoder = StreamingDecoder(chimera_layer, chunk_size=16)
    for tok in decoder.generate(context_hidden, lm_head, max_new=256):
        print(tok)
"""
from __future__ import annotations
import math, time, copy
from typing import Optional, List, Callable, Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clone_cache(cache):
    """Copia profunda de cualquier cache (dict/tensor/None)."""
    if cache is None:
        return None
    if isinstance(cache, torch.Tensor):
        return cache.clone()
    if isinstance(cache, dict):
        return {k: _clone_cache(v) for k, v in cache.items()}
    if isinstance(cache, (list, tuple)):
        cloned = [_clone_cache(v) for v in cache]
        return type(cache)(cloned)
    return cache


def _cat_bus(bus_a, bus_b):
    """
    Combina dos bus_cache (tensor o dict {'bus_cache': tensor}).
    Si ambos son None devuelve None.
    Si uno es None devuelve el otro.
    """
    def _unwrap(b):
        if b is None: return None
        if isinstance(b, dict): return b.get('bus_cache')
        return b

    ta = _unwrap(bus_a); tb = _unwrap(bus_b)
    if ta is None: return bus_b
    if tb is None: return bus_a
    # ta/tb: [B, L, d] — tomamos el bus más reciente (bus_b wins)
    return bus_b


# ─────────────────────────────────────────────────────────────────────────────
# 1. Chunked Prefill con carry inter-chunk
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def chunked_prefill(
    model: nn.Module,
    x: torch.Tensor,               # [B, S, D] — secuencia completa de prefill
    chunk_size: int = 4096,
    bus_cache=None,                 # cache inicial (None para primer contexto)
    return_all_outputs: bool = False,
) -> tuple:
    """
    Procesa x en trozos de chunk_size, pasando bus_cache entre chunks.
    Esto resuelve el drift +107% detectado en T5 al integrar el estado
    del contexto anterior en cada chunk sucesivo.

    Args:
        model:           AdvancedChimeraLayer (o stack compatible con forward(x, bus_cache))
        x:               [B, S, D] — secuencia a procesar
        chunk_size:      tamaño de cada chunk (default 4096 — manejable en 6GB)
        bus_cache:       estado inicial del bus (de un prefill anterior, o None)
        return_all_outputs: si True, devuelve tensor [B, S, D] completo concatenado

    Returns:
        (last_hidden [B, chunk_size, D], final_bus_cache)   — si return_all_outputs=False
        (all_hidden  [B, S,          D], final_bus_cache)   — si return_all_outputs=True
    """
    B, S, D = x.shape
    model.eval()

    outputs = [] if return_all_outputs else None
    current_bus = bus_cache
    last_out = None

    n_chunks = math.ceil(S / chunk_size)
    for i in range(n_chunks):
        x_chunk = x[:, i * chunk_size : (i + 1) * chunk_size, :]

        out_chunk, new_bus = model(x_chunk, bus_cache=current_bus)

        # carry: el bus del chunk i se pasa al chunk i+1
        current_bus = new_bus
        last_out    = out_chunk

        if return_all_outputs:
            outputs.append(out_chunk)

        # garbage collect para liberar activaciones intermedias
        if i % 4 == 3:
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    if return_all_outputs:
        return torch.cat(outputs, dim=1), current_bus
    return last_out, current_bus


# ─────────────────────────────────────────────────────────────────────────────
# 2. StreamingDecoder — D3 completo
# ─────────────────────────────────────────────────────────────────────────────

class StreamingDecoder:
    """
    D3: Streaming Inference Mode.

    En lugar de llamar step() token a token (simple pero sin TTT update),
    el StreamingDecoder acumula `chunk_size` tokens en un buffer y luego
    ejecuta un forward() paralelo sobre el chunk completo.

    Ventajas:
      - Mamba2 hace el scan paralelo sobre chunk_size tokens → mejor GPU utilization
      - TTT-Lite se ejecuta sobre chunk_size tokens → optimización real del dt_bias
      - SLR/archive operan sobre chunk_size tokens → mejor retrieval
      - Latencia por token ≈ latency(full_chunk) / chunk_size → mucho mejor throughput

    Uso:
        dec = StreamingDecoder(chimera_layer, chunk_size=16)
        initial_cache = dec.prefill(context_hidden, bus_cache=None)
        for token_id in dec.generate(lm_head, embed, max_new=512, cache=initial_cache):
            print(token_id)

    Nota: para generación autoregresiva real necesitas embedding + lm_head externos.
    El StreamingDecoder opera en el espacio oculto [B, 1, D].
    """

    def __init__(
        self,
        model: nn.Module,          # AdvancedChimeraLayer
        chunk_size: int = 16,      # tokens por TTT update
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ):
        self.model       = model
        self.chunk_size  = chunk_size
        self.temperature = temperature
        self.top_k       = top_k
        self.top_p       = top_p

        # Buffer interno de tokens hidden
        self._buffer_hidden: Optional[torch.Tensor] = None  # [B, n_buffered, D]
        self._bus_cache = None

    # ── prefill ───────────────────────────────────────────────────────────────

    def prefill(self, x: torch.Tensor, bus_cache=None, chunk_size: int = 4096) -> dict:
        """
        Procesa el contexto (prefill) con chunked_prefill para evitar OOM
        y asegurar bus carry correcto entre chunks.

        Args:
            x:          [B, S, D] — contexto completo
            bus_cache:  cache heredado de paso anterior (None si es el primero)
            chunk_size: tamaño de chunk para el prefill (default 4096)

        Returns:
            cache dict con {'bus_cache': ..., 'last_hidden': last_out}
        """
        last_out, final_bus = chunked_prefill(
            self.model, x, chunk_size=chunk_size, bus_cache=bus_cache
        )
        self._bus_cache = final_bus
        self._buffer_hidden = None  # limpiar buffer de tokens pendientes
        return {'bus_cache': final_bus, 'last_hidden': last_out}

    # ── step interno (acumula en buffer, ejecuta forward cada N tokens) ────────

    def step(
        self,
        x_single: torch.Tensor,    # [B, D] — hidden de UN token
    ) -> torch.Tensor:
        """
        Procesa un token acumulándolo en el buffer.
        Cuando el buffer llega a chunk_size, ejecuta un forward() real.
        Devuelve el hidden del token actual [B, D].
        """
        B, D = x_single.shape
        x_tok = x_single.unsqueeze(1)   # [B, 1, D]

        if self._buffer_hidden is None:
            self._buffer_hidden = x_tok
        else:
            self._buffer_hidden = torch.cat([self._buffer_hidden, x_tok], dim=1)

        n_buf = self._buffer_hidden.shape[1]

        if n_buf >= self.chunk_size:
            # Flush: forward() sobre el chunk completo (TTT update real)
            chunk = self._buffer_hidden                       # [B, chunk_size, D]
            with torch.no_grad():
                out_chunk, new_bus = self.model(chunk, bus_cache=self._bus_cache)
            self._bus_cache     = new_bus
            self._buffer_hidden = None
            return out_chunk[:, -1, :]                        # último token del chunk

        else:
            # Buffer aún incompleto — forward individual (sin TTT update)
            # Siempre no_grad: step() es siempre inferencia.
            with torch.no_grad():
                out, new_bus = self.model(x_tok, bus_cache=self._bus_cache)
            self._bus_cache = new_bus
            return out[:, 0, :]                               # [B, D]

    def flush(self) -> Optional[torch.Tensor]:
        """
        Fuerza el procesamiento de los tokens pendientes en el buffer,
        aunque el buffer no esté lleno.  Devuelve el hidden del último token.
        """
        if self._buffer_hidden is None:
            return None
        chunk = self._buffer_hidden
        with torch.no_grad():
            out_chunk, new_bus = self.model(chunk, bus_cache=self._bus_cache)
        self._bus_cache     = new_bus
        self._buffer_hidden = None
        return out_chunk[:, -1, :]

    # ── generate loop completo ─────────────────────────────────────────────────

    def generate(
        self,
        lm_head: nn.Module,        # Linear(D, vocab_size)
        embed_fn: Callable,         # vocab_id → [B, D]
        context_hidden: torch.Tensor,   # [B, S, D] de prefill
        max_new_tokens: int = 256,
        initial_cache: Optional[dict] = None,
    ) -> Iterator[torch.Tensor]:
        """
        Generación autoregresiva con streaming.
        Yielda tensor de token_ids [B] en cada paso.

        Ejemplo:
            for tok_ids in decoder.generate(lm_head, embed_fn, ctx_hidden, 256):
                next_word = tokenizer.decode(tok_ids[0].item())
        """
        device = context_hidden.device

        # Prefill
        if initial_cache is not None:
            self._bus_cache     = initial_cache.get('bus_cache')
            self._buffer_hidden = None
        else:
            _ = self.prefill(context_hidden)

        # Primer token: proyectamos el último hidden del contexto
        last_hidden = context_hidden[:, -1, :]   # [B, D]

        t0 = time.perf_counter()
        n_generated = 0

        for _ in range(max_new_tokens):
            # Logits desde el último hidden
            logits = lm_head(last_hidden)            # [B, vocab]

            # Sampling
            if self.temperature != 1.0:
                logits = logits / self.temperature
            if self.top_k > 0:
                v, _ = torch.topk(logits, self.top_k, dim=-1)
                logits[logits < v[:, -1:]] = -float('inf')
            if self.top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                cum_probs = sorted_logits.softmax(-1).cumsum(-1)
                sorted_logits[cum_probs > self.top_p] = -float('inf')
                logits.scatter_(-1, sorted_idx, sorted_logits)

            probs    = logits.softmax(-1)
            tok_ids  = torch.multinomial(probs, 1).squeeze(-1)   # [B]
            yield tok_ids

            # Siguiente hidden
            x_emb       = embed_fn(tok_ids)             # [B, D]
            last_hidden = self.step(x_emb)               # [B, D] (streaming)
            n_generated += 1

        # Flush tokens pendientes en el buffer
        self.flush()

        elapsed = time.perf_counter() - t0
        tps     = n_generated / elapsed
        print(f"  [StreamingDecoder] {n_generated} tokens en {elapsed:.2f}s = {tps:.1f} tok/s")

    def reset(self):
        """Limpia el estado del decoder (bus_cache + buffer) para nueva secuencia."""
        self._bus_cache     = None
        self._buffer_hidden = None


# ─────────────────────────────────────────────────────────────────────────────
# Null context manager (compatibilidad torch.no_grad vs train)
# ─────────────────────────────────────────────────────────────────────────────
import contextlib
_null_ctx = contextlib.nullcontext


# ─────────────────────────────────────────────────────────────────────────────
# 3. StreamingLMDecoder — end-to-end (embedding + chimera + lm_head)
# ─────────────────────────────────────────────────────────────────────────────

class StreamingLMDecoder:
    """
    Wrapper completo embedding → CHIMERA streaming → lm_head → token_ids.
    Compatible con niah_eval.ChimeraLM.

    Uso:
        from chimera_streaming import StreamingLMDecoder
        slm = StreamingLMDecoder(chimera_lm, chunk_size=16)
        tokens = slm.generate_text(prompt_ids, max_new=128)
    """

    def __init__(self, lm: nn.Module, chunk_size: int = 16, **gen_kwargs):
        """
        lm: debe tener .embed, .chimera (o .stack), .lm_head, .d_model
        """
        self.lm         = lm
        self.chunk_size = chunk_size
        self.gen_kwargs = gen_kwargs

        # Extrae submódulos del LM
        self._embed   = getattr(lm, 'embed', None) or getattr(lm, 'embedding', None)
        self._chimera = getattr(lm, 'chimera', None) or getattr(lm, 'stack', None)
        self._lm_head = getattr(lm, 'lm_head', None) or getattr(lm, 'head', None)

        if self._chimera is None:
            raise ValueError("El LM debe tener atributo 'chimera' o 'stack' con AdvancedChimeraLayer")

        self._decoder = StreamingDecoder(
            self._chimera, chunk_size=chunk_size, **gen_kwargs
        )

    def encode_context(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids [B, S] → hidden [B, S, D]"""
        if self._embed is None:
            raise ValueError("LM sin embedding encontrado")
        with torch.no_grad():
            return self._embed(token_ids).float()

    def _embed_fn(self, tok_ids: torch.Tensor) -> torch.Tensor:
        """tok_ids [B] → hidden [B, D]"""
        emb = self._embed(tok_ids.unsqueeze(1)).float()   # [B, 1, D]
        return emb.squeeze(1)                              # [B, D]

    def generate_text(
        self,
        prompt_ids: torch.Tensor,      # [B, S] — token IDs del contexto
        max_new: int = 128,
    ) -> torch.Tensor:
        """
        Genera `max_new` tokens dado un prompt en formato token_ids.
        Devuelve tensor [B, max_new] con los IDs generados.
        """
        ctx_hidden = self.encode_context(prompt_ids)   # [B, S, D]
        self._decoder.reset()

        generated = []
        for tok_ids in self._decoder.generate(
            lm_head=self._lm_head,
            embed_fn=self._embed_fn,
            context_hidden=ctx_hidden,
            max_new_tokens=max_new,
        ):
            generated.append(tok_ids)

        return torch.stack(generated, dim=1)  # [B, max_new]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Benchmark streaming vs step()
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_streaming(
    n_new: int = 128,
    chunk_sizes: List[int] = [1, 4, 16, 32, 64],
):
    """
    Compara throughput de diferentes chunk_size en streaming vs step() puro.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from advanced_chimera import AdvancedChimeraLayer

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print("  [benchmark_streaming] requiere CUDA"); return

    d = 256
    model = AdvancedChimeraLayer(d_model=d).to(device).float()
    model.eval()   # imprescindible: evita autograd en step()

    # LM head ficticio
    lm_head = nn.Linear(d, 512, bias=False).to(device).float()
    embed   = nn.Embedding(512, d).to(device).float()

    # Contexto inicial pequeño
    ctx_ids  = torch.randint(0, 512, (1, 64), device=device)
    ctx_hidden = embed(ctx_ids).float()   # [1, 64, D]

    # Warmup: compilar kernels CUDA antes de medir
    print("  Warmup (compilando kernels CUDA)...")
    _wdec = StreamingDecoder(model, chunk_size=16)
    _ = _wdec.prefill(ctx_hidden)
    for _ in range(32):
        _wdec.step(torch.randn(1, d, device=device))
    torch.cuda.synchronize()
    del _wdec

    def embed_fn(ids): return embed(ids.unsqueeze(1)).squeeze(1).float()

    print(f"\n  Benchmark Streaming: {n_new} tokens nuevos, d={d}")
    print(f"  {'chunk':>8}  {'ms/tok':>9}  {'tok/s':>9}  {'speedup':>9}")
    print(f"  {'-'*44}")

    results = []
    ref_tps = None

    for cs in chunk_sizes:
        decoder = StreamingDecoder(model, chunk_size=cs)
        _ = decoder.prefill(ctx_hidden)

        t0 = time.perf_counter()
        count = 0
        for tok_ids in decoder.generate(lm_head, embed_fn, ctx_hidden, max_new_tokens=n_new):
            count += 1
        elapsed = time.perf_counter() - t0
        tps = count / elapsed
        ms_tok = elapsed / count * 1e3

        if cs == 1:
            ref_tps = tps   # step() puro
            speedup_str = "1.00x (base)"
        else:
            speedup_str = f"{tps/ref_tps:.2f}x"

        print(f"  chunk={cs:>4}  {ms_tok:>8.2f}  {tps:>9,.0f}  {speedup_str:>9}")
        results.append({'chunk_size': cs, 'tps': tps, 'ms_per_tok': ms_tok})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[chimera_streaming] self-test  device={device}")
    if device != 'cuda':
        print("  Requiere CUDA para AdvancedChimeraLayer. Saliendo.")
        sys.exit(0)

    from advanced_chimera import AdvancedChimeraLayer
    d = 256
    model = AdvancedChimeraLayer(d_model=d).cuda().float()
    model.eval()  # imprescindible: sin esto step() construye grafo autograd

    # Warmup global (compilar kernels CUDA una vez)
    print("  Warmup CUDA...")
    _x_warm = torch.randn(1, 64, d, device='cuda')
    with torch.no_grad():
        _, _ = model(_x_warm)
    torch.cuda.synchronize()
    del _x_warm

    # Test 1: chunked_prefill con carry
    print("\n[1] chunked_prefill carry test")
    x_long = torch.randn(1, 2048, d, device='cuda')
    out_chunked, bus = chunked_prefill(model, x_long, chunk_size=512, return_all_outputs=True)
    assert out_chunked.shape == (1, 2048, d), f"Shape incorrecto: {out_chunked.shape}"
    assert not out_chunked.isnan().any(), "NaN en salida chunked"
    print(f"  chunked_prefill OK  shape={tuple(out_chunked.shape)}")

    # Test 2: StreamingDecoder prefill + step
    print("\n[2] StreamingDecoder step test")
    dec = StreamingDecoder(model, chunk_size=16)
    _ = dec.prefill(torch.randn(1, 256, d, device='cuda'))
    # Warmup step
    for _ in range(8):
        dec.step(torch.randn(1, d, device='cuda'))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(64):
        x_tok = torch.randn(1, d, device='cuda')
        h = dec.step(x_tok)
        assert h.shape == (1, d), f"Step shape {h.shape}"
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tps = 64 / elapsed
    print(f"  64 steps  shape={tuple(h.shape)}  {elapsed*1e3:.1f} ms  {tps:,.0f} tok/s")

    # Test 3: benchmark
    print("\n[3] Benchmark streaming vs step()")
    results = benchmark_streaming(n_new=64, chunk_sizes=[1, 8, 16, 32])
    if results:
        best = max(results, key=lambda r: r['tps'])
        print(f"\n  Mejor chunk_size: {best['chunk_size']} ({best['tps']:,.0f} tok/s)")

    print("\n[ALL OK] chimera_streaming.py ready")
