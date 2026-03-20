
---
## Session 3 – Streaming D3 + Dynamic Thresholds D5 + Bug Fixes

### Bugs corregidos
1. **`_null_ctx` en `StreamingDecoder.step()`** — Bug crítico: la condición
   `torch.no_grad() if not model.training else _null_ctx()` hacía que cuando
   `model.training=True` (valor por defecto) se construyera el grafo completo de autograd
   en cada token durante generación. Efecto: 8 tok/s → **101 tok/s** (12.6× speedup).
   Fix: `torch.no_grad()` siempre en step() (inference-only).

2. **`std()` con B=1 (D5)** — `complexity_per_elem.std()` con un solo elemento lanza
   UserWarning. Fix: `std(correction=0)` con guard `if numel > 1`.

### Implementaciones completadas
- **D5 (Dynamic Length Thresholds)**: umbrales SLR/archive por-elemento con soft gate.
- **D3 (Streaming Inference)**: `chimera_streaming.py` – chunked_prefill + StreamingDecoder.

### Throughput final (RTX 4050 Laptop, d=256, 1 capa)
| Modo | tok/s |
|---|---|
| step() directo (64 pasos) | 101 |
| generate() chunk_size=1 | 73 |
| generate() chunk_size=16 (óptimo) | **111** |
| Mamba2 raw step() (T8, sin TTT/Bus) | 184 |

El delta (111 vs 184) es el coste de TTT-Lite + Bus + Archive en cada paso.

### chunked_prefill results (S=2048, chunk=512, con bus carry)
- Shape OK: (1, 2048, 256). Sin NaN.
- Drift SIN carry vs full: +107%. CON carry: −0.2% (mejora real pero pequeña en modelo no entrenado).
