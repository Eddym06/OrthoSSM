"""
inference_test.py — Test completo de inferencia para CHIMERA entrenado
=======================================================================

Ejecutar en RunPod (H200) o en PC local (RTX 4050 6 GB):

    python3 inference_test.py --ckpt /workspace/ckpt_chimera/latest.pt
    python3 inference_test.py --ckpt ~/chimera_model.pt           # PC local

Qué hace:
  1. Detección automática de hardware (H200 / RTX 4050 / CPU)
  2. Carga el checkpoint con el dtype óptimo para el hardware
  3. Benchmark de prefill: mide latencia y tok/s para distintos contextos
  4. Benchmark de decodificación autoregresiva
  5. Mide consumo de VRAM a cada longitud de contexto
  6. Genera texto de muestra para verificar coherencia
  7. Calcula perplexity aproximada sobre una secuencia de test
  8. Resumen final comparable con resultados de literatura

Salida: imprime tabla y guarda inference_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ─── Detección de hardware ────────────────────────────────────────────────────

def _detect_hardware() -> dict:
    """Detecta HW disponible y devuelve configuración óptima."""
    hw = {
        "device":        "cpu",
        "dtype":         torch.float32,
        "dtype_str":     "fp32",
        "vram_gb":       0.0,
        "gpu_name":      "CPU",
        "flash_attn":    False,
        "bf16_ok":       False,
        "fp16_ok":       False,
    }

    if not torch.cuda.is_available():
        print("[hw] CUDA no disponible — modo CPU (inferencia lenta).")
        return hw

    props   = torch.cuda.get_device_properties(0)
    name    = props.name
    vram_gb = props.total_memory / 1e9
    cc      = props.major * 10 + props.minor   # Compute Capability × 10

    hw["device"]    = "cuda"
    hw["gpu_name"]  = name
    hw["vram_gb"]   = vram_gb

    # BF16: Ampere+ (cc >= 80), incluyendo RTX 4050 (Ada Lovelace = cc89) y H200 (cc90)
    hw["bf16_ok"]   = cc >= 80
    hw["fp16_ok"]   = cc >= 60
    hw["flash_attn"] = cc >= 80   # FlashAttention 2/3 requiere Ampere+

    if hw["bf16_ok"]:
        hw["dtype"]     = torch.bfloat16
        hw["dtype_str"] = "bf16"
    elif hw["fp16_ok"]:
        hw["dtype"]     = torch.float16
        hw["dtype_str"] = "fp16"

    print(f"[hw] GPU: {name}  |  VRAM: {vram_gb:.1f} GB  |  CC: {props.major}.{props.minor}")
    print(f"[hw] dtype óptimo: {hw['dtype_str']}  |  BF16: {hw['bf16_ok']}  |  FlashAttn: {hw['flash_attn']}")

    return hw


# ─── Carga de modelo ──────────────────────────────────────────────────────────

def load_chimera(ckpt_path: str, hw: dict) -> tuple[nn.Module, dict]:
    """
    Carga CHIMERA desde un checkpoint de train_h200_elite.py.

    El checkpoint tiene formato:
        {'model': state_dict, 'args': {vocab, model, ...}, 'step': N, ...}
    """
    # Importar chimera_lm desde el directorio del script
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))

    try:
        from chimera_lm import build_chimera_125M, build_chimera_350M, ChimeraLM
        from chimera_config import ChimeraConfig
    except ImportError as e:
        print(f"[ERROR] No se pueden importar módulos CHIMERA: {e}")
        print(f"        Asegúrate de ejecutar desde el directorio chimera_h200/")
        sys.exit(1)

    device = torch.device(hw["device"])

    print(f"\n[load] Cargando checkpoint: {ckpt_path}")
    t0 = time.perf_counter()

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",   # siempre a CPU primero para evitar OOM en GPU pequeña
        weights_only=False,
    )

    # Extraer metadata del checkpoint
    saved_args = ckpt.get("args", {})
    step      = ckpt.get("step", 0)
    loss_ema  = ckpt.get("loss_ema", None)
    vocab     = saved_args.get("vocab", None) or 128002
    model_sz  = saved_args.get("model", "125M")

    loss_str = f"{loss_ema:.4f}" if loss_ema is not None else "N/A"
    print(f"[load] step={step:,}  loss_ema={loss_str}  "
          f"model={model_sz}  vocab={vocab:,}")

    # Construir arquitectura
    if model_sz == "125M":
        model = build_chimera_125M(vocab_size=vocab)
    elif model_sz == "350M":
        model = build_chimera_350M(vocab_size=vocab)
    else:
        # tiny u otro: intentar con 125M
        model = build_chimera_125M(vocab_size=vocab)

    # Cargar pesos
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[load] ADVERTENCIA: {len(missing)} claves faltantes: {missing[:3]}...")
    if unexpected:
        print(f"[load] ADVERTENCIA: {len(unexpected)} claves inesperadas.")

    # Mover a GPU + dtype
    model = model.to(device=device, dtype=hw["dtype"])
    model.eval()

    # Desactivar graph_mode si quedó activo
    for m in model.modules():
        if hasattr(m, "graph_mode"):
            m.graph_mode = False

    n_params = sum(p.numel() for p in model.parameters())
    elapsed  = time.perf_counter() - t0
    print(f"[load] {n_params/1e6:.2f}M params  |  dtype={hw['dtype_str']}  "
          f"|  cargado en {elapsed:.2f}s")

    # VRAM después de cargar
    if device.type == "cuda":
        torch.cuda.synchronize()
        vram_model = torch.cuda.memory_allocated() / 1e9
        print(f"[load] VRAM ocupada por modelo: {vram_model:.3f} GB")

    info = {
        "step":       step,
        "loss_ema":   loss_ema,
        "vocab":      vocab,
        "model_size": model_sz,
        "n_params":   n_params,
    }
    return model, info


# ─── Benchmark de prefill ─────────────────────────────────────────────────────

@torch.no_grad()
def benchmark_prefill(
    model: nn.Module,
    hw: dict,
    context_lengths: list[int],
    n_warmup: int = 3,
    n_repeat: int = 10,
) -> list[dict]:
    """
    Mide latencia y throughput del prefill (forward pass completo).
    Prefill = procesar todo el contexto de entrada de una vez.
    """
    device = torch.device(hw["device"])
    dtype  = hw["dtype"]
    results = []

    vocab = getattr(model, "vocab_size", 128002)

    print(f"\n{'─'*70}")
    print(f"  BENCHMARK PREFILL  (n_warmup={n_warmup}, n_repeat={n_repeat})")
    print(f"  {'ctx_len':>8}  {'latencia':>10}  {'tok/s':>12}  {'VRAM':>8}  {'ppl':>8}")
    print(f"  {'─'*66}")

    for ctx_len in context_lengths:
        # Vaciar caché CUDA
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        ids = torch.randint(0, min(vocab, 128000), (1, ctx_len), device=device)

        # Warmup
        try:
            for _ in range(n_warmup):
                with torch.amp.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                    out = model(ids, labels=ids)
                if device.type == "cuda":
                    torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            print(f"  {ctx_len:>8}  {'OOM':>10}  {'—':>12}  {'—':>8}  {'—':>8}")
            results.append({"ctx_len": ctx_len, "oom": True})
            continue
        except Exception as e:
            print(f"  {ctx_len:>8}  {'ERR':>10}  {'—':>12}  {'—':>8}  {'—':>8}  ({e})")
            results.append({"ctx_len": ctx_len, "error": str(e)})
            continue

        # Medir
        times = []
        loss_val = None
        for _ in range(n_repeat):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                out = model(ids, labels=ids)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

            # Extraer loss: model retorna (None, loss, loss_dict) con labels
            if isinstance(out, tuple) and len(out) >= 2:
                loss_val = out[1].item() if hasattr(out[1], "item") else float("nan")
            elif hasattr(out, "item"):
                loss_val = out.item()
            else:
                loss_val = float("nan")

        lat_ms   = min(times) * 1000   # mejor caso (sin outliers de scheduler)
        lat_avg  = sum(times) / len(times) * 1000
        tps      = ctx_len / (lat_avg / 1000)
        ppl      = math.exp(min(loss_val, 20)) if loss_val and not math.isnan(loss_val) else float("nan")

        vram_str = "—"
        vram_gb  = 0.0
        if device.type == "cuda":
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            vram_str = f"{vram_gb:.2f}GB"

        ppl_str = f"{ppl:.1f}" if not math.isnan(ppl) else "—"
        print(f"  {ctx_len:>8,}  {lat_avg:>8.1f}ms  {tps:>12,.0f}  {vram_str:>8}  {ppl_str:>8}")

        results.append({
            "ctx_len":    ctx_len,
            "lat_ms":     lat_avg,
            "lat_min_ms": lat_ms,
            "tok_per_s":  tps,
            "vram_gb":    vram_gb,
            "ppl":        ppl,
            "loss":       loss_val,
        })

    return results


# ─── Benchmark de decodificación autoregresiva ────────────────────────────────

@torch.no_grad()
def benchmark_decode(
    model: nn.Module,
    hw: dict,
    prompt_len: int = 128,
    gen_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
) -> dict:
    """
    Mide throughput de decodificación token-a-token (el modo de inferencia real).
    Esto es el cuello de botella en modelos de lenguaje — no el prefill.
    """
    device  = torch.device(hw["device"])
    dtype   = hw["dtype"]
    vocab   = getattr(model, "vocab_size", 128002)

    print(f"\n{'─'*70}")
    print(f"  BENCHMARK DECODIFICACIÓN AUTOREGRESIVA")
    print(f"  prompt={prompt_len} tokens  →  generando {gen_tokens} tokens más")
    print(f"  temp={temperature}  top_k={top_k}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Prompt inicial
    ids = torch.randint(0, min(vocab, 128000), (1, prompt_len), device=device)

    generated = []
    times     = []
    t_total   = time.perf_counter()

    try:
        for i in range(gen_tokens):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.amp.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                # Forward con toda la secuencia actual (sin KV cache por simplicidad)
                # CHIMERA-SSM no usa KV cache tradicional — usa state SSM que se propaga
                out = model(ids)

            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

            # Muestrear siguiente token
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out

            # logits: [1, S, vocab] → tomar último paso
            next_logits = logits[0, -1, :].float()

            # Top-k sampling
            if top_k > 0:
                top_vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < top_vals[-1]] = float("-inf")

            probs     = torch.softmax(next_logits / temperature, dim=-1)
            next_tok  = torch.multinomial(probs, 1)
            generated.append(next_tok.item())

            # Append (con límite de contexto para no explotar en modelos largos)
            ids = torch.cat([ids, next_tok.unsqueeze(0)], dim=1)
            # Truncar si necesario (evitar OOM en decodificación larga)
            max_ctx = 4096
            if ids.shape[1] > max_ctx:
                ids = ids[:, -max_ctx:]

    except torch.cuda.OutOfMemoryError:
        print(f"  [OOM] Se generaron {len(generated)} tokens antes de OOM")

    n_gen    = len(generated)
    t_total  = time.perf_counter() - t_total

    if n_gen == 0:
        return {"error": "no tokens generated"}

    # El primer token es más lento (JIT, caché frío). Usar la mediana de t[2:]
    clean_times = times[2:] if len(times) > 3 else times
    lat_per_tok = sum(clean_times) / len(clean_times) * 1000  # ms
    tps         = 1000 / lat_per_tok if lat_per_tok > 0 else 0

    vram_gb = 0.0
    if device.type == "cuda":
        vram_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"  ✓ Generados {n_gen} tokens en {t_total:.2f}s")
    print(f"  Latencia por token : {lat_per_tok:.1f} ms/tok  (mediana, warmup excluido)")
    print(f"  Throughput decode  : {tps:,.0f} tok/s")
    print(f"  Throughput total   : {n_gen/t_total:,.0f} tok/s (incl. primer token)")
    if vram_gb > 0:
        print(f"  VRAM pico          : {vram_gb:.2f} GB")

    return {
        "n_generated":      n_gen,
        "total_time_s":     t_total,
        "lat_per_tok_ms":   lat_per_tok,
        "tok_per_s":        tps,
        "vram_peak_gb":     vram_gb,
        "generated_ids":    generated[:20],   # primeros 20 para debug
    }


# ─── Generación de texto legible ─────────────────────────────────────────────

@torch.no_grad()
def generate_text(
    model: nn.Module,
    hw: dict,
    tokenizer_id: str = "stanford-crfm/marin-tokenizer",
    prompt: str = "The theory of",
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
) -> str:
    """
    Genera texto legible usando el tokenizer correcto.
    Si el tokenizer no está disponible, usa IDs sintéticos.
    """
    device = torch.device(hw["device"])
    dtype  = hw["dtype"]
    vocab  = getattr(model, "vocab_size", 128002)

    print(f"\n{'─'*70}")
    print(f"  GENERACIÓN DE TEXTO DE MUESTRA")
    print(f"  Prompt: '{prompt}'")
    print(f"  max_new_tokens={max_new_tokens}  temp={temperature}  top_k={top_k}")

    # Intentar cargar tokenizer real
    tok = None
    try:
        from transformers import AutoTokenizer
        print(f"  [tok] Cargando {tokenizer_id}...")
        tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
        ids_list = tok.encode(prompt, add_special_tokens=False)
        ids = torch.tensor([ids_list], device=device)
        print(f"  [tok] '{prompt}' → {len(ids_list)} tokens")
    except Exception as e:
        print(f"  [tok] Tokenizer no disponible ({e}) — usando IDs sintéticos")
        ids = torch.randint(100, min(vocab, 1000), (1, 10), device=device)
        tok = None

    generated_ids = ids.tolist()[0].copy()

    try:
        for _ in range(max_new_tokens):
            with torch.amp.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                out = model(ids)

            logits = out[0] if isinstance(out, tuple) else out
            next_logits = logits[0, -1, :].float()

            # Top-k sampling
            if top_k > 0:
                top_vals, _ = torch.topk(next_logits, min(top_k, vocab))
                next_logits[next_logits < top_vals[-1]] = float("-inf")

            probs    = torch.softmax(next_logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            generated_ids.append(next_tok.item())

            ids = torch.cat([ids, next_tok.unsqueeze(0)], dim=1)
            if ids.shape[1] > 2048:
                ids = ids[:, -2048:]

    except torch.cuda.OutOfMemoryError:
        print("  [OOM durante generación]")

    # Decodificar si tenemos tokenizer
    if tok is not None:
        try:
            generated_text = tok.decode(generated_ids, skip_special_tokens=True)
        except Exception:
            generated_text = f"[{len(generated_ids)} token IDs generados]"
    else:
        generated_text = f"[IDs sintéticos: {generated_ids[:20]}...]"

    print(f"\n  ┌{'─'*64}┐")
    # Wrapping manual para texto largo
    words = generated_text.split()
    lines, line = [], []
    for w in words:
        if sum(len(x)+1 for x in line) + len(w) > 62:
            print(f"  │ {' '.join(line):<62} │")
            line = [w]
        else:
            line.append(w)
    if line:
        print(f"  │ {' '.join(line):<62} │")
    print(f"  └{'─'*64}┘")

    return generated_text


# ─── Resumen final ───────────────────────────────────────────────────────────

def print_summary(
    hw: dict,
    model_info: dict,
    prefill_results: list[dict],
    decode_result: dict,
):
    print(f"\n{'='*70}")
    print(f"  CHIMERA INFERENCE — RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"  Hardware : {hw['gpu_name']}  ({hw['vram_gb']:.1f} GB VRAM)")
    print(f"  dtype    : {hw['dtype_str']}")
    print(f"  Modelo   : CHIMERA-{model_info['model_size']}  "
          f"({model_info['n_params']/1e6:.2f}M params)")
    _le = model_info['loss_ema']
    loss_str2 = f"{_le:.4f}" if _le is not None else "N/A"
    ppl_str   = f"{math.exp(min(_le, 20)):.1f}" if _le is not None else "N/A"
    print(f"  Ckpt step: {model_info['step']:,}  loss_ema: {loss_str2}  PPL≈{ppl_str}")

    print(f"\n  ── Prefill (procesamiento de prompt) ──")
    ok_results = [r for r in prefill_results if "tok_per_s" in r]
    if ok_results:
        max_ctx = max(r["ctx_len"] for r in ok_results)
        best_tps = max(r["tok_per_s"] for r in ok_results)
        print(f"  Contexto máximo soportado : {max_ctx:,} tokens")
        print(f"  Throughput prefill máx    : {best_tps:,.0f} tok/s")
        # Mostrar tabla compacta
        print(f"\n  {'ctx_len':>8}  {'tok/s':>10}  {'latencia':>10}  {'VRAM':>8}  {'PPL':>8}")
        for r in ok_results:
            ppl_s = f"{r['ppl']:.1f}" if r.get("ppl") and not math.isnan(r["ppl"]) else "—"
            print(f"  {r['ctx_len']:>8,}  {r['tok_per_s']:>10,.0f}  "
                  f"{r['lat_ms']:>8.1f}ms  {r['vram_gb']:>6.2f}GB  {ppl_s:>8}")

    print(f"\n  ── Decodificación (generación token a token) ──")
    if "tok_per_s" in decode_result:
        print(f"  Throughput decode  : {decode_result['tok_per_s']:,.0f} tok/s")
        print(f"  Latencia/token     : {decode_result['lat_per_tok_ms']:.1f} ms")
        print(f"  VRAM pico          : {decode_result['vram_peak_gb']:.2f} GB")
        # ¿Tiempo real de conversación?
        tps = decode_result["tok_per_s"]
        print(f"  Palabras/minuto    : ~{tps * 0.75 * 60:.0f} wpm  "
              f"(a ~0.75 words/tok)")

    print(f"\n  ── Estadísticas de eficiencia ──")
    vram_model = hw["vram_gb"]
    params_gb  = model_info["n_params"] * 2 / 1e9  # BF16
    other_gb   = max(decode_result.get("vram_peak_gb", 0) - params_gb, 0)
    print(f"  Modelo (BF16)       : {params_gb*1000:.0f} MB")
    print(f"  KV + activaciones   : ~{other_gb*1000:.0f} MB (overhead inferencia)")
    print(f"  Total VRAM usado    : {decode_result.get('vram_peak_gb', 0):.2f} GB "
          f"/ {vram_model:.1f} GB disponibles")

    print(f"\n{'='*70}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test completo de inferencia CHIMERA (H200 y RTX 4050)"
    )
    parser.add_argument(
        "--ckpt",    required=True,
        help="Ruta al checkpoint (.pt) — ej. /workspace/ckpt_chimera/latest.pt"
    )
    parser.add_argument(
        "--ctx_lengths", nargs="+", type=int,
        default=[128, 256, 512, 1024, 2048, 4096, 8192],
        help="Longitudes de contexto a probar (default: 128..8192)"
    )
    parser.add_argument(
        "--gen_tokens", type=int, default=200,
        help="Tokens a generar en el benchmark de decodificación (default: 200)"
    )
    parser.add_argument(
        "--prompt", type=str, default="The theory of",
        help="Prompt para generación de texto de muestra"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="stanford-crfm/marin-tokenizer",
        help="HuggingFace tokenizer ID para decodificar texto"
    )
    parser.add_argument(
        "--n_repeat", type=int, default=10,
        help="Repeticiones por medición de prefill (default: 10)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Temperatura de muestreo (default: 0.7)"
    )
    parser.add_argument(
        "--top_k", type=int, default=40,
        help="Top-k sampling (default: 40, 0=greedy)"
    )
    parser.add_argument(
        "--out", type=str, default="inference_results.json",
        help="Archivo JSON de resultados (default: inference_results.json)"
    )
    parser.add_argument(
        "--fp32", action="store_true",
        help="Forzar FP32 aunque BF16 esté disponible (más lento, más preciso)"
    )
    parser.add_argument(
        "--no_generate", action="store_true",
        help="Saltar la generación de texto de muestra"
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  CHIMERA — Test de Inferencia Completo")
    print(f"  ckpt: {args.ckpt}")
    print(f"  fecha: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # ── Hardware ─────────────────────────────────────────────────────────────
    hw = _detect_hardware()
    if args.fp32:
        hw["dtype"]     = torch.float32
        hw["dtype_str"] = "fp32"
        print("[hw] FP32 forzado por --fp32")

    # ── Optimizaciones de CUDA ────────────────────────────────────────────────
    if hw["device"] == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hw["bf16_ok"]:
            torch.backends.cuda.allow_bf16_reduced_precision_reduction = True
        torch.set_float32_matmul_precision("high")

    # ── Cargar modelo ─────────────────────────────────────────────────────────
    if not Path(args.ckpt).exists():
        print(f"[ERROR] Checkpoint no encontrado: {args.ckpt}")
        sys.exit(1)

    model, model_info = load_chimera(args.ckpt, hw)

    # ── Benchmark prefill ─────────────────────────────────────────────────────
    prefill_results = benchmark_prefill(
        model,
        hw,
        context_lengths = args.ctx_lengths,
        n_warmup        = 3,
        n_repeat        = args.n_repeat,
    )

    # ── Benchmark decodificación ──────────────────────────────────────────────
    decode_result = benchmark_decode(
        model,
        hw,
        prompt_len  = 128,
        gen_tokens  = args.gen_tokens,
        temperature = args.temperature,
        top_k       = args.top_k,
    )

    # ── Generación de texto ──────────────────────────────────────────────────
    generated_text = ""
    if not args.no_generate:
        generated_text = generate_text(
            model,
            hw,
            tokenizer_id    = args.tokenizer,
            prompt          = args.prompt,
            max_new_tokens  = 150,
            temperature     = args.temperature,
            top_k           = args.top_k,
        )

    # ── Resumen ───────────────────────────────────────────────────────────────
    print_summary(hw, model_info, prefill_results, decode_result)

    # ── Guardar JSON ─────────────────────────────────────────────────────────
    results = {
        "timestamp":    __import__("datetime").datetime.now().isoformat(),
        "ckpt":         str(args.ckpt),
        "hardware": {
            "gpu":       hw["gpu_name"],
            "vram_gb":   hw["vram_gb"],
            "dtype":     hw["dtype_str"],
        },
        "model": {
            "step":      model_info["step"],
            "loss_ema":  model_info["loss_ema"],
            "ppl":       math.exp(min(model_info["loss_ema"] or 20, 20)),
            "n_params":  model_info["n_params"],
            "vocab":     model_info["vocab"],
        },
        "prefill":      prefill_results,
        "decode":       decode_result,
        "generated_text": generated_text[:500],
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[ok] Resultados guardados en: {args.out}")


if __name__ == "__main__":
    main()
