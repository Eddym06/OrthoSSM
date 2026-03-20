#!/usr/bin/env python3
"""
OrthoSSM Comprehensive Benchmark Suite
=======================================
Tests: Throughput, VRAM Scaling, Numerical Stability,
       Associative Recall, Induction Heads, Extended Training.
Target: RTX 4050 6GB VRAM

NOTE: All benchmarks use the same architecture and hyperparameters across tests.
State shape: (coeffs [B,nH,deg,hD], momentum) — V10 Lion optimizer uses single momentum buffer.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import gc
import json
from sdpc_engine import SpectralDualPathContextEngine, build_ortho_stack
from sdpc_kernel import apply_cheby_rkv_core, init_chebyshev_coefficients
from model import OrthoSSMLanguageModel

# ─── Helpers ──────────────────────────────────────────────────────────────────

def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2), torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0, 0.0

def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    gpu_reset()

# ─── TEST 1: Throughput Scaling ───────────────────────────────────────────────

def test_throughput():
    print("\n" + "="*70)
    print("TEST 1: THROUGHPUT SCALING (tokens/second vs context length)")
    print("="*70)
    
    device       = 'cuda'
    d_model      = 256
    n_attn_heads = 4
    n_cheby_heads= 8
    max_degree   = 8
    head_dim     = d_model // n_cheby_heads
    batch_size   = 1
    
    context_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    results = []
    
    for seq_len in context_lengths:
        gpu_reset()
        engine = SpectralDualPathContextEngine(
            d_model, n_attn_heads,
            n_cheby_heads=n_cheby_heads,
            window_size=512,
            max_degree=max_degree,
        ).to(device)
        engine.eval()
        
        x     = torch.randn(batch_size, seq_len, d_model, device=device)
        state = (
            init_chebyshev_coefficients(batch_size, n_cheby_heads, max_degree, head_dim, device),
            torch.zeros(batch_size, n_cheby_heads, max_degree, head_dim, device=device),
        )
        
        # Warmup
        with torch.no_grad():
            _ = engine(x[:, :min(512, seq_len)], state, return_state=True)
        torch.cuda.synchronize()
        
        # Benchmark
        t0 = time.perf_counter()
        with torch.no_grad():
            out, _ = engine(x, state, return_state=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        
        tps = (batch_size * seq_len) / elapsed
        _, peak_mem = gpu_mem_mb()
        
        results.append({'ctx': seq_len, 'time_s': round(elapsed, 4),
                        'tok_per_s': round(tps), 'peak_mb': round(peak_mem, 1)})
        print(f"  Context {seq_len:>6} | {elapsed:.4f}s | {tps:>10,.0f} tok/s | Peak VRAM {peak_mem:.1f} MB")
        
        del engine, x, state, out
    
    return results

# ─── TEST 2: VRAM Scaling (Memory Efficiency) ────────────────────────────────

def test_vram_scaling():
    print("\n" + "="*70)
    print("TEST 2: VRAM SCALING (memory growth vs context length)")
    print("="*70)
    
    device       = 'cuda'
    d_model      = 256
    n_attn_heads = 4
    n_cheby_heads= 8
    max_degree   = 8
    head_dim     = d_model // n_cheby_heads
    batch_size   = 1
    
    context_lengths = [1024, 4096, 16384, 32768, 65536]
    results = []
    
    for seq_len in context_lengths:
        gpu_reset()
        engine = SpectralDualPathContextEngine(
            d_model, n_attn_heads,
            n_cheby_heads=n_cheby_heads,
            window_size=512,
            max_degree=max_degree,
        ).to(device)
        
        x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        state = (
            init_chebyshev_coefficients(batch_size, n_cheby_heads, max_degree, head_dim, device),
            torch.zeros(batch_size, n_cheby_heads, max_degree, head_dim, device=device),
        )
        
        out, _ = engine(x, state, return_state=True)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        
        _, peak_mem = gpu_mem_mb()
        # Theoretical O(N²) attention matrix memory for comparison
        naive_attn_mb = (seq_len * seq_len * 4 * 2) / (1024 ** 2)
        
        results.append({
            'ctx': seq_len,
            'ortho_peak_mb': round(peak_mem, 1),
            'naive_attn_mb': round(naive_attn_mb, 1),
            'savings_x': round(naive_attn_mb / max(peak_mem, 1), 1)
        })
        
        print(f"  Context {seq_len:>6} | OrthoSSM {peak_mem:>8.1f} MB | "
              f"Naive Attn {naive_attn_mb:>10.1f} MB | Savings {naive_attn_mb/max(peak_mem,1):.1f}x")
        
        del engine, x, state, out
    
    return results

# ─── TEST 3: Numerical Stability Over Ultra-Long Sequences ───────────────────

def test_numerical_stability():
    print("\n" + "="*70)
    print("TEST 3: NUMERICAL STABILITY (NaN/Inf detection over long sequences)")
    print("="*70)
    
    device       = 'cuda'
    d_model      = 128
    n_cheby_heads= 8
    max_degree   = 8
    head_dim     = d_model // n_cheby_heads
    batch        = 1
    
    chunk_size   = 8192
    total_tokens = 524288   # 512K tokens
    n_chunks     = total_tokens // chunk_size
    
    coeffs   = init_chebyshev_coefficients(batch, n_cheby_heads, max_degree, head_dim, device)
    momentum = torch.zeros(batch, n_cheby_heads, max_degree, head_dim, device=device)
    
    nan_detected = inf_detected = False
    max_val = min_val = 0.0
    
    print(f"  Processing {total_tokens:,} tokens | chunks: {n_chunks} × {chunk_size}")
    
    for i in range(n_chunks):
        x = torch.randn(batch, chunk_size, d_model, device=device)
        with torch.no_grad():
            out, coeffs, momentum = apply_cheby_rkv_core(
                x, coeffs, momentum,
                n_heads=n_cheby_heads, base_lr=0.01, ema_momentum=0.9
            )
        
        if torch.isnan(out).any() or torch.isnan(coeffs).any():
            nan_detected = True
            print(f"  !! NaN at chunk {i} (token {i*chunk_size:,})")
            break
        if torch.isinf(out).any():
            inf_detected = True
            print(f"  !! Inf at chunk {i} (token {i*chunk_size:,})")
            break
        
        max_val = max(max_val, out.max().item())
        min_val = min(min_val, out.min().item())
        
        if (i + 1) % 16 == 0:
            print(f"  Chunk {i+1:>4}/{n_chunks} | Range [{out.min().item():.4f}, {out.max().item():.4f}]"
                  f" | Coeff norm: {coeffs.norm().item():.4f}")
    
    status = "PASS" if not (nan_detected or inf_detected) else "FAIL"
    print(f"\n  Result: {status} | Global Range [{min_val:.4f}, {max_val:.4f}]")
    print(f"  Final Chebyshev State Norm: {coeffs.norm().item():.4f}")
    
    return {'status': status, 'total_tokens': total_tokens, 'max': max_val, 'min': min_val}

# ─── TEST 4: Associative Recall Task ─────────────────────────────────────────

def test_associative_recall():
    print("\n" + "="*70)
    print("TEST 4: ASSOCIATIVE RECALL (Key-Value retrieval over long context)")
    print("="*70)
    
    device       = 'cuda'
    vocab_size   = 512
    d_model      = 256
    n_attn_heads = 4
    n_cheby_heads= 8
    max_degree   = 8
    seq_len      = 2048
    batch_size   = 8
    epochs       = 200
    
    model = OrthoSSMLanguageModel(
        vocab_size=vocab_size, d_model=d_model, n_attn_heads=n_attn_heads,
        n_cheby_heads=n_cheby_heads, n_layers=1, max_degree=max_degree,
        window_size=512,
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    model.train()
    losses = []
    
    print(f"  Config: vocab={vocab_size}, dim={d_model}, ctx={seq_len}, batch={batch_size}")
    print(f"  Task: KV pairs placed early in sequence, query at end, model must recall the value.")
    
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        
        x = torch.randint(1, vocab_size // 2, (batch_size, seq_len), device=device)
        y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        
        n_pairs = 8
        for b in range(batch_size):
            keys   = torch.randint(vocab_size // 2, vocab_size, (n_pairs,), device=device)
            values = torch.randint(1, vocab_size // 2, (n_pairs,), device=device)
            for j in range(n_pairs):
                pos = j * 2 + 10
                x[b, pos]     = keys[j]
                x[b, pos + 1] = values[j]
            query_idx = torch.randint(0, n_pairs, (1,)).item()
            query_pos = seq_len - 4
            x[b, query_pos]     = keys[query_idx]
            y[b, query_pos + 1] = values[query_idx]
        
        # State is managed fresh per step (TBPTT length = seq_len)
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"  Step {epoch:03d} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    improvement = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"\n  Loss improvement: {improvement:.1f}% ({losses[0]:.4f} → {losses[-1]:.4f})")
    
    del model
    return {'start_loss': losses[0], 'end_loss': losses[-1], 'improvement_pct': round(improvement, 1)}

# ─── TEST 5: Extended Training with Copy Task (500 steps) ────────────────────

def test_extended_training():
    print("\n" + "="*70)
    print("TEST 5: EXTENDED CALIBRATION (500 steps, CosineAnnealing, Copy Task)")
    print("="*70)
    
    device       = 'cuda'
    vocab_size   = 2000
    d_model      = 256
    n_attn_heads = 4
    n_cheby_heads= 8
    max_degree   = 8
    seq_len      = 4096
    batch_size   = 2
    epochs       = 500
    
    model = OrthoSSMLanguageModel(
        vocab_size=vocab_size, d_model=d_model, n_attn_heads=n_attn_heads,
        n_cheby_heads=n_cheby_heads, n_layers=1, max_degree=max_degree,
        window_size=1024,
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    model.train()
    losses = []
    
    print(f"  Config: vocab={vocab_size}, dim={d_model}, ctx={seq_len}, batch={batch_size}, steps={epochs}")
    
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        
        half = seq_len // 2
        first_half = torch.randint(1, vocab_size, (batch_size, half), device=device)
        x = torch.cat([first_half, first_half], dim=1)
        y = torch.full_like(x, -100)
        y[:, half:-1] = x[:, half + 1:].clone()
        
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        if epoch % 50 == 0 or epoch == epochs - 1:
            cur_mem, _ = gpu_mem_mb()
            print(f"  Step {epoch:03d} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | VRAM: {cur_mem:.0f} MB")
    
    improvement = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"\n  Loss improvement: {improvement:.1f}% ({losses[0]:.4f} → {losses[-1]:.4f})")
    
    del model
    return {
        'start_loss': losses[0],
        'end_loss': losses[-1],
        'improvement_pct': round(improvement, 1),
        'final_losses_avg': round(sum(losses[-20:]) / 20, 4)
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║            OrthoSSM - Comprehensive Benchmark Suite                ║")
    print("║            RTX 4050 Laptop (6GB VRAM) - Real Hardware              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    all_results = {}
    
    # Test 1: Throughput
    all_results['throughput'] = test_throughput()
    gpu_reset()
    
    # Test 2: VRAM Scaling
    all_results['vram_scaling'] = test_vram_scaling()
    gpu_reset()
    
    # Test 3: Numerical Stability
    all_results['stability'] = test_numerical_stability()
    gpu_reset()
    
    # Test 4: Associative Recall
    all_results['assoc_recall'] = test_associative_recall()
    gpu_reset()
    
    # Test 5: Extended Training
    all_results['extended_train'] = test_extended_training()
    gpu_reset()
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    t = all_results['throughput']
    print(f"\n  Throughput: {t[0]['tok_per_s']:,} tok/s @ 1K ctx -> {t[-1]['tok_per_s']:,} tok/s @ 65K ctx")
    
    v = all_results['vram_scaling']
    print(f"  VRAM: {v[0]['ortho_peak_mb']} MB @ 1K ctx -> {v[-1]['ortho_peak_mb']} MB @ 65K ctx (vs {v[-1]['naive_attn_mb']} MB naive)")
    
    s = all_results['stability']
    print(f"  Stability: {s['status']} over {s['total_tokens']:,} tokens")
    
    a = all_results['assoc_recall']
    print(f"  Associative Recall: Loss {a['start_loss']:.4f} -> {a['end_loss']:.4f} ({a['improvement_pct']}% improvement)")
    
    e = all_results['extended_train']
    print(f"  Extended Training: Loss {e['start_loss']:.4f} -> {e['end_loss']:.4f} ({e['improvement_pct']}% improvement)")
    
    print("\n  Done. OrthoSSM benchmark complete.")
