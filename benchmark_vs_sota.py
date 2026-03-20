#!/usr/bin/env python3
"""
OrthoSSM V11 — Benchmark vs SOTA Baselines (B5-B8)
====================================================
Head-to-head comparison against Mamba, RWKV, Transformer, and GRU baselines.
All models implemented in pure PyTorch for fair comparison (same backend).

Tests:
  B5: Model instantiation + parameter count verification
  B6: Needle-In-A-Haystack (NIAH) + Associative Recall
  B7: Throughput (tok/s) + VRAM scaling vs context length
  B8: Short training convergence on synthetic language data

Hardware target: RTX 4050 6GB (SM89)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import gc
import math
import sys
import json
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE IMPLEMENTATIONS (Pure PyTorch — fair comparison)
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1) Mamba-style Selective SSM ──────────────────────────────────────────────

class MambaBlock(nn.Module):
    """
    Mamba-style Selective State Space block (Gu & Dao, 2023).
    Pure PyTorch implementation of the selective scan algorithm.
    S6 architecture: input-dependent A, B, C, Δ parameters.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv

        # Input projection: x → (z, x_proj) where z is the gate
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # 1D causal convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True
        )

        # SSM parameters projection: x → (Δ, B, C)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # dt_rank=1

        # dt projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A parameter (diagonal, learnable log-space)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A.repeat(self.d_inner, 1)))  # [d_inner, d_state]

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Norm
        self.norm = nn.RMSNorm(d_model)

    def _selective_scan(self, x, dt, B, C, D):
        """
        Selective scan (sequential for correctness).
        x: [B, S, d_inner]
        dt: [B, S, d_inner]
        B: [B, S, d_state]
        C: [B, S, d_state]
        """
        batch, seq_len, d_inner = x.shape
        d_state = B.shape[-1]

        # Discretize: A_bar = exp(A * dt), B_bar = dt * B
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        dt_A = torch.einsum('bsd,dn->bsdn', dt, A)  # [B, S, d_inner, d_state]
        A_bar = torch.exp(dt_A)  # [B, S, d_inner, d_state]
        B_bar = torch.einsum('bsd,bsn->bsdn', dt * x, B)  # [B, S, d_inner, d_state]

        # Sequential scan
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = A_bar[:, t] * h + B_bar[:, t]  # [B, d_inner, d_state]
            y_t = torch.einsum('bdn,bn->bd', h, C[:, t])  # [B, d_inner]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # [B, S, d_inner]
        y = y + x * D.unsqueeze(0).unsqueeze(0)
        return y

    def forward(self, x):
        residual = x
        x = self.norm(x)

        # Input projection
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)  # each [B, S, d_inner]

        # Causal conv1d
        x_conv = x_inner.transpose(1, 2)  # [B, d_inner, S]
        x_conv = self.conv1d(x_conv)[:, :, :x.shape[1]]  # trim to causal
        x_conv = x_conv.transpose(1, 2)  # [B, S, d_inner]
        x_conv = F.silu(x_conv)

        # SSM parameters
        ssm_params = self.x_proj(x_conv)  # [B, S, 2*d_state+1]
        dt_raw = ssm_params[..., :1]  # [B, S, 1]
        B = ssm_params[..., 1:1 + self.d_state]
        C = ssm_params[..., 1 + self.d_state:]

        dt = F.softplus(self.dt_proj(dt_raw))  # [B, S, d_inner]

        # Selective scan
        y = self._selective_scan(x_conv, dt, B, C, self.D)

        # Gate and project
        y = y * F.silu(z)
        out = self.out_proj(y)

        return residual + out


class MambaLM(nn.Module):
    """Mamba language model with N layers."""
    def __init__(self, vocab_size, d_model, n_layers, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # tie weights

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)


# ── 2) RWKV-style Time-Mixing + Channel-Mixing ───────────────────────────────

class RWKVTimeMixing(nn.Module):
    """
    RWKV-4 time mixing block.
    Uses the WKV (weighted key-value) mechanism with exponential decay.
    """
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Time-mix parameters (learnable interpolation between current and previous)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)

        # Projections
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

        # Per-head decay (learned in log-space)
        self.time_decay = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.1 - 5.0)
        # Per-head first bonus
        self.time_first = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.1)

        self.ln = nn.GroupNorm(n_heads, d_model)

    def forward(self, x):
        B, S, D = x.shape
        H, hD = self.n_heads, self.head_dim

        # Time-shift: shifted = [zeros, x[:, :-1, :]]
        shifted = F.pad(x[:, :-1, :], (0, 0, 1, 0))

        # Mix current and previous
        xr = x * self.time_mix_r + shifted * (1 - self.time_mix_r)
        xk = x * self.time_mix_k + shifted * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + shifted * (1 - self.time_mix_v)

        r = self.receptance(xr).view(B, S, H, hD)
        k = self.key(xk).view(B, S, H, hD)
        v = self.value(xv).view(B, S, H, hD)

        # WKV computation (sequential scan)
        w = -torch.exp(self.time_decay)  # [H, hD]
        u = self.time_first  # [H, hD]

        # Sequential WKV
        wkv = torch.zeros(B, H, hD, device=x.device, dtype=x.dtype)
        wkv_state_a = torch.zeros(B, H, hD, device=x.device, dtype=x.dtype)
        wkv_state_b = torch.zeros(B, H, hD, device=x.device, dtype=x.dtype) - 1e38

        outs = []
        for t in range(S):
            kt = k[:, t]  # [B, H, hD]
            vt = v[:, t]

            # wkv = (sum_i exp(-(t-i)*w + k_i) * v_i) / (sum_i exp(-(t-i)*w + k_i))
            # Simplified linear attention with decay
            wk = kt + u  # [B, H, hD]
            # Numerically stable: max(wkv_state_b, wk)
            p = torch.maximum(wkv_state_b, wk)
            e1 = torch.exp(wkv_state_b - p)
            e2 = torch.exp(wk - p)
            out_t = (e1 * wkv_state_a + e2 * vt) / (e1 + e2 + 1e-8)
            outs.append(out_t)

            # Update state
            ww = wkv_state_b + w
            p2 = torch.maximum(ww, kt)
            e1 = torch.exp(ww - p2)
            e2 = torch.exp(kt - p2)
            wkv_state_a = e1 * wkv_state_a + e2 * vt
            wkv_state_b = p2 + torch.log(e1 + e2 + 1e-8)

        wkv_out = torch.stack(outs, dim=1)  # [B, S, H, hD]

        # Gate with receptance
        rwkv = torch.sigmoid(r) * wkv_out
        rwkv = rwkv.reshape(B, S, D)
        rwkv = self.ln(rwkv.transpose(1, 2)).transpose(1, 2)
        return self.output(rwkv)


class RWKVChannelMixing(nn.Module):
    """RWKV-4 channel mixing (FFN equivalent)."""
    def __init__(self, d_model, expand=4):
        super().__init__()
        hidden = d_model * expand
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.key = nn.Linear(d_model, hidden, bias=False)
        self.value = nn.Linear(hidden, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        shifted = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        xk = x * self.time_mix_k + shifted * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + shifted * (1 - self.time_mix_r)
        k = torch.relu(self.key(xk)) ** 2  # squared ReLU (RWKV FFN)
        return torch.sigmoid(self.receptance(xr)) * self.value(k)


class RWKVBlock(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.time_mix = RWKVTimeMixing(d_model, n_heads)
        self.channel_mix = RWKVChannelMixing(d_model)

    def forward(self, x):
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x


class RWKVLM(nn.Module):
    """RWKV language model with N layers."""
    def __init__(self, vocab_size, d_model, n_layers, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            RWKVBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)


# ── 3) Standard Transformer (Flash Attention via SDPA) ────────────────────────

class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block with Flash Attention (via PyTorch SDPA)."""
    def __init__(self, d_model, n_heads, expand=4, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.RMSNorm(d_model)
        self.ln2 = nn.RMSNorm(d_model)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * expand, bias=False),
            nn.GELU(),
            nn.Linear(d_model * expand, d_model, bias=False),
        )

    def forward(self, x):
        B, S, D = x.shape
        h = self.ln1(x)

        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B, nH, S, hD]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash attention via SDPA (causal)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        x = x + self.o_proj(attn_out)

        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Standard Transformer language model."""
    def __init__(self, vocab_size, d_model, n_layers, n_heads=4, expand=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(65536, d_model)  # positional up to 64K
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, expand)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)


# ── 4) GRU Baseline ──────────────────────────────────────────────────────────

class GRULM(nn.Module):
    """GRU language model baseline."""
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, d_model, n_layers, batch_first=True)
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.gru(x)
        x = self.final_norm(x)
        return self.lm_head(x)


# ═══════════════════════════════════════════════════════════════════════════════
# OrthoSSM Wrapper (uses the real V11 engine)
# ═══════════════════════════════════════════════════════════════════════════════

def create_orthossm(vocab_size, d_model, n_layers):
    """Create OrthoSSM LM using the real V11 engine."""
    from model import OrthoSSMLanguageModel
    return OrthoSSMLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_attn_heads=4,
        n_cheby_heads=8,
        n_layers=n_layers,
        max_degree=4,
        window_size=512,
        use_bf16=True,
        tie_weights=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)
    return 0.0

def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def safe_forward(model, input_ids, max_retries=2):
    """Forward pass with OOM recovery."""
    for attempt in range(max_retries):
        try:
            return model(input_ids)
        except torch.cuda.OutOfMemoryError:
            gpu_reset()
            if attempt == max_retries - 1:
                return None
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# B5: MODEL INSTANTIATION + PARAMETER AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_b5():
    """B5: Instantiate all models and verify parameter counts."""
    print("\n" + "=" * 80)
    print("  B5: MODEL INSTANTIATION + PARAMETER COUNT AUDIT")
    print("=" * 80)

    device = 'cuda'
    # Configuration: target ~equal effective parameters
    # Note: with weight tying, effective params = total - vocab*d duplicate
    vocab = 4096  # small vocab for benchmark (fair: same for all)
    d = 256
    results = {}

    configs = {
        'OrthoSSM_V11': lambda: create_orthossm(vocab, d, n_layers=2),
        'Mamba':       lambda: MambaLM(vocab, d, n_layers=8, d_state=16, expand=2),
        'RWKV':        lambda: RWKVLM(vocab, d, n_layers=6, n_heads=4),
        'Transformer': lambda: TransformerLM(vocab, d, n_layers=6, n_heads=4, expand=4),
        'GRU':         lambda: GRULM(vocab, d, n_layers=4),
    }

    for name, create_fn in configs.items():
        gpu_reset()
        model = create_fn().to(device)
        n_params = count_params(model)
        results[name] = n_params

        # Quick smoke test
        test_ids = torch.randint(0, vocab, (1, 128), device=device)
        with torch.no_grad():
            out = safe_forward(model, test_ids)
        ok = out is not None and out.shape == (1, 128, vocab)
        status = "✓" if ok else "✗"

        print(f"  {status} {name:20s}  {n_params:>12,} params ({n_params/1e6:.1f}M)  "
              f"output={'OK' if ok else 'FAIL'}")
        del model
        gpu_reset()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# B6: NEEDLE-IN-A-HAYSTACK + ASSOCIATIVE RECALL
# ═══════════════════════════════════════════════════════════════════════════════

def create_niah_dataset(vocab_size, seq_len, n_samples=32, n_needles=4):
    """
    Create Needle-in-a-Haystack + Associative Recall dataset.

    Format: [haystack_tokens... KEY_i VALUE_i ... QUERY_KEY_i] → predict VALUE_i
    The model must remember key-value associations across long context.
    """
    # Reserve special tokens
    KEY_START = 1     # marks start of key region
    VAL_START = 2     # marks start of value region  
    QUERY_START = 3   # marks query region
    PAD = 0
    
    inputs = []
    targets = []  # target value token for each needle

    for _ in range(n_samples):
        seq = torch.randint(4, vocab_size, (seq_len,))  # random haystack

        # Place n_needles key-value pairs at random positions in first 80% of seq
        kv_pairs = []
        positions = sorted(torch.randint(10, int(seq_len * 0.7), (n_needles,)).tolist())
        for i, pos in enumerate(positions):
            key_tok = torch.randint(4, vocab_size, (1,)).item()
            val_tok = torch.randint(4, vocab_size, (1,)).item()
            kv_pairs.append((key_tok, val_tok))
            # Insert: KEY_START key_tok VAL_START val_tok
            if pos + 4 < seq_len:
                seq[pos] = KEY_START
                seq[pos + 1] = key_tok
                seq[pos + 2] = VAL_START
                seq[pos + 3] = val_tok

        # Place queries at the end of the sequence
        query_start = seq_len - n_needles * 3 - 1
        target_vals = []
        for i, (key_tok, val_tok) in enumerate(kv_pairs):
            qpos = query_start + i * 3
            if qpos + 2 < seq_len:
                seq[qpos] = QUERY_START
                seq[qpos + 1] = key_tok
                seq[qpos + 2] = PAD  # model should predict val_tok here
                target_vals.append((qpos + 2, val_tok))

        inputs.append(seq)
        targets.append(target_vals)

    return torch.stack(inputs), targets


def evaluate_niah(model, inputs, targets, device, batch_size=8):
    """Evaluate NIAH accuracy: what fraction of values does the model recall?"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size].to(device)
            logits = safe_forward(model, batch)
            if logits is None:
                return -1.0  # OOM

            batch_targets = targets[i:i + batch_size]
            for b_idx, target_list in enumerate(batch_targets):
                for pos, val in target_list:
                    if pos < logits.shape[1]:
                        pred = logits[b_idx, pos - 1].argmax().item()  # predict next token
                        if pred == val:
                            correct += 1
                        total += 1

    return correct / max(total, 1)


def benchmark_b6():
    """B6: Pattern learning test — how fast does each model learn sequences?"""
    print("\n" + "=" * 80)
    print("  B6: SEQUENCE PATTERN LEARNING (copy + predict)")
    print("  NOTE: Tests how quickly each model learns repeating patterns")
    print("  RWKV/Mamba use sequential Python scans (no custom CUDA),")
    print("  so training steps are limited for them.")
    print("=" * 80)

    device = 'cuda'
    vocab = 256  # small vocab → easier pattern
    d = 128       # smaller model → faster training
    seq_len = 256
    batch_size = 8
    n_steps = 100  # 100 training steps

    configs = {
        'OrthoSSM_V11': lambda: create_orthossm(vocab, d, n_layers=2),
        'Transformer': lambda: TransformerLM(vocab, d, n_layers=4, n_heads=4),
        'GRU':         lambda: GRULM(vocab, d, n_layers=3),
        'Mamba':       lambda: MambaLM(vocab, d, n_layers=4, d_state=16, expand=2),
    }

    # Generate deterministic repeating patterns
    # Pattern: ABCABC... where ABC is a fixed motif. Model must predict next token.
    motif_len = 16
    motifs = torch.randint(0, vocab, (8, motif_len))  # 8 distinct motifs
    data = []
    for _ in range(batch_size * (n_steps + 5)):
        midx = torch.randint(0, 8, (1,)).item()
        reps = seq_len // motif_len + 1
        seq = motifs[midx].repeat(reps)[:seq_len]
        data.append(seq)
    data = torch.stack(data)

    results = {}

    for name, create_fn in configs.items():
        print(f"\n  ── {name} ──")
        gpu_reset()
        model = create_fn().to(device)
        n_params = count_params(model)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        model.train()

        losses = []
        accs = []
        t0_total = time.perf_counter()

        for step in range(1, n_steps + 1):
            batch = data[(step - 1) * batch_size: step * batch_size].to(device)
            logits = model(batch)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), batch[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Accuracy: what fraction of next-token predictions are correct?
            with torch.no_grad():
                preds = logits[:, :-1].argmax(dim=-1)
                acc = (preds == batch[:, 1:]).float().mean().item()

            losses.append(loss.item())
            accs.append(acc)

            if step in [1, 10, 25, 50, 100]:
                print(f"    step {step:>3d}: loss={loss.item():.3f}  acc={acc*100:.1f}%")

        torch.cuda.synchronize()
        total_time = time.perf_counter() - t0_total

        results[name] = {
            'final_loss': losses[-1],
            'final_acc': accs[-1],
            'losses': losses,
            'accs': accs,
            'n_params': n_params,
            'total_time': total_time,
        }
        print(f"    FINAL: loss={losses[-1]:.3f}  acc={accs[-1]*100:.1f}%  "
              f"time={total_time:.1f}s  params={n_params/1e6:.1f}M")

        del model, optimizer
        gpu_reset()

    # Summary
    print("\n  PATTERN LEARNING SUMMARY (100 steps, motif_len=16, S=256):")
    print("  ┌─────────────────┬──────────┬──────────┬──────────┬──────────┐")
    print("  │ Model           │ Params   │ Loss@100 │ Acc@100  │ Time(s)  │")
    print("  ├─────────────────┼──────────┼──────────┼──────────┼──────────┤")
    for name in configs:
        r = results.get(name, {})
        print(f"  │ {name:15s} │ {r.get('n_params',0)/1e6:>5.1f}M  │ {r.get('final_loss',99):>8.3f} │"
              f" {r.get('final_acc',0)*100:>6.1f}% │ {r.get('total_time',0):>7.1f}s │")
    print("  └─────────────────┴──────────┴──────────┴──────────┴──────────┘")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# B7: THROUGHPUT + VRAM SCALING
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_b7():
    """B7: Throughput (tok/s) and VRAM scaling vs context length."""
    print("\n" + "=" * 80)
    print("  B7: THROUGHPUT + VRAM SCALING vs CONTEXT LENGTH")
    print("=" * 80)

    device = 'cuda'
    vocab = 4096
    d = 256
    batch = 1
    warmup = 2
    n_iters = 5

    # Different seq_lengths per model to avoid multi-hour runs on sequential scans
    fast_seqs  = [512, 1024, 2048, 4096, 8192, 16384]  # OrthoSSM, Transformer
    slow_seqs  = [512, 1024, 2048, 4096]                # Mamba, RWKV, GRU (sequential scan)

    configs = {
        'OrthoSSM_V11': (lambda: create_orthossm(vocab, d, n_layers=2), fast_seqs),
        'Mamba':       (lambda: MambaLM(vocab, d, n_layers=6, d_state=16, expand=2), slow_seqs),
        'RWKV':        (lambda: RWKVLM(vocab, d, n_layers=4, n_heads=4), slow_seqs),
        'Transformer': (lambda: TransformerLM(vocab, d, n_layers=4, n_heads=4), fast_seqs),
        'GRU':         (lambda: GRULM(vocab, d, n_layers=3), fast_seqs),
    }

    results = {}
    all_seqs = sorted(set(fast_seqs + slow_seqs))

    for name, (create_fn, seq_lengths) in configs.items():
        print(f"\n  ── {name} ──")
        model_results = {}

        for seq_len in seq_lengths:
            gpu_reset()
            model = create_fn().to(device).eval()
            test_ids = torch.randint(0, vocab, (batch, seq_len), device=device)

            # Adaptive iterations: fewer for slow models at long seqs
            effective_iters = max(2, n_iters) if seq_len <= 2048 else max(1, n_iters // 2)

            try:
                with torch.no_grad():
                    # Warmup
                    for _ in range(warmup):
                        out = model(test_ids)
                    torch.cuda.synchronize()

                    torch.cuda.reset_peak_memory_stats()

                    start = time.perf_counter()
                    for _ in range(effective_iters):
                        out = model(test_ids)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) / effective_iters

                    vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
                    tok_per_sec = (batch * seq_len) / elapsed

                    model_results[seq_len] = {
                        'ms': elapsed * 1000,
                        'tok_s': tok_per_sec,
                        'vram_mb': vram_mb,
                    }
                    print(f"    S={seq_len:>5d}: {elapsed*1000:>8.1f}ms  "
                          f"{tok_per_sec/1000:>7.1f}K tok/s  "
                          f"VRAM={vram_mb:>7.1f}MB")

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    model_results[seq_len] = {'ms': -1, 'tok_s': 0, 'vram_mb': -1}
                    print(f"    S={seq_len:>5d}: OOM")
                else:
                    model_results[seq_len] = {'ms': -1, 'tok_s': 0, 'vram_mb': -1}
                    print(f"    S={seq_len:>5d}: ERROR ({type(e).__name__})")
                gpu_reset()

            del model
            gpu_reset()

        results[name] = model_results

    # Summary tables
    print("\n  THROUGHPUT SUMMARY (K tok/s, higher is better):")
    header_seqs = all_seqs
    hdr = "  │ Model           │"
    for sl in header_seqs:
        hdr += f" S={sl:>5d} │"
    sep = "  ├─────────────────┼" + "─────────┼" * len(header_seqs)
    sep = sep[:-1] + "┤"
    top = "  ┌─────────────────┬" + "─────────┬" * len(header_seqs)
    top = top[:-1] + "┐"
    bot = "  └─────────────────┴" + "─────────┴" * len(header_seqs)
    bot = bot[:-1] + "┘"

    print(top)
    print(hdr)
    print(sep)
    for name in configs:
        row = f"  │ {name:15s} │"
        for sl in header_seqs:
            r = results.get(name, {}).get(sl, {})
            tok_s = r.get('tok_s', 0)
            if tok_s > 0:
                row += f" {tok_s/1000:>6.1f}K │"
            else:
                row += "    --  │"
        print(row)
    print(bot)

    print("\n  VRAM USAGE (MB):")
    print(top)
    print(hdr)
    print(sep)
    for name in configs:
        row = f"  │ {name:15s} │"
        for sl in header_seqs:
            r = results.get(name, {}).get(sl, {})
            vram = r.get('vram_mb', -1)
            if vram > 0:
                row += f" {vram:>6.0f}  │"
            else:
                row += "    --  │"
        print(row)
    print(bot)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# B8: SHORT TRAINING CONVERGENCE
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_b8():
    """B8: Training convergence comparison (300 steps on synthetic data)."""
    print("\n" + "=" * 80)
    print("  B8: TRAINING CONVERGENCE (300 steps, synthetic language data)")
    print("=" * 80)

    device = 'cuda'
    vocab = 4096
    d = 256
    batch_size = 4
    seq_len = 512
    n_steps = 300
    log_every = 30

    # Generate synthetic structured data (repeated patterns + noise)
    # This tests how well models learn sequential patterns
    print("  Generating structured training data...")
    data_size = batch_size * seq_len * (n_steps + 10)
    # Create data with learnable patterns: repeated subsequences + noise
    pattern_len = 32
    n_patterns = 16
    patterns = torch.randint(4, vocab, (n_patterns, pattern_len))
    data = torch.zeros(data_size, dtype=torch.long)
    for i in range(0, data_size - pattern_len, pattern_len):
        if torch.rand(1).item() < 0.7:  # 70% patterned, 30% noise
            pidx = torch.randint(0, n_patterns, (1,)).item()
            data[i:i + pattern_len] = patterns[pidx]
        else:
            data[i:i + pattern_len] = torch.randint(4, vocab, (pattern_len,))

    configs = {
        'OrthoSSM_V11': lambda: create_orthossm(vocab, d, n_layers=2),
        'Transformer': lambda: TransformerLM(vocab, d, n_layers=4, n_heads=4),
        'GRU':         lambda: GRULM(vocab, d, n_layers=3),
        'Mamba':       lambda: MambaLM(vocab, d, n_layers=6, d_state=16, expand=2),
    }

    results = {}

    for name, create_fn in configs.items():
        print(f"\n  ── {name} ──")
        gpu_reset()
        model = create_fn().to(device)
        n_params = count_params(model)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

        losses = []
        times = []
        model.train()

        data_ptr = 0
        total_time = 0.0

        for step in range(1, n_steps + 1):
            # Get batch
            batch_tokens = data[data_ptr:data_ptr + batch_size * seq_len].view(batch_size, seq_len).to(device)
            data_ptr += batch_size * seq_len
            if data_ptr + batch_size * seq_len >= len(data):
                data_ptr = 0

            t0 = time.perf_counter()

            logits = model(batch_tokens)
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, vocab),
                batch_tokens[:, 1:].contiguous().view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            step_time = t1 - t0
            total_time += step_time

            losses.append(loss.item())
            times.append(step_time)

            if step % log_every == 0 or step == 1:
                avg_loss = sum(losses[-log_every:]) / min(len(losses), log_every)
                avg_time = sum(times[-log_every:]) / min(len(times), log_every)
                tok_s = (batch_size * seq_len) / avg_time
                print(f"    step {step:>4d}/{n_steps}  loss={avg_loss:.4f}  "
                      f"{avg_time*1000:.0f}ms/step  {tok_s/1000:.1f}K tok/s")

        avg_last = sum(losses[-30:]) / min(30, len(losses))
        ppl_final = math.exp(min(avg_last, 20))
        results[name] = {
            'losses': losses,
            'final_loss': avg_last,
            'final_ppl': ppl_final,
            'avg_step_ms': sum(times) / len(times) * 1000,
            'total_time': total_time,
            'n_params': n_params,
            'tok_per_sec': (batch_size * seq_len) / (sum(times) / len(times)),
        }

        print(f"    FINAL: loss={losses[-1]:.4f}  PPL={ppl_final:.1f}  "
              f"total={total_time:.1f}s  params={n_params/1e6:.1f}M")

        del model, optimizer, scheduler
        gpu_reset()

    # Summary table
    print("\n  CONVERGENCE SUMMARY (300 steps, B=4, S=512):")
    print("  ┌─────────────────┬──────────┬──────────┬──────────┬───────────┬───────────┐")
    print("  │ Model           │ Params   │ Loss@300 │ PPL@300  │ ms/step   │ K tok/s   │")
    print("  ├─────────────────┼──────────┼──────────┼──────────┼───────────┼───────────┤")
    for name in configs:
        r = results.get(name, {})
        n_p = r.get('n_params', 0)
        loss = r.get('final_loss', 99)
        ppl = r.get('final_ppl', 9999)
        ms = r.get('avg_step_ms', 0)
        tok = r.get('tok_per_sec', 0)
        print(f"  │ {name:15s} │ {n_p/1e6:>5.1f}M  │ {loss:>8.4f} │ {ppl:>8.1f} │ {ms:>8.1f}  │ {tok/1000:>8.1f}  │")
    print("  └─────────────────┴──────────┴──────────┴──────────┴───────────┴───────────┘")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   OrthoSSM V11 — BENCHMARK vs SOTA (Mamba, RWKV, Transformer, GRU) ║")
    print("║   All baselines: Pure PyTorch (same backend, fair comparison)       ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    gpu = torch.cuda.get_device_properties(0)
    print(f"  GPU: {gpu.name}  VRAM: {gpu.total_memory // 1024**2}MB")

    all_results = {}

    # B5: Model instantiation
    all_results['B5'] = benchmark_b5()

    # B7: Throughput (run before B6/B8 since it's inference-only)
    all_results['B7'] = benchmark_b7()

    # B6: NIAH
    all_results['B6'] = benchmark_b6()

    # B8: Training convergence
    all_results['B8'] = benchmark_b8()

    # ── Final summary ──
    print("\n" + "=" * 80)
    print("  BENCHMARK COMPLETE")
    print("=" * 80)

    # Save results
    # Convert non-serializable items
    def json_safe(obj):
        if isinstance(obj, (float, int, str, bool, type(None))):
            return obj
        if isinstance(obj, dict):
            return {str(k): json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [json_safe(i) for i in obj]
        return str(obj)

    with open('benchmark_vs_sota_results.json', 'w') as f:
        json.dump(json_safe(all_results), f, indent=2)
    print("  Results saved to benchmark_vs_sota_results.json")


if __name__ == '__main__':
    main()
