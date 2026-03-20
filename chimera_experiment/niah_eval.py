"""
NIAH Eval — Needle in a Haystack para CHIMERA
==============================================
Generador de data sintética de estrés + evaluación formal de retrieval.

Qué prueba este módulo:
  El Landmark Archive y el TTT-Lite son los dos componentes que, en teoría,
  permiten a CHIMERA superar a Mamba2 puro en retrieval de largo rango. Sin
  training, el archive acumula 0 landmarks (confirmado en T6 del benchmark).
  Este módulo forza la activación del sistema completo mediante:

  1. NIAHDataset — genera secuencias con needles (clave→valor) en posiciones
     controladas, rodeadas de ruido ("haystack"), seguidas de una query
     que el modelo debe resolver.

  2. ChimeraLM — wrapper ligero que añade una cabeza LM (embedding lookup +
     linear) sobre AdvancedChimeraLayer para hacer retrieval concreto.

  3. train_niah() — 100-300 pasos de gradient descent forzando que el sistema
     active tier FULL (inputs complejos) y archive los needles.

  4. eval_niah() — mide accuracy@1 a múltiples longitudes de contexto y
     compara CHIMERA vs Mamba2-solo (baseline).

Protocolo de needle:
  Para evitar ambigüedad en el test (el modelo no conoce el vocabulario),
  usamos un vocabulario sintético pequeño:
    vocab_size = 512
    needle   = (key_token, value_token) — dos enteros distintos al ruido
    haystack = tokens uniformes U[0, vocab_size) que NO son key/value
    query    = key_token al final → el modelo debe predecir value_token

  Embedding: nn.Embedding(vocab_size, d_model) compartido entre input y cabeza LM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import json
import time
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from advanced_chimera import AdvancedChimeraLayer
from chimera_losses import ChimeraRoutingLoss

# ─────────────────────────────────────────────────────────────────────────────
# 1. Constantes y vocabulario sintético
# ─────────────────────────────────────────────────────────────────────────────

VOCAB_SIZE       = 512
HAYSTACK_RANGE   = (2, 400)   # tokens de ruido: IDs 2..399
NEEDLE_RANGE     = (400, 512) # tokens especiales: IDs 400..511 (keys + values)
PAD_ID           = 0
BOS_ID           = 1

# ─────────────────────────────────────────────────────────────────────────────
# 2. Generador de secuencias NIAH
# ─────────────────────────────────────────────────────────────────────────────

class NIAHDataset:
    """
    Generador online de secuencias NIAH (sin ficheros, bajo memoria).

    Secuencia generada [BOS | haystack_pre | key | sep | value | haystack_post | key]:
      - BOS: token de inicio
      - haystack_pre: ruido de longitud (depth_frac × context_len)
      - key: token clave (ID en NEEDLE_RANGE[0]..NEEDLE_RANGE[1]//2)
      - sep: token separador (ID 1 = BOS reutilizado como sep, convencion)
      - value: respuesta correcta (ID en NEEDLE_RANGE[1]//2..NEEDLE_RANGE[1])
      - haystack_post: ruido hasta completar context_len - 3 tokens
      - query: key repetida → el modelo debe predecir value en la posición siguiente

    Label: solo la posición query tiene target != -100 (→ value_id)
    """

    def __init__(
        self,
        context_lengths: list = [512, 1024, 2048],
        n_needles:       int  = 1,            # 1 = S-NIAH, >1 = MK-NIAH
        depth_fracs:     list = [0.1, 0.3, 0.5, 0.7, 0.9],  # dónde insertar needle
        seed:            int  = 42,
    ):
        self.context_lengths = context_lengths
        self.n_needles       = n_needles
        self.depth_fracs     = depth_fracs
        random.seed(seed)
        torch.manual_seed(seed)

        # Asignar pares (key, value) únicos por n_needles
        pool_keys   = list(range(NEEDLE_RANGE[0], (NEEDLE_RANGE[0]+NEEDLE_RANGE[1])//2))
        pool_values = list(range((NEEDLE_RANGE[0]+NEEDLE_RANGE[1])//2, NEEDLE_RANGE[1]))
        random.shuffle(pool_keys)
        random.shuffle(pool_values)
        self.key_ids   = pool_keys[:n_needles]
        self.value_ids = pool_values[:n_needles]

    def sample(self, context_len: int, depth_frac: float = None, device='cuda'):
        """
        Genera UNA secuencia NIAH de longitud context_len.
        Returns:
            tokens [1, context_len]  — input ids
            labels [1, context_len]  — -100 salvo en posiciones de query
            positions: list de (key_pos, value_pos, query_pos) por needle
        """
        if depth_frac is None:
            depth_frac = random.choice(self.depth_fracs)

        tokens = [BOS_ID]
        labels = [-100]
        positions = []

        # Espacio para n_needles × 3 tokens (key, sep, value) + query al final
        needle_tokens = self.n_needles * 3  # key + sep + value por needle
        query_tokens  = self.n_needles * 2  # key + label_position por needle
        haystack_total = context_len - 1 - needle_tokens - query_tokens

        # Posición de inserción del needle relativa al haystack
        pre_len = max(1, int(haystack_total * depth_frac))
        post_len = max(0, haystack_total - pre_len)

        def rand_hay(n):
            return [random.randint(*HAYSTACK_RANGE) for _ in range(n)]

        tokens += rand_hay(pre_len)
        labels += [-100] * pre_len

        # Insertar needles
        for i in range(self.n_needles):
            key_pos   = len(tokens)
            tokens   += [self.key_ids[i], BOS_ID, self.value_ids[i]]
            labels   += [-100, -100, -100]
            positions.append({'key': key_pos, 'value': key_pos + 2})

        tokens += rand_hay(post_len)
        labels += [-100] * post_len

        # Query section — preguntas al final
        for i in range(self.n_needles):
            query_pos = len(tokens)
            tokens   += [self.key_ids[i]]
            labels   += [-100]
            # Posición de la predicción (siguiente token)
            tokens   += [self.value_ids[i]]   # el modelo debe predecir esto
            labels   += [self.value_ids[i]]
            positions[i]['query'] = query_pos

        # Truncar / pad al tamaño exacto
        tokens = tokens[:context_len]
        labels = labels[:context_len]
        while len(tokens) < context_len:
            tokens.append(PAD_ID)
            labels.append(-100)

        t = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        l = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0)
        return t, l, positions


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mini-LM sobre CHIMERA
# ─────────────────────────────────────────────────────────────────────────────

class ChimeraLM(nn.Module):
    """
    CHIMERA + cabeza LM para NIAH.
    Usa embedding compartido (weight tying) entre input y output.
    """

    def __init__(self, d_model: int = 256, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocab_size

        # Embedding compartido
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)

        # CHIMERA layer
        self.chimera = AdvancedChimeraLayer(d_model=d_model, expand=2, headdim=32)

        # Cabeza LM — usa misma matriz de embedding (weight-tying)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight   # weight tying

        # Normalización final
        self.norm_out = nn.RMSNorm(d_model)

    def forward(self, input_ids: torch.Tensor, bus_cache=None, return_aux=False):
        """
        input_ids: [B, S]  long
        Returns: logits [B, S, vocab_size]
        """
        x = self.embed(input_ids)                  # [B, S, D]
        if return_aux:
            x, bus_cache, aux = self.chimera(x, bus_cache=bus_cache, return_aux=True)
        else:
            x, bus_cache = self.chimera(x, bus_cache=bus_cache)
            aux = None
        x = self.norm_out(x)
        logits = self.lm_head(x)                   # [B, S, V]
        if return_aux:
            return logits, bus_cache, aux
        return logits, bus_cache


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training loop NIAH
# ─────────────────────────────────────────────────────────────────────────────

def train_niah(
    model:          ChimeraLM,
    dataset:        NIAHDataset,
    context_lens:   list         = [512, 1024],
    n_steps:        int          = 200,
    lr:             float        = 3e-4,
    routing_coeff:  float        = 0.02,
    device:         str          = 'cuda',
    log_every:      int          = 20,
    seed:           int          = 42,
) -> list:
    """
    Entrena ChimeraLM en el task NIAH.
    Mezcla contextos de múltiples longitudes para que el router aprenda
    a usar tier FULL para secuencias complejas (needles profundos).

    Returns: lista de dicts con métricas por paso.
    """
    torch.manual_seed(seed)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    rl = ChimeraRoutingLoss(
        entropy_weight=0.05, supervision_weight=0.10, balance_weight=0.02
    ).to(device)

    history = []
    t0_total = time.time()

    for step in range(n_steps):
        # Curriculo: empezar con contextos cortos, ir creciendo
        curriculum_frac = min(1.0, step / (n_steps * 0.5))
        max_idx = max(1, int(len(context_lens) * curriculum_frac))
        ctx_len = random.choice(context_lens[:max_idx])
        depth   = random.uniform(0.1, 0.9)

        input_ids, labels, positions = dataset.sample(ctx_len, depth, device=device)

        opt.zero_grad()

        logits, _, aux = model(input_ids, bus_cache=None, return_aux=True)
        # [B, S, V] → cross-entropy ignorando -100
        task_loss = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )

        # Routing loss para forzar especialización
        r_loss, r_info = rl(aux)
        total = task_loss + routing_coeff * r_loss
        total.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        if step % log_every == 0:
            ppl = math.exp(min(task_loss.item(), 20))
            info = {
                'step':     step,
                'ctx_len':  ctx_len,
                'depth':    round(depth, 2),
                'loss':     round(task_loss.item(), 4),
                'ppl':      round(ppl, 1),
                'r_loss':   round(r_loss.item(), 5),
                'H_bits':   round(r_info['routing/H_bits'], 4),
                'p_full':   round(r_info['routing/p_full'], 4),
            }
            history.append(info)
            elapsed = time.time() - t0_total
            print(f"  step {step:>4} | ctx={ctx_len:>5} | loss={info['loss']:.4f} "
                  f"| ppl={info['ppl']:>7.1f} | p_full={info['p_full']:.3f} "
                  f"| H={info['H_bits']:.3f} | {elapsed:.1f}s")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluación: accuracy@1 por contexto y profundidad
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_niah(
    model:       ChimeraLM,
    dataset:     NIAHDataset,
    context_lens: list = [512, 1024, 2048],
    n_samples:   int  = 50,
    device:      str  = 'cuda',
) -> dict:
    """
    Evalúa accuracy@1 para cada combinación (context_len, depth_frac).
    Devuelve un dict con matrices de accuracy.
    """
    model.eval()
    results = {}

    for ctx_len in context_lens:
        acc_by_depth = {}
        for depth in dataset.depth_fracs:
            correct = 0
            for _ in range(n_samples):
                input_ids, labels, positions = dataset.sample(ctx_len, depth, device=device)

                logits, _ = model(input_ids, bus_cache=None)   # [1, S, V]
                # Evaluar en las posiciones con label != -100
                mask = labels[0] != -100   # [S]
                if mask.sum() == 0:
                    continue
                pred_ids   = logits[0].argmax(dim=-1)   # [S]
                true_ids   = labels[0]                  # [S], -100 donde no cuenta
                correct   += ((pred_ids == true_ids) & mask).sum().item()

            total = n_samples * dataset.n_needles
            acc_by_depth[round(depth, 2)] = round(correct / max(total, 1), 4)

        results[ctx_len] = acc_by_depth

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. Baseline: Mamba2-solo (sin SLR / archive)
# ─────────────────────────────────────────────────────────────────────────────

class Mamba2LM(nn.Module):
    """
    Baseline Mamba2-solo, sin SLR, bus ni archive.
    Usa exactamente los mismos hyperparámetros que ChimeraLM para comparación justa.
    """

    def __init__(self, d_model: int = 256, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        from mamba_ssm import Mamba2
        self.d_model    = d_model
        self.vocab_size = vocab_size
        self.embed      = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        d_inner         = d_model * 2
        nheads          = d_inner // 32
        self.mamba2     = Mamba2(
            d_model=d_model, d_state=64, d_conv=4,
            expand=2, headdim=32, ngroups=1,
        )
        self.norm       = nn.RMSNorm(d_model)
        self.lm_head    = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, **kwargs):
        x      = self.embed(input_ids)
        x      = x + self.mamba2(self.norm(x))
        logits = self.lm_head(x)
        return logits, None


def train_baseline(model, dataset, context_lens, n_steps, lr, device, log_every):
    """Training idéntico para el baseline Mamba2."""
    model.train()
    opt  = torch.optim.AdamW(model.parameters(), lr=lr)
    sched= torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    history = []
    for step in range(n_steps):
        ctx_len = random.choice(context_lens[:max(1, int(len(context_lens)*min(1.0, step/(n_steps*0.5))))])
        depth   = random.uniform(0.1, 0.9)
        input_ids, labels, _ = dataset.sample(ctx_len, depth, device=device)
        opt.zero_grad()
        logits, _ = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, model.vocab_size), labels.view(-1), ignore_index=-100)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0:
            history.append({'step': step, 'loss': round(loss.item(), 4)})
    return history


@torch.no_grad()
def eval_baseline(model, dataset, context_lens, n_samples, device):
    model.eval()
    results = {}
    for ctx_len in context_lens:
        acc_by_depth = {}
        for depth in dataset.depth_fracs:
            correct = 0
            for _ in range(n_samples):
                input_ids, labels, _ = dataset.sample(ctx_len, depth, device=device)
                logits, _ = model(input_ids)
                mask     = labels[0] != -100
                if mask.sum() == 0:
                    continue
                pred_ids = logits[0].argmax(-1)
                correct += ((pred_ids == labels[0]) & mask).sum().item()
            acc_by_depth[round(depth, 2)] = round(correct / max(n_samples * dataset.n_needles, 1), 4)
        results[ctx_len] = acc_by_depth
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. Runner principal
# ─────────────────────────────────────────────────────────────────────────────

def run_niah_benchmark(
    context_lens:  list = [512, 1024, 2048],
    n_needles:     int  = 1,
    n_train_steps: int  = 150,
    n_eval:        int  = 30,
    d_model:       int  = 256,
    device:        str  = 'cuda',
    save_results:  bool = True,
):
    print("=" * 68)
    print("  CHIMERA NIAH BENCHMARK — Needle in a Haystack")
    print(f"  context_lens={context_lens}  n_needles={n_needles}")
    print("=" * 68)

    dataset = NIAHDataset(
        context_lengths=context_lens,
        n_needles=n_needles,
        depth_fracs=[0.1, 0.3, 0.5, 0.7, 0.9],
    )

    # ── CHIMERA ──────────────────────────────────────────────────────────────
    chimera_lm = ChimeraLM(d_model=d_model, vocab_size=VOCAB_SIZE).to(device).float()
    n_params_c = sum(p.numel() for p in chimera_lm.parameters())
    print(f"\n  CHIMERA params: {n_params_c:,}")

    print("\n--- Training CHIMERA ---")
    history_c = train_niah(
        chimera_lm, dataset,
        context_lens=context_lens,
        n_steps=n_train_steps,
        device=device,
        log_every=max(1, n_train_steps // 8),
    )

    print("\n--- Evaluating CHIMERA ---")
    acc_chimera = eval_niah(chimera_lm, dataset, context_lens, n_eval, device)

    # ── Mamba2 Baseline ──────────────────────────────────────────────────────
    try:
        mamba_lm = Mamba2LM(d_model=d_model, vocab_size=VOCAB_SIZE).to(device).float()
        n_params_m = sum(p.numel() for p in mamba_lm.parameters())
        print(f"\n  Mamba2 baseline params: {n_params_m:,}")

        print("\n--- Training Mamba2 baseline ---")
        train_baseline(mamba_lm, dataset, context_lens, n_train_steps,
                       lr=3e-4, device=device, log_every=max(1, n_train_steps // 8))

        print("\n--- Evaluating Mamba2 baseline ---")
        acc_mamba = eval_baseline(mamba_lm, dataset, context_lens, n_eval, device)
        baseline_available = True
    except Exception as e:
        print(f"  [WARNING] Mamba2 baseline skipped: {e}")
        acc_mamba = {}
        baseline_available = False

    # ── Tabla de resultados ───────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  NIAH ACCURACY RESULTS  (accuracy@1)")
    print("=" * 68)
    header = f"  {'Context':>8}  {'Depth':>6}  {'CHIMERA':>9}"
    if baseline_available:
        header += f"  {'Mamba2':>8}  {'Δ':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_results = []
    for ctx in context_lens:
        for depth in dataset.depth_fracs:
            d = round(depth, 2)
            c_acc = acc_chimera.get(ctx, {}).get(d, 0.0)
            row = f"  {ctx:>8}  {d:>6.2f}  {c_acc:>9.4f}"
            if baseline_available:
                m_acc = acc_mamba.get(ctx, {}).get(d, 0.0)
                delta = c_acc - m_acc
                row += f"  {m_acc:>8.4f}  {delta:>+7.4f}"
            print(row)
            all_results.append({'ctx': ctx, 'depth': d, 'chimera': c_acc,
                                 'mamba2': acc_mamba.get(ctx, {}).get(d, None)})

    # Resumen
    chimera_mean = sum(r['chimera'] for r in all_results) / len(all_results)
    print(f"\n  CHIMERA mean accuracy: {chimera_mean:.4f}")
    if baseline_available:
        mamba_mean = sum(r['mamba2'] for r in all_results if r['mamba2'] is not None) / len(all_results)
        print(f"  Mamba2  mean accuracy: {mamba_mean:.4f}")
        print(f"  CHIMERA advantage:     {chimera_mean - mamba_mean:+.4f}")

    if save_results:
        out = {
            'context_lens':   context_lens,
            'n_needles':      n_needles,
            'n_train_steps':  n_train_steps,
            'acc_chimera':    {str(k): v for k, v in acc_chimera.items()},
            'acc_mamba2':     {str(k): v for k, v in acc_mamba.items()},
            'training_history': history_c,
            'chimera_mean_acc': chimera_mean,
        }
        path = os.path.join(os.path.dirname(__file__), 'niah_results.json')
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\n  Resultados guardados: {path}")

    print("\n" + "=" * 68)
    return acc_chimera, acc_mamba


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Single-needle, contextos moderados (ajustados a 6GB VRAM)
    run_niah_benchmark(
        context_lens  = [256, 512, 1024],
        n_needles     = 1,
        n_train_steps = 120,
        n_eval        = 25,
        d_model       = 256,
        device        = device,
    )

    print("\n--- MK-NIAH (3 needles) ---")
    run_niah_benchmark(
        context_lens  = [256, 512],
        n_needles     = 3,
        n_train_steps = 100,
        n_eval        = 20,
        d_model       = 256,
        device        = device,
    )
