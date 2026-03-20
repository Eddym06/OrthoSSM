"""
Mechanistic Interpretability — CHIMERA
=======================================
Herramientas para analizar qué aprende internamente cada componente de CHIMERA:

  1. bus_cache_probe()       — ¿qué posiciones domina el bus_cache?
  2. routing_probe()         — PCA del espacio de decisión del router
  3. lambda_tracker()        — evolución de λ (DiffAttn) a lo largo del entrenamiento
  4. sgr_selection_heatmap() — heat-map de los top-K tokens seleccionados por SGR
  5. landmark_content_profile() — qué posiciones de input se almacenan en landmarks
  6. dt_bias_evolution()     — cómo adapta TTT su escala temporal en secuencias largas
  7. run_all_analyses()      — wrapper que ejecuta todo y guarda PNGs + JSON

Flujo de trabajo:
  model = ChimeraLM(...)   # del niah_eval.py (o cualquier ChimeraLM)
  x = ...                  # tensor [B, S, D]
  run_all_analyses(model, x, out_dir='/home/OrthoSSM/chimera_experiment/interp_out')

Dependencias opcionales:
  - matplotlib  → genera PNGs; si no está, imprime stats y guarda JSON
  - sklearn     → PCA; si no está, usa SVD manual
"""
import torch
import torch.nn.functional as F
import json
import os
import math
from typing import Optional, Dict, Any, List

# Matplotlib es opcional — fallback sin plots visuales
try:
    import matplotlib
    matplotlib.use("Agg")              # no-display backend
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    print("[mech_interp] matplotlib no instalado — solo JSON stats")

# sklearn PCA opcional
try:
    from sklearn.decomposition import PCA as SklearnPCA
    _HAS_SKL = True
except ImportError:
    _HAS_SKL = False


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades comunes
# ─────────────────────────────────────────────────────────────────────────────

def _pca_2d(X: torch.Tensor) -> torch.Tensor:
    """
    X: [N, D] — proyecta a 2D con PCA.
    Usa sklearn si está disponible, si no SVD de PyTorch.
    """
    X_cpu = X.detach().cpu().float()
    X_c   = X_cpu - X_cpu.mean(0, keepdim=True)

    if _HAS_SKL:
        pca    = SklearnPCA(n_components=2)
        coords = torch.from_numpy(pca.fit_transform(X_c.numpy()))
    else:
        U, S, V = torch.linalg.svd(X_c, full_matrices=False)
        coords  = U[:, :2] * S[:2]
    return coords  # [N, 2]


def _save_fig(fig, path: str):
    if _HAS_MPL and fig is not None:
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  [mech_interp] figura guardada: {path}")


def _hook_collector(target_module, attr="last_output"):
    """Registra un forward hook que guarda la salida del módulo."""
    storage = {}

    def hook(m, inp, out):
        if isinstance(out, tuple):
            storage[attr] = out[0].detach()
        else:
            storage[attr] = out.detach()

    handle = target_module.register_forward_hook(hook)
    return storage, handle


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bus cache probe — ¿qué posiciones representa el bus_cache?
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def bus_cache_probe(
    model,
    x: torch.Tensor,           # [B, S, D]
    out_dir: str = ".",
) -> Dict[str, Any]:
    """
    Mide la similaridad coseno entre los vectores del bus_cache
    y cada posición del input x, revelando qué posiciones «resumen»
    el bus guarda.

    Retorna: {
      'sim_matrix': [[float]] (S_cache × S_input),
      'top_positions': [int],   (top-3 posiciones de input por slot de cache)
    }
    """
    model.eval()
    x = x.float()
    B, S, D = x.shape

    # Forward con return_aux para capturar bus_cache
    with torch.no_grad():
        x_in = x
        # Acceder al bus_cache del chimera layer directamente
        chimera = model.chimera if hasattr(model, 'chimera') else model
        out  = chimera(x_in, bus_cache=None, return_aux=True)
        if len(out) == 3:
            out_x, bus_cache, aux = out
        elif len(out) == 2:
            out_x, bus_cache = out
            aux = None
        else:
            out_x = out[0]; bus_cache = out[1]; aux = None

    if bus_cache is None or not isinstance(bus_cache, dict) or 'bus_cache' not in bus_cache:
        # bus_cache puede ser un dict con clave 'bus_cache'
        print("  [bus_cache_probe] bus_cache no disponible, usando out_x como proxy")
        cache_vecs = out_x[0]           # [S, D]
    else:
        raw = bus_cache.get('bus_cache', out_x[0])
        if raw.dim() == 3:
            cache_vecs = raw[0]         # [N_cache, D]
        else:
            cache_vecs = raw

    # Normalizar para cosine
    x_norm     = F.normalize(x[0].float(), dim=-1)          # [S, D]
    cache_norm = F.normalize(cache_vecs.float(), dim=-1)     # [N, D]
    sim        = cache_norm @ x_norm.T                       # [N, S]

    top_positions = sim.argmax(dim=-1).tolist()              # [N]
    sim_np        = sim.cpu().tolist()

    stats = {
        'sim_matrix':    sim_np[:8],   # máx 8 filas para evitar JSON enorme
        'top_positions': top_positions,
        'mean_sim':      float(sim.mean()),
        'max_sim':       float(sim.max()),
    }

    if _HAS_MPL:
        fig, ax = plt.subplots(figsize=(min(S // 32, 20), 4))
        show = sim[:8, :].cpu()
        im   = ax.imshow(show, aspect='auto', cmap='hot', vmin=0, vmax=1)
        ax.set_xlabel("Input position")
        ax.set_ylabel("Bus cache slot")
        ax.set_title("Bus cache ↔ Input cosine similarity")
        plt.colorbar(im, ax=ax)
        _save_fig(fig, os.path.join(out_dir, "bus_cache_probe.png"))

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 2. Routing probe — PCA del espacio de decisión del router
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def routing_probe(
    model,
    sequences: List[torch.Tensor],     # lista de [1, S, D]
    out_dir: str = ".",
) -> Dict[str, Any]:
    """
    Recolecta routing_probs de N forward passes, aplica PCA 2D a las representaciones
    intermedias y colorea por tier dominante (0=fast, 1=medium, 2=full).

    Retorna: {
      'coords': [[x, y], ...],
      'tier_labels': [int, ...],
      'explained_var': [float, float],
    }
    """
    model.eval()
    chimera = model.chimera if hasattr(model, 'chimera') else model

    all_probs   = []   # recolectar probs [B*S, 3]
    all_hidden  = []   # hidden state antes del router [B*S, D]

    # Hook en la capa de pre-routing (la norma de entrada del chimera)
    storage  = {}
    def hook_pre(m, inp, out):
        storage['pre_router'] = out.detach() if not isinstance(out, tuple) else out[0].detach()

    handle = None
    if hasattr(chimera, 'norm_in'):
        handle = chimera.norm_in.register_forward_hook(hook_pre)

    for seq in sequences[:32]:            # máx 32 secuencias
        seq = seq.float()
        out = chimera(seq, bus_cache=None, return_aux=True)
        if len(out) >= 3:
            _, _, aux = out[0], out[1], out[2]
            if aux and 'routing_probs' in aux:
                probs = aux['routing_probs']   # [B, S, 3] o [B*S, 3]
                if probs.dim() == 3:
                    all_probs.append(probs.reshape(-1, probs.shape[-1]).cpu())
                else:
                    all_probs.append(probs.cpu())
        if 'pre_router' in storage:
            h = storage['pre_router'].reshape(-1, storage['pre_router'].shape[-1])
            all_hidden.append(h.cpu())
            storage.clear()

    if handle:
        handle.remove()

    if not all_probs:
        print("  [routing_probe] routing_probs no disponibles — ¿model entrenado?")
        return {}

    probs_all  = torch.cat(all_probs, dim=0).float()    # [N, 3]
    tier_labels= probs_all.argmax(dim=-1).tolist()      # tier dominante por token

    if all_hidden:
        hidden_all = torch.cat(all_hidden, dim=0).float()
        coords     = _pca_2d(hidden_all)
    else:
        coords     = _pca_2d(probs_all)                 # PCA sobre probs en su defecto

    explained = [0.0, 0.0]   # no disponible sin sklearn varianza explicada

    # Plot
    if _HAS_MPL:
        colors    = ['tab:blue', 'tab:orange', 'tab:red']
        tier_names= ['fast', 'medium', 'full']
        fig, ax   = plt.subplots(figsize=(7, 6))
        c2        = coords.numpy()
        for t, color, name in zip(range(3), colors, tier_names):
            idx = [i for i, l in enumerate(tier_labels) if l == t]
            if idx:
                ax.scatter(c2[idx, 0], c2[idx, 1], c=color, label=name, alpha=0.4, s=10)
        ax.legend()
        ax.set_title("Router decision space (PCA 2D)")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        _save_fig(fig, os.path.join(out_dir, "routing_probe.png"))

    # Distribución global de tiers
    tier_counts = [tier_labels.count(t) for t in range(3)]
    n = len(tier_labels)
    return {
        'n_tokens':      n,
        'tier_dist':     [round(c / max(n, 1), 4) for c in tier_counts],
        'tier_labels':   tier_labels[:200],   # muestra en JSON
        'coords':        coords[:200].tolist(),
        'explained_var': explained,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Lambda tracker — evolución de λ SLR a lo largo de pasos
# ─────────────────────────────────────────────────────────────────────────────

def lambda_tracker(
    model,
    n_steps:  int  = 50,
    out_dir:  str  = ".",
) -> Dict[str, Any]:
    """
    Registra lam_logit.sigmoid() del módulo SLR a inicio, mitad y final
    de un mini-entrenamiento de identidad (target = input), mostrando
    cómo λ evoluciona.

    No necesita datos reales — usa ruido gaussiano.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from sgr_slr import SLRDifferentialModule

    # Buscar módulos SLR en el modelo
    slr_mods = [m for m in model.modules() if isinstance(m, SLRDifferentialModule)]
    if not slr_mods:
        print("  [lambda_tracker] No hay módulos SLRDifferentialModule")
        return {}

    slr = slr_mods[0]
    d_model = model.d_model

    def get_lambda():
        if hasattr(slr, 'lam_logit'):
            return slr.lam_logit.detach().sigmoid().mean().item()
        return None

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    device = next(model.parameters()).device
    history = []

    for step in range(n_steps):
        x = torch.randn(1, 64, d_model, device=device)
        opt.zero_grad()
        if hasattr(model, 'chimera'):
            out, _ = model.chimera(x)
        else:
            out, _ = model(x)
        loss = F.mse_loss(out, x)
        loss.backward()
        opt.step()

        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            lam = get_lambda()
            history.append({'step': step, 'lambda': lam, 'loss': round(loss.item(), 5)})

    lams = [h['lambda'] for h in history if h['lambda'] is not None]

    if _HAS_MPL and lams:
        steps = [h['step'] for h in history if h['lambda'] is not None]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(steps, lams, marker='o')
        axes[0].set_title("λ (differential attn blend) evolution")
        axes[0].set_xlabel("Step"); axes[0].set_ylabel("λ (sigmoid)")
        axes[0].set_ylim(0, 1)

        losses = [h['loss'] for h in history]
        axes[1].plot([h['step'] for h in history], losses, color='orange', marker='x')
        axes[1].set_title("Identity loss")
        axes[1].set_xlabel("Step"); axes[1].set_ylabel("MSE")

        plt.tight_layout()
        _save_fig(fig, os.path.join(out_dir, "lambda_tracker.png"))

    return {'history': history, 'lambda_init': lams[0] if lams else None,
            'lambda_final': lams[-1] if lams else None}


# ─────────────────────────────────────────────────────────────────────────────
# 4. SGR selection heatmap — qué posiciones elige el selector
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sgr_selection_heatmap(
    model,
    x: torch.Tensor,          # [B, S, D]
    out_dir: str = ".",
) -> Dict[str, Any]:
    """
    Hookeamos SGRSelector para ver qué posiciones del input son seleccionadas
    por el top-K gate de SGR.

    Muestra una heat-map de frecuencia de selección por posición.
    """
    from sgr_slr import SGRSelector

    selectors = [m for m in model.modules() if isinstance(m, SGRSelector)]
    if not selectors:
        print("  [sgr_selection_heatmap] No hay módulos SGRSelector")
        return {}

    count_store: Dict[int, torch.Tensor] = {}
    handles = []

    for idx, sel in enumerate(selectors):
        def make_hook(i):
            def hook(m, inp, out):
                # SGRSelector.forward() retorna (top_idx [B,K], K_int)
                if isinstance(out, tuple) and len(out) >= 1:
                    indices = out[0]          # primer elemento = top_idx [B, K]
                    if not isinstance(indices, torch.Tensor):
                        return               # no es tensor, saltar
                    if i not in count_store:
                        count_store[i] = torch.zeros(x.shape[1], dtype=torch.float)
                    flat = indices.reshape(-1).cpu()
                    flat = flat[flat < x.shape[1]]
                    for pos in flat.tolist():
                        count_store[i][int(pos)] += 1
                elif isinstance(out, torch.Tensor):
                    indices = out
                    if i not in count_store:
                        count_store[i] = torch.zeros(x.shape[1], dtype=torch.float)
                    flat = indices.reshape(-1).cpu()
                    flat = flat[flat < x.shape[1]]
                    for pos in flat.tolist():
                        count_store[i][int(pos)] += 1
            return hook
        handles.append(sel.register_forward_hook(make_hook(idx)))

    model.eval()
    chimera = model.chimera if hasattr(model, 'chimera') else model
    chimera(x.float(), bus_cache=None)

    for h in handles:
        h.remove()

    if not count_store:
        print("  [sgr_selection_heatmap] hook no capturó datos (SGR retorna distinto formato)")
        return {}

    if _HAS_MPL:
        n_sess = len(count_store)
        fig, axes = plt.subplots(n_sess, 1, figsize=(12, 3 * n_sess + 1))
        if n_sess == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            freq = count_store.get(i, torch.zeros(x.shape[1]))
            ax.bar(range(len(freq)), freq.tolist(), color='steelblue', width=1)
            ax.set_title(f"SGR selector {i} — selection frequency by position")
            ax.set_xlabel("Token position"); ax.set_ylabel("Count")
        plt.tight_layout()
        _save_fig(fig, os.path.join(out_dir, "sgr_selection_heatmap.png"))

    return {
        'n_selectors': len(selectors),
        'heatmap_shape': [len(count_store), x.shape[1]],
        'coverage_frac': float((count_store[0] > 0).float().mean()) if 0 in count_store else 0.0,
        'freq_by_selector': {
            i: count_store[i].tolist() for i in count_store
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Landmark content profile — qué posiciones de input acaban en el archive
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def landmark_content_profile(
    model,
    x: torch.Tensor,          # [B, S, D]
    out_dir: str = ".",
) -> Dict[str, Any]:
    """
    Después de un forward, recupera los landmarks almacenados y mide
    su similaridad coseno con cada posición del input para identificar
    qué fragmentos del input acabaron archivados.
    """
    from landmark_native import NativeLandmarkArchive

    archives = [m for m in model.modules() if isinstance(m, NativeLandmarkArchive)]
    if not archives:
        print("  [landmark_content_profile] No hay NativeLandmarkArchive")
        return {}

    model.eval()
    chimera = model.chimera if hasattr(model, 'chimera') else model
    chimera(x.float(), bus_cache=None)

    results = []
    for arch_idx, arch in enumerate(archives):
        info = arch.get_archive_info()
        n_stored = info.get('n_stored', 0)
        print(f"  Archive {arch_idx}: {n_stored} landmarks stored")

        if n_stored == 0:
            results.append({'archive': arch_idx, 'n_stored': 0})
            continue

        # Recuperar todos los landmarks con una query genérica (media del input)
        query = x[0].mean(0, keepdim=True)   # [1, D]
        retrieved = arch.retrieve(query)      # [K, D]

        if retrieved is None or retrieved.numel() == 0:
            results.append({'archive': arch_idx, 'n_stored': n_stored, 'retrieved': 0})
            continue

        # Similaridad con el input
        x_norm   = F.normalize(x[0].float(), dim=-1)         # [S, D]
        r_norm   = F.normalize(retrieved.float(), dim=-1)    # [K, D]
        sim      = r_norm @ x_norm.T                         # [K, S]
        top_pos  = sim.argmax(dim=-1).tolist()

        if _HAS_MPL and sim.shape[0] <= 32:
            fig, ax = plt.subplots(figsize=(max(8, x.shape[1] // 32), 4))
            ax.imshow(sim.cpu().numpy(), aspect='auto', cmap='viridis')
            ax.set_title(f"Archive {arch_idx}: landmark ↔ input similarity")
            ax.set_xlabel("Input position"); ax.set_ylabel("Landmark")
            _save_fig(fig, os.path.join(out_dir, f"landmark_profile_{arch_idx}.png"))

        results.append({
            'archive':   arch_idx,
            'n_stored':  n_stored,
            'retrieved': retrieved.shape[0],
            'top_input_positions': top_pos,
            'mean_sim': float(sim.mean()),
        })

    return {'archives': results}


# ─────────────────────────────────────────────────────────────────────────────
# 6. dt_bias evolution — cómo adapta TTT su escala temporal
# ─────────────────────────────────────────────────────────────────────────────

def dt_bias_evolution(
    model,
    seq_lengths: List[int] = [64, 256, 512, 1024],
    out_dir: str = ".",
) -> Dict[str, Any]:
    """
    Para cada longitud de secuencia, ejecuta un forward e inspecciona dt_bias
    del módulo TTT (el sesgo que controla la escala temporal de actualización).
    Muestra si TTT aumenta/reduce la plasticidad para secuencias más largas.
    """
    # Buscar módulo TTT (contiene dt_bias)
    ttt_mods = [(name, m) for name, m in model.named_modules()
                if hasattr(m, 'dt_bias') or 'ttt' in name.lower()]

    if not ttt_mods:
        print("  [dt_bias_evolution] No se encontró módulo TTT con dt_bias")
        return {}

    model.eval()
    d_model = model.d_model
    device  = next(model.parameters()).device

    history = {}
    chimera = model.chimera if hasattr(model, 'chimera') else model

    def get_dt_bias():
        for _, m in ttt_mods:
            if hasattr(m, 'dt_bias'):
                return m.dt_bias.detach().sigmoid().mean().item()
        return None

    for S in seq_lengths:
        x = torch.zeros(1, S, d_model, device=device)
        with torch.no_grad():
            chimera(x)
        dt = get_dt_bias()
        history[S] = dt
        print(f"  S={S:>5}: dt_bias(σ)={dt:.4f}" if dt else f"  S={S}: dt_bias N/A")

    if _HAS_MPL and any(v is not None for v in history.values()):
        seqs = [s for s, v in history.items() if v is not None]
        vals = [history[s] for s in seqs]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(seqs, vals, marker='s', color='teal')
        ax.set_title("TTT dt_bias (σ) vs sequence length")
        ax.set_xlabel("Sequence length"); ax.set_ylabel("dt_bias sigmoid")
        ax.set_ylim(0, 1)
        _save_fig(fig, os.path.join(out_dir, "dt_bias_evolution.png"))

    return {'dt_bias_by_length': {str(k): v for k, v in history.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# 7. run_all_analyses — wrapper principal
# ─────────────────────────────────────────────────────────────────────────────

def run_all_analyses(
    model,
    x:       torch.Tensor,
    out_dir: str = None,
    n_steps: int = 30,
) -> Dict[str, Any]:
    """
    Ejecuta los 6 análisis y guarda todos los resultados en out_dir.
    """
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "interp_out")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CHIMERA Mechanistic Interpretability")
    print(f"  input: {list(x.shape)}  out_dir: {out_dir}")
    print(f"{'='*60}\n")

    all_results = {}

    print("--- 1. Bus cache probe ---")
    all_results['bus_cache'] = bus_cache_probe(model, x, out_dir)

    print("\n--- 2. Routing probe ---")
    # Genera 16 secuencias de ruido para el probe
    device = x.device
    B, S, D = x.shape
    seqs = [torch.randn(1, S, D, device=device) for _ in range(16)]
    all_results['routing'] = routing_probe(model, seqs, out_dir)

    print("\n--- 3. Lambda tracker ---")
    all_results['lambda'] = lambda_tracker(model, n_steps=n_steps, out_dir=out_dir)

    print("\n--- 4. SGR selection heatmap ---")
    all_results['sgr'] = sgr_selection_heatmap(model, x, out_dir)

    print("\n--- 5. Landmark content profile ---")
    all_results['landmarks'] = landmark_content_profile(model, x, out_dir)

    print("\n--- 6. dt_bias evolution ---")
    B, S, D = x.shape
    all_results['dt_bias'] = dt_bias_evolution(
        model, seq_lengths=[min(S, l) for l in [32, 64, 128, 256] if l <= S],
        out_dir=out_dir
    )

    # Guardar resumen JSON
    json_path = os.path.join(out_dir, "interp_summary.json")
    def _to_json_safe(obj):
        if isinstance(obj, dict):
            return {k: _to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_json_safe(v) for v in obj]
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)

    with open(json_path, 'w') as f:
        json.dump(_to_json_safe(all_results), f, indent=2)
    print(f"\n  Resumen guardado: {json_path}")

    # Imprimir texto de calidad rápida
    print(f"\n{'='*60}")
    print("  RESUMEN RAPID:")
    rt = all_results.get('routing', {})
    if rt:
        td = rt.get('tier_dist', [])
        if td:
            print(f"  Routing: fast={td[0]:.1%}  medium={td[1]:.1%}  full={td[2]:.1%}")
    lm = all_results.get('lambda', {})
    if lm:
        print(f"  Lambda SLR: {lm.get('lambda_init'):.4f} → {lm.get('lambda_final'):.4f}")
    lk = all_results.get('landmarks', {})
    for a in lk.get('archives', []):
        print(f"  Archive {a['archive']}: {a.get('n_stored', 0)} landmarks stored")
    print(f"{'='*60}\n")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from niah_eval import ChimeraLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Crear un modelo pequeño y un tensor de entrada de prueba
    model = ChimeraLM(d_model=256, vocab_size=512).to(device).float()
    B, S, D = 1, 256, 256
    x = torch.randn(B, S, D, device=device)

    results = run_all_analyses(model, x, n_steps=20)
    print(f"\n[OK] mech_interp.py completado sin errores.")
