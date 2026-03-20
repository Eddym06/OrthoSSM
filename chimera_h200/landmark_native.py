"""
NativeLandmarkArchive — reconstrucción nativa para el ecosistema CHIMERA
=======================================================================
Reemplaza landmark_archive.py (OrthoSSM legacy) con una implementación que:

  • Archiva embeddings comprimidos del output del SSD scan de Mamba2
    (en lugar de cheby_state — que no existe en CHIMERA)
  • Usa el error TTT (per_token_err) como señal de importancia directa
    (elimina el importance_predictor MLP de 2 capas — redundante)
  • Usa diff_attn_v2_triton para landmark-to-query retrieval y self-attn
    (elimina nn.MultiheadAttention — ya tenemos el kernel Triton)
  • Se integra con el AsyncLightBus de advanced_chimera.py
    (elimina el bus duplicado del archivo legacy)
  • Archiva por umbral de error TTT en vez de intervalo de tokens ciego
    (archiva cuando hay contenido complejo, no temporalmente)

Overhead: ~2-5% del SSD scan (solo cuando hay archivado/retrieval activo).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sgr_slr import diff_attn_v2_triton


class NativeLandmarkArchive(nn.Module):
    """
    Landmark Archive nativo al ecosistema CHIMERA.

    En vez de almacenar el estado SSM (legacy cheby_state), almacena
    embeddings comprimidos de los tokens más importantes del scan output.
    El error TTT actúa como proxy de complejidad — sin MLPs extra.

    Pipeline por forward call:
      1. maybe_archive(scan_out, ttt_importance, tier_probs)
         → si complejidad alta: comprime top-K tokens → nuevo landmark
      2. retrieve(query, device)
         → diff_attn_v2 (Triton) entre query y landmarks acumulados
         → salida: [B, d_model] que se inyecta en el residual stream
    """

    def __init__(self, d_model: int, landmark_dim: int = 128,
                 max_landmarks: int = 64, ttt_err_threshold: float = 0.3):
        super().__init__()
        self.d_model           = d_model
        self.landmark_dim      = landmark_dim
        self.max_landmarks     = max_landmarks
        self.ttt_err_threshold = ttt_err_threshold

        # Compresión del scan output → embedding de landmark
        # Un solo Linear (vs 2-layer MLP del legacy) — usa el mismo d_model
        self.compress = nn.Linear(d_model, landmark_dim, bias=False)

        # Self-atención entre landmarks antes de retrieval — ramas Q/K para diff-attn
        # Usamos landmark_dim como d_head para el kernel Triton
        self.lm_q1 = nn.Linear(landmark_dim, landmark_dim, bias=False)
        self.lm_q2 = nn.Linear(landmark_dim, landmark_dim, bias=False)
        self.lm_k1 = nn.Linear(landmark_dim, landmark_dim, bias=False)
        self.lm_k2 = nn.Linear(landmark_dim, landmark_dim, bias=False)
        self.lm_v  = nn.Linear(landmark_dim, landmark_dim, bias=False)
        self.lm_norm = nn.RMSNorm(landmark_dim)

        # λ para self-atención entre landmarks
        self.lm_lam = nn.Parameter(torch.tensor(-2.0))

        # Retrieval: query desde scan_out actual → landmarks
        self.ret_q1 = nn.Linear(d_model,      landmark_dim, bias=False)
        self.ret_q2 = nn.Linear(d_model,      landmark_dim, bias=False)
        self.ret_k1 = nn.Linear(landmark_dim, landmark_dim, bias=False)
        self.ret_k2 = nn.Linear(landmark_dim, landmark_dim, bias=False)
        self.ret_v  = nn.Linear(landmark_dim, landmark_dim, bias=False)
        self.ret_norm = nn.RMSNorm(landmark_dim)

        # λ para retrieval
        self.ret_lam = nn.Parameter(torch.tensor(-2.0))

        # Proyección del resultado del retrieval de vuelta al d_model
        self.expand = nn.Linear(landmark_dim, d_model, bias=False)

        # Gate de compresión diferenciable (garantiza gradiente a compress.weight).
        # En retrieve(), siempre proyectamos la query más reciente a través
        # de compress, dando un path de gradiente aunque el archive esté vacío.
        # Inicializado en -6 → sigmoid(-6) ≈ 0.0025 → contribución ~0 en warm-up.
        self.compress_query_gate = nn.Parameter(torch.tensor(-6.0))

        # Gate de inyección al residual (inicializado a 0 → warm-up gradual)
        self.inject_gate = nn.Parameter(torch.zeros(1))

        # Buffers del archivo — se mueven con model.to(device), quedan en GPU durante entrenamiento
        self.register_buffer('archived_embeddings',
            torch.zeros(max_landmarks, landmark_dim))
        self.register_buffer('archived_importance',
            torch.zeros(max_landmarks))
        self.register_buffer('n_archived',
            torch.tensor(0, dtype=torch.long))

        # ── Threshold adaptativo EMA ──────────────────────────────────────────
        # Los escalares EMA se registran como buffers (no parámetros) para que:
        #   (a) persistan en state_dict y sobrevivan checkpoint/restart,
        #   (b) se muevan automáticamente a GPU con .to(device),
        #   (c) se serialicen y carguen correctamente en DDP.
        # _ema_alpha es un hiperparámetro fijo — no necesita persistir.
        self.register_buffer('_err_ema_mean', torch.tensor(0.0))
        self.register_buffer('_err_ema_var',  torch.tensor(0.1))
        self._ema_alpha = 0.05    # factor de actualización (ventana ~20 pasos)

    # ─────────────────────────────────────────────────────────────────────────
    # ARCHIVADO
    # ─────────────────────────────────────────────────────────────────────────

    def maybe_archive(self,
                      scan_out:        torch.Tensor,    # [B, S, D]
                      ttt_importance:  torch.Tensor,    # [B, S] — error TTT
                      tier_probs:      torch.Tensor,    # [B, 3] — FAST/HYB/FULL
                      sgr_indices:     torch.Tensor,    # [B, K] — top-K de SGR
                      ) -> bool:
        """
        Decide si archivar y, si es el caso, crea un nuevo landmark.
        Retorna True si se archivó algo.

        OPTIMIZACIÓN PERF-3: el EMA adaptativo ahora usa operaciones in-place
        sobre buffers GPU (add_, mul_, etc.) sin .item(). La decisión de
        archivar o no requiere UN solo .item() (reducción escalar del resultado
        de la comparación tensorial). En el path que SÍ archiva se necesitan
        2 .item() adicionales para importance_score. Total: 1 sync vs 6-8 antes.
        """
        B, S, D = scan_out.shape

        # ── Threshold adaptativo EMA — todo en GPU, sin CPU sync ─────────────
        if ttt_importance is not None:
            mean_err_t = ttt_importance.mean()          # tensor escalar — sin sync
        else:
            mean_err_t = self._err_ema_mean             # proxy: usar la EMA actual
        if ttt_importance is None:
            return False   # sin señal TTT no hay criterio de importancia
        full_prob_t = tier_probs[:, 2].mean()           # tensor escalar — sin sync

        # Actualizar EMA con operaciones in-place — sin .item(), sin CPU sync
        if ttt_importance is not None:
            a     = self._ema_alpha
            delta = mean_err_t.detach() - self._err_ema_mean   # tensor delta
            self._err_ema_mean.add_(delta, alpha=a)             # EMA media, in-place
            self._err_ema_var.mul_(1.0 - a).add_(               # EMA varianza, in-place
                delta.detach().square_().mul_(a)
            )

        # Umbral adaptativo como tensor GPU (sin sync)
        adaptive_thr = (
            self._err_ema_mean +
            0.5 * self._err_ema_var.clamp(min=1e-8).sqrt()
        )
        # clamp: fallback al umbral fijo si el EMA aún no ha convergido
        threshold_t = adaptive_thr.clamp(min=self.ttt_err_threshold)

        # ÚNICO CPU sync del hot path: decidir si archivar
        # Comparación tensorial → 1 .item() para el branch Python
        should_skip = (mean_err_t < threshold_t) & (full_prob_t < 0.5)
        if should_skip.item():
            return False

        # Obtener escalares Python para importance_score (2 syncs solo en archive)
        mean_err  = mean_err_t.item()
        full_prob = full_prob_t.item()

        # ── Compresión batch-aware ─────────────────────────────────────────────
        # Promedia la contribución de TODOS los elementos del batch.
        # Esto da un landmark más representativo que usar solo batch[0].
        with torch.no_grad():
            batch_embeddings = []
            for b_idx in range(B):
                idx      = sgr_indices[b_idx]               # [K]
                key_toks = scan_out[b_idx, idx].float()     # [K, D]
                imp_w    = ttt_importance[b_idx, idx]       # [K]
                imp_w    = torch.softmax(imp_w * 5.0, dim=0).unsqueeze(-1)  # [K,1]
                batch_embeddings.append((key_toks * imp_w).sum(dim=0))      # [D]

            # Media a través del batch
            landmark_raw = torch.stack(batch_embeddings).mean(dim=0)  # [D]
            # Cast al dtype del modelo (BF16 o FP32) antes de compress
            landmark_raw = landmark_raw.to(self.compress.weight.dtype)
            landmark_emb = self.compress(landmark_raw)                # [landmark_dim]

            importance_score = mean_err * full_prob + 1e-6

        self._store_landmark(landmark_emb.detach(), importance_score)
        return True

    def _store_landmark(self, emb: torch.Tensor, importance: float):
        """Almacena un nuevo landmark, ejecuta GC semántico si está lleno."""
        n = self.n_archived.item()
        if n >= self.max_landmarks:
            self._semantic_gc()
            n = self.n_archived.item()

        self.archived_embeddings[n] = emb.to(self.archived_embeddings.device)
        self.archived_importance[n] = importance
        self.n_archived += 1

    def _semantic_gc(self):
        """
        Recolector de Basura Semántico para landmarks.

        Estrategia: eliminar la información más REDUNDANTE semánticamente.

        Algoritmo O(N²) sobre N ≤ 64 (trivial en CPU):
          1. Calcular la matriz de similitud coseno entre todos los landmarks.
          2. Encontrar el par (i, j) más similar (excluyendo la diagonal).
          3. Fusionar ese par en un solo landmark ponderado por importancia:
               emb_merged = (imp_i*emb_i + imp_j*emb_j) / (imp_i + imp_j)
               imp_merged = max(imp_i, imp_j)  # cobertura semántica máxima
          4. Eliminar la posición j (menor importancia), mantener i.

        Por qué coseno y no pares ciegos:
          El merge ciego asume que landmarks ADYACENTES son similares, lo cual
          es falso: tokens complejos surgen en cualquier posición de la secuencia.
          La similitud coseno captura redundancia semántica real independientemente
          del orden de archivado.

        Ejecución híbrida:
          · La parte pesada (cosine sim [N,D]×[D,N]) corre en GPU.
          · El merge y el shift usan escalares Python (más rápido que GPU para N≤64).
          GC ocurre solo cuando el buffer se llena (cada max_landmarks archivados).
        """
        n = self.n_archived.item()
        if n < 2:
            return

        embs = self.archived_embeddings[:n].float()   # [N, D] en GPU
        imps = self.archived_importance[:n].float()   # [N]   en GPU

        # Similitud coseno en GPU: S[i,j] = emb_i·emb_j / (||emb_i|| ||emb_j||)
        embs_norm = F.normalize(embs, dim=-1)          # [N, D]
        sim = embs_norm @ embs_norm.T                  # [N, N]
        sim.fill_diagonal_(-2.0)                       # excluir auto-similitud

        # Par más similar = más redundante → candidato a merge
        flat_idx = int(torch.argmax(sim).item())
        i, j = flat_idx // n, flat_idx % n
        if i > j:
            i, j = j, i    # asegurar i < j para facilitar el shift

        # Fusionar los dos embeddings ponderados por importancia
        imp_i  = float(imps[i].item())
        imp_j  = float(imps[j].item())
        total  = imp_i + imp_j + 1e-8
        merged_emb = (imp_i * embs[i] + imp_j * embs[j]) / total
        merged_imp = max(imp_i, imp_j)   # cobertura máxima

        # Escribir resultado en slots i y j desplazando el segmento superior
        self.archived_embeddings[i] = merged_emb.to(self.archived_embeddings.dtype)
        self.archived_importance[i] = merged_imp
        # Shift izquierda para eliminar j
        if j < n - 1:
            self.archived_embeddings[j:n-1] = self.archived_embeddings[j+1:n].clone()
            self.archived_importance[j:n-1] = self.archived_importance[j+1:n].clone()
        self.archived_embeddings[n-1].zero_()
        self.archived_importance[n-1] = 0.0
        self.n_archived.fill_(n - 1)

    # Alias de compatibilidad hacia atrás
    def _importance_based_merge(self):
        """Deprecated: delegado a _semantic_gc()."""
        self._semantic_gc()

    def preload_context(self, context_embs: torch.Tensor,
                        importance_scores=None):
        """
        Pre-carga landmarks para el 'cold start problem'.

        Permite inyectar contexto externo (embeddings de sesión anterior,
        prompt de sistema, o documento de referencia) ANTES de que el modelo
        procese cualquier token nuevo. El archivo empieza  caliente en vez de
        vacío.

        Args:
            context_embs:     [K, d_model] — embeddings crudos (se comprimen)
                               OR [K, landmark_dim] — ya en espacio de landmark
            importance_scores: [K] o None (default: uniforme 0.5)
        """
        K, dim = context_embs.shape
        if importance_scores is None:
            importance_scores = [0.5] * K

        with torch.no_grad():
            for k in range(K):
                emb_raw = context_embs[k].cpu().float()
                if dim == self.d_model:
                    dev  = self.compress.weight.device
                    dtyp = self.compress.weight.dtype
                    emb_comp = self.compress(emb_raw.to(dev).to(dtyp)).cpu()
                elif dim == self.landmark_dim:
                    emb_comp = emb_raw
                else:
                    raise ValueError(
                        f"preload_context: dim={dim} debe ser d_model={self.d_model} "
                        f"o landmark_dim={self.landmark_dim}"
                    )
                imp = float(importance_scores[k])
                self._store_landmark(emb_comp, imp)

    # ─────────────────────────────────────────────────────────────────────────
    # RETRIEVAL
    # ─────────────────────────────────────────────────────────────────────────

    def _get_processed_landmarks(self, device):
        """
        Retorna landmarks procesados con self-atención diff-attn-V2.

        Ejecutado en torch.no_grad() + detach() para garantizar invariante de
        gradient-checkpoint: forward y recompute producen exactamente 0 saves
        desde esta función (los buffers de landmarks no tienen grad de todas
        formas). Esto resuelve el CheckpointError 171-vs-145 y también evita
        el bug BF16-backward de diff_attn_v2_triton con entradas detachadas.

        lm_q1/lm_q2/lm_k1/lm_k2/lm_v/lm_norm no reciben gradiente desde
        esta ruta (equivalente al comportamiento previo con combined.detach()
        en el caché); el gradiente fluye a esos parámetros únicamente si en
        el futuro se añade una ruta de actualización explícita.
        """
        with torch.no_grad():
            n = self.n_archived.item()
            if n == 0:
                return None

            lm = self.archived_embeddings[:n].to(device)  # [n, landmark_dim]
            # Cast al dtype del modelo (BF16 o FP32) — los buffers se registran
            # siempre en float32 pero las Linear layers siguen model.dtype.
            lm = lm.to(self.lm_q1.weight.dtype)

            if n >= 2:
                # self-atención entre landmarks con diff-attn V2 Triton
                lam = torch.sigmoid(self.lm_lam).item()
                Q1 = self.lm_q1(lm); Q2 = self.lm_q2(lm)
                K1 = self.lm_k1(lm); K2 = self.lm_k2(lm)
                V  = self.lm_v(lm)
                try:
                    sa_out = diff_attn_v2_triton(Q1, Q2, K1, K2, V, lam)  # [n, lm_dim]
                except Exception:
                    # Fallback PyTorch cuando Triton overflows SMEM (n >= 16, d=128)
                    import math as _math
                    scale  = 1.0 / _math.sqrt(Q1.shape[-1])
                    s1 = (Q1.float() @ K1.float().T) * scale
                    s2 = (Q2.float() @ K2.float().T) * scale
                    a1 = torch.softmax(s1, dim=-1); a2 = torch.softmax(s2, dim=-1)
                    sa_out = (a1 @ V.float() - lam * (a2 @ V.float())).to(Q1.dtype)
                sa_out = self.lm_norm(sa_out.to(lm.dtype))
                lm = lm + sa_out

            # Ponderar por importancia (gate suave)
            imp_w = torch.softmax(
                self.archived_importance[:n].to(device) * 5.0, dim=0
            ).unsqueeze(-1)                                   # [n, 1]
            global_summary = (lm * imp_w).sum(dim=0, keepdim=True)  # [1, lm_dim]

            # Combinar: landmarks individiuales + resumen global
            k       = min(n, 12)
            recent  = lm[-k:]                              # [k, lm_dim]
            combined = torch.cat([recent, global_summary], dim=0)  # [k+1, lm_dim]
            combined = combined.to(self.lm_q1.weight.dtype)  # force model dtype (autocast softmax→fp32 contaminates)

        return combined.detach()

    def get_compress_ctx(self, scan_out: torch.Tensor) -> torch.Tensor:
        """
        Devuelve ÚNICAMENTE la contribución compress→expand, SIEMPRE con gradiente.

        Esta función debe llamarse FUERA de cualquier arch_gate para garantizar
        que compress.weight y expand.weight reciben gradiente en TODOS los pasos,
        independientemente de si el archivo tiene landmarks o de la complejidad del batch.

        La razón por la que esto es necesario:
          Si se llama retrieve() y su resultado se multiplica por arch_gate=0
          (cuando el batch no supera el umbral de complejidad), compress_ctx
          se zeroes junto con el resto → grad_compress=0. Este método rompe
          esa dependencia manteniendo compress siempre activo.
        """
        B, S, D = scan_out.shape
        q_repr     = scan_out.mean(dim=1)                             # [B, D]
        compress_q = self.compress(q_repr)                            # [B, lm_dim]
        cq_gate    = torch.sigmoid(self.compress_query_gate)          # escalar
        return (self.expand(compress_q) * cq_gate                     # [B, D]
                ).unsqueeze(1).expand(B, S, D).contiguous()          # [B, S, D]

    def retrieve(self, scan_out: torch.Tensor) -> torch.Tensor:
        """
        Recupera SOLO la información de los landmarks archivados.

        CAMBIO v2: compress_ctx ya NO se incluye aquí. Debe obtenerse
        mediante get_compress_ctx() y añadirse FUERA del arch_gate en el
        caller (advanced_chimera.py forward). Esto garantiza el invariante
        de gradiente aunque arch_gate=0.

        scan_out: [B, S, D]
        Returns:  [B, S, D] — scan_out + contexto de landmarks (sin compress_ctx)
        """
        B, S, D = scan_out.shape
        device  = scan_out.device

        lm = self._get_processed_landmarks(device)
        if lm is None:
            # Archivo vacío: devolver scan_out sin modificar.
            # compress_ctx se añade externamente por el caller.
            return scan_out

        n_lm = lm.shape[0]          # k+1 landmarks procesados
        gate = torch.sigmoid(self.inject_gate)
        lam  = torch.sigmoid(self.ret_lam).item()
        out  = scan_out.clone()

        # ── Query repr (sin gradiente a compress — eso lo gestiona get_compress_ctx) ──
        q_repr = scan_out.mean(dim=1).detach()    # [B, D] — detach: compress no pasa aquí

        # ── Retrieval vectorizado (sin loop sobre B) ───────────────────────────
        Q1_all = self.ret_q1(q_repr)           # [B, lm_dim]
        Q2_all = self.ret_q2(q_repr)

        K1 = self.ret_k1(lm)                   # [n_lm, lm_dim]
        K2 = self.ret_k2(lm)
        V  = self.ret_v(lm)                    # [n_lm, lm_dim]

        if B >= 16:
            ret_emb = diff_attn_v2_triton(Q1_all, Q2_all, K1, K2, V, lam)  # [B, lm_dim]
        else:
            ret_list = []
            for b in range(B):
                r = diff_attn_v2_triton(
                    Q1_all[b:b+1], Q2_all[b:b+1], K1, K2, V, lam
                )  # [1, lm_dim]
                ret_list.append(r)
            ret_emb = torch.cat(ret_list, dim=0)   # [B, lm_dim]

        ret_emb = self.ret_norm(ret_emb.to(scan_out.dtype))   # [B, lm_dim]

        # Proyecto a d_model y broadcast a [B, S, D]
        ctx = self.expand(ret_emb).unsqueeze(1).expand(B, S, D)   # [B, S, D]
        out = out + gate * ctx

        return out

    # ─────────────────────────────────────────────────────────────────────────
    # INFO
    # ─────────────────────────────────────────────────────────────────────────

    def get_archive_info(self):
        n = self.n_archived.item()
        return {
            'n_landmarks':      n,
            'max_landmarks':    self.max_landmarks,
            'landmark_dim':     self.landmark_dim,
            'cache_valid':      True,  # cache removed; always recomputes
            'memory_kb':        n * self.landmark_dim * 4 / 1024,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    B, S, D = 2, 512, 256
    print("=== NativeLandmarkArchive test ===")

    archive = NativeLandmarkArchive(d_model=D, landmark_dim=128).cuda()

    scan_out = torch.randn(B, S, D, device="cuda")
    imp      = torch.rand(B, S, device="cuda") * 0.8   # error TTT simulado
    imp[0, 50:100] *= 3.0                              # zona compleja
    tier_probs = torch.tensor([[0.1, 0.3, 0.6]]).expand(B, -1).cuda()
    # SGR indices simulados
    K = max(1, int(0.125 * S))
    sgr_idx = torch.topk(imp, K, dim=-1).indices

    t0 = time.time()

    # Simular 3 archivados consecutivos
    for i in range(3):
        archived = archive.maybe_archive(scan_out, imp, tier_probs, sgr_idx)
        print(f"  Archive call {i+1}: {'ARCHIVADO' if archived else 'skipped'}")

    print(f"  Info: {archive.get_archive_info()}")

    # Retrieval
    enriched = archive.retrieve(scan_out)
    torch.cuda.synchronize()
    t1 = time.time()

    print(f"[OK] scan_out: {scan_out.shape} → enriched: {enriched.shape}")
    print(f"[OK] n_landmarks: {archive.n_archived.item()}")
    print(f"[OK] Tiempo total: {(t1-t0)*1000:.2f} ms")
    print("[SUCCESS] NativeLandmarkArchive con diff-attn Triton V2 OK")
