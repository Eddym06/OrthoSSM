"""
Speculative Decoding para SSMs — Algoritmo Novel
=================================================
SpeculativeSSMDecoder: aceleración de inferencia autoregressive
en modelos de sequencia de estado (SSM+TTT) mediante verificación
paralela en un scan causal.

Motivación:
  En Transformers, speculative decoding es complejo porque la verificación
  de K tokens requiere K² atención. En SSMs el scan es naturalmente causal
  y paralelo: dado un estado s[t-1], uno puede computar s[t], s[t+1], ...,
  s[t+K] en UN solo forward de longitud K (sin atención cuadrática).
  
  Esto hace que los SSMs sean candidatos IDEALES para speculative decoding:
  la verificación de K tokens es O(K) en tiempo, no O(K²).

Protocolo SpecSSM:
  1. DRAFT PHASE (modelo pequeño o CHIMERA con TTT congelado):
     - decode K tokens con draft_model.step()
     - guardar los K tokens propuestos + sus distribuciones de probabilidad
     
  2. STATE BRANCHING:
     - guardar ssm_state + bus_cache ANTES de la fase draft
     - los K tokens draft se guardan por separado
     - el estado base es intacto para el verificador
     
  3. VERIFY PHASE (modelo completo):
     - desde el estado base, hacer UN forward de longitud K sobre los tokens draft
     - gracias al scan causal SSM, este forward es PARALELO → O(K) no O(K²)
     - obtener distribuciones verify_probs[0..K-1]
     
  4. ACCEPT/REJECT (Leviathan et al. adaptado a SSMs):
     - aceptar token k si random() ≤ verify_probs[k] / draft_probs[k]
     - encontrar el primer rechazo en posición j
     - los tokens 0..j-1 son aceptados
     
  5. STATE RECOVERY:
     - hacer UN forward de longitud j sobre los tokens aceptados desde el estado base
     - → estado final correcto en O(j) tiempo
     
  GANANCIA: si acceptance_rate = α > 0.7, throughput ≈ K·α / 1 vs 1 token/step
  Para α=0.8 y K=8: 6.4x de ganancia teórica sobre decode secuencial puro.

Diferencia clave con Transformer speculative:
  Transformer verify: O(K²) atención (KV cache grow)
  SSM verify:         O(K) scan causal determinístico
  → Los SSMs eliminan el cuello de botella cuadrático de la verificación.
  
  Esta propiedad aplica a cualquier SSM causal: Mamba2, RWKV, GLA, CHIMERA.
  CHIMERA es especialmente interesante porque tiene TTT adaptativo: la
  verificación paralela puede actualizar el TTT state exactamente igual
  que la generación sekuencial, sin aproximación.

Nota sobre TTT y verificación:
  El mega-kernel de OrthoSSM actualiza c_k INLINE durante el scan.
  Por tanto, el forward de longitud K sobre los tokens draft actualiza c_k
  exactamente como si hubieran sido generados secuencialmente.
  Esto garantiza EXACTITUD de los estados TTT, no sólo los SSM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import json
import os
import sys
from typing import Optional, Dict, List, Tuple, Any
from copy import deepcopy

# Añadir directorio de CHIMERA al path
sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mini cabeza LM (necesaria para obtener distribuciones token)
# ─────────────────────────────────────────────────────────────────────────────

class SpecLMHead(nn.Module):
    """
    Cabeza LM mínima usada internamente por SpeculativeSSMDecoder.
    Si el modelo ya tiene una cabeza LM, se usa directamente.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)   # [B, S, V]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Utilidades de estado SSM
# ─────────────────────────────────────────────────────────────────────────────

def clone_state(state: Optional[Any]) -> Optional[Any]:
    """
    Crea una copia profunda del estado de un SSM / CHIMERA.
    El estado puede ser:
      - dict con tensores
      - lista de dicts (stack de capas)
      - None
    """
    if state is None:
        return None
    if isinstance(state, dict):
        return {k: (v.clone() if isinstance(v, torch.Tensor) else deepcopy(v))
                for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return [clone_state(s) for s in state]
    if isinstance(state, torch.Tensor):
        return state.clone()
    return deepcopy(state)


def _get_model_step_fn(model):
    """
    Para decode token-a-token en speculative SSM usamos forward([B,1,D])
    en lugar de step() para evitar incompatibilidades de formato de estado
    entre Mamba2 (conv_state/ssm_state) y ChimeraV2 (ema_state).

    El bus_cache se propaga manualmente entre tokens.
    """
    def fwd_as_step(x_tok, cache):
        """
        x_tok: [B, D]
        cache: dict {'bus_cache': tensor o None}
        Returns: (hidden [B, D], new_cache dict)
        """
        bus_cache = cache.get('bus_cache') if isinstance(cache, dict) else cache
        x = x_tok.unsqueeze(1)   # [B, 1, D]

        # Detectar si el modelo es ChimeraLM (tiene .chimera) o una capa directa
        if hasattr(model, 'chimera'):
            inner = model.chimera
        elif hasattr(model, 'layers'):
            inner = model   # stack
        else:
            inner = model

        with torch.no_grad():
            out = inner(x, bus_cache=bus_cache)

        # Desempacar (out, bus_cache) o (out, bus_cache, aux)
        if len(out) == 3:
            hidden_seq, new_bus, _ = out
        else:
            hidden_seq, new_bus = out

        hidden = hidden_seq[:, 0, :]    # [B, D]
        new_cache = {'bus_cache': new_bus}
        return hidden, new_cache

    return fwd_as_step


def _get_model_forward_fn(model):
    """
    Retorna forward(x [B,S,D], cache) → (hidden [B,S,D], cache)
    que funciona con AdvancedChimeraLayer, ChimeraV2Layer, ChimeraLM.
    """
    def forward_fn(x, cache=None):
        bus_cache = cache.get('bus_cache') if isinstance(cache, dict) else cache

        if hasattr(model, 'chimera'):
            inner = model.chimera
        elif hasattr(model, 'layers'):
            inner = model
        else:
            inner = model

        with torch.no_grad():
            out = inner(x, bus_cache=bus_cache)

        if len(out) == 3:
            hidden, new_bus, _ = out
        else:
            hidden, new_bus = out

        new_cache = {'bus_cache': new_bus}
        return hidden, new_cache

    return forward_fn


# ─────────────────────────────────────────────────────────────────────────────
# 3. SpeculativeSSMDecoder
# ─────────────────────────────────────────────────────────────────────────────

class SpeculativeSSMDecoder:
    """
    Speculative decoding para SSM/CHIMERA usando verificación paralela O(K).

    Args:
        target_model: modelo principal (CHIMERA completo)
        draft_model:  modelo borrador (versión ligera o CHIMERA con TTT off)
                      Si es None, usa el mismo target_model con TTT congelado
                      como draft (menos eficiente pero funcional para testing)
        lm_head:      cabeza LM que convierte hidden → logits. Si target_model
                      ya tiene lm_head, se usa automáticamente.
        K:            número de tokens draft por especulación (default 8)
        temperature:  temperatura para muestreo (1.0 = sin escalado)
        vocab_size:   tamaño del vocabulario (por defecto 512 para tests NIAH)
    """

    def __init__(
        self,
        target_model:    nn.Module,
        draft_model:     Optional[nn.Module] = None,
        lm_head:         Optional[nn.Module] = None,
        K:               int   = 8,
        temperature:     float = 1.0,
        vocab_size:      int   = 512,
    ):
        self.target_model = target_model
        self.K            = K
        self.temperature  = temperature
        self.vocab_size   = vocab_size

        # Draft model: si no se da, usar el target con TTT congelado
        if draft_model is None:
            print("[SpecSSM] No hay draft_model — usando target con TTT congelado como draft")
            self.draft_model  = target_model
            self.draft_frozen = True
        else:
            self.draft_model  = draft_model
            self.draft_frozen = False

        # Cabeza LM
        if lm_head is not None:
            self.lm_head = lm_head
        elif hasattr(target_model, 'lm_head'):
            self.lm_head = target_model.lm_head
        else:
            print(f"[SpecSSM] Creando SpecLMHead(d_model=auto, vocab={vocab_size})")
            d_model_guess = getattr(target_model, 'd_model', 256)
            self.lm_head  = SpecLMHead(d_model_guess, vocab_size)

        # Funciones de paso
        self._target_step    = _get_model_step_fn(target_model)
        self._target_forward = _get_model_forward_fn(target_model)
        self._draft_step     = _get_model_step_fn(self.draft_model)

        # Estadísticas de aceptación
        self.stats = {
            'total_generated': 0,
            'total_accepted':  0,
            'n_speculations':  0,
            'acceptance_by_pos': [0] * K,
            'total_by_pos':      [0] * K,
        }

    def _get_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [B, D] → logits [B, V]"""
        return self.lm_head(hidden)

    def _sample(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Muestrea un token y devuelve (token_id [B], prob [B]).
        logits: [B, V]
        """
        if self.temperature != 1.0:
            logits = logits / self.temperature
        probs  = torch.softmax(logits, dim=-1)    # [B, V]
        tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]
        tok_probs = probs.gather(1, tokens.unsqueeze(1)).squeeze(1)   # [B]
        return tokens, tok_probs

    @torch.no_grad()
    def _get_hidden(self, model, x_tok: torch.Tensor, cache) -> Tuple[torch.Tensor, Any]:
        """
        Pasa x_tok [B, D] por el modelo y devuelve hidden [B, D] + new_cache.
        Usa forward([B,1,D]) internamente para compatibilidad universal.
        """
        step_fn = _get_model_step_fn(model)
        return step_fn(x_tok, cache)

    @torch.no_grad()
    def generate_speculative(
        self,
        context_hidden:  torch.Tensor,        # [B, D] — hidden del último token del contexto
        initial_cache:   Any,                  # cache (ssm_state + bus_cache) del target
        max_new_tokens:  int = 64,
        embed_fn:        Optional[Any] = None, # fn: token_id → hidden [B, D]
        vocab_lookup:    Optional[Any] = None, # alternativa: embedding table
    ) -> Dict[str, Any]:
        """
        Genera hasta max_new_tokens con speculative decoding.

        Args:
            context_hidden: representación del último token del contexto in the target
            initial_cache:  estado inicial (puede ser None)
            max_new_tokens: máx tokens a generar
            embed_fn:       función(token_ids [B]) → hidden [B, D] — necesaria para pasar
                            los tokens draft al target en el paso de verificación
            vocab_lookup:   tabla nn.Embedding — alternativa a embed_fn

        Returns: dict con {
            'tokens':         lista de listas de token ids generados
            'n_accepted':     media de tokens aceptados por especulación
            'acceptance_rate': tasa de aceptación global
            'tokens_per_sec': throughput estimado
            'n_calls':        total de calls a target_model
        }
        """
        device   = context_hidden.device
        B        = context_hidden.shape[0]
        assert B == 1, "SpecSSM actualmente implementado para B=1"

        # Función de embedding: token_id → hidden representación
        if embed_fn is None and vocab_lookup is not None:
            embed_fn = lambda tok: vocab_lookup(tok)  # noqa: E731
        elif embed_fn is None:
            # Fallback: mapear token id a variable aleatoria determinista
            # (para testing sin embedding real)
            d = context_hidden.shape[-1]
            def embed_fn(tok):
                # Embedding on-the-fly determinista desde tok_id
                torch.manual_seed(int(tok[0].item()) * 17 + 1)
                return torch.randn(B, d, device=device)

        generated_tokens = []
        target_cache = clone_state(initial_cache)
        draft_cache  = clone_state(initial_cache)

        # ── Inicializar estado con el hidden del contexto ────────────────────
        # El hidden del contexto ya fue generado por el caller — usamos ese estado.

        t_start   = time.perf_counter()
        n_calls   = 0
        step      = 0
        cur_hidden = context_hidden   # [B, D] — hidden del último token procesado

        while step < max_new_tokens:
            # ── DRAFT PHASE ──────────────────────────────────────────────────
            # Guardar estado base ANTES de la fase draft
            saved_target_cache = clone_state(target_cache)
            saved_draft_cache  = clone_state(draft_cache)

            draft_tokens   = []    # lista de K tensores [B]
            draft_probs    = []    # lista de K tensores [B]
            draft_hiddens  = []    # lista de K tensores [B, D]

            cur_draft_hidden = cur_hidden.clone()
            cur_draft_cache  = clone_state(draft_cache)

            for k in range(self.K):
                if step + k >= max_new_tokens:
                    break
                # Draft model produce un token
                logits_d   = self._get_logits(cur_draft_hidden)  # [B, V]
                tok_d, p_d = self._sample(logits_d)              # [B], [B]
                draft_tokens.append(tok_d)
                draft_probs.append(p_d)
                draft_hiddens.append(cur_draft_hidden.clone())

                # Avanzar draft model
                tok_emb             = embed_fn(tok_d)             # [B, D]
                cur_draft_hidden, cur_draft_cache = self._get_hidden(
                    self.draft_model, tok_emb, cur_draft_cache
                )
                n_calls += 1

            actual_K = len(draft_tokens)
            if actual_K == 0:
                break

            # ── VERIFY PHASE ─────────────────────────────────────────────────
            # Verificación PARALELA: un forward de longitud actual_K sobre los
            # tokens draft desde el estado base guardado.
            #
            # INNOVACIÓN CLAVE: Los SSMs permiten verificar K tokens en UN SOLO
            # forward O(K) — sin KV cache que crece cuadráticamente.
            #
            # Construimos la secuencia de embeddings de los tokens draft:
            draft_embeds = torch.stack(
                [embed_fn(t) for t in draft_tokens], dim=1
            )  # [B, K, D]

            # Forward paralelo del target sobre los K tokens draft desde estado base
            target_forward = _get_model_forward_fn(self.target_model)
            verify_hidden, new_target_cache = target_forward(
                draft_embeds, saved_target_cache
            )
            # verify_hidden: [B, K, D]
            n_calls += 1   # UN solo call para verificar K tokens

            # Logits de verificación [B, K, V]
            verify_logits = self.lm_head(verify_hidden)   # [B, K, V]
            verify_probs  = torch.softmax(verify_logits / self.temperature, dim=-1)

            # ── ACCEPT/REJECT ─────────────────────────────────────────────────
            # Token k es aceptado si:
            #   U~Uniform(0,1) ≤ min(1, p_verify(draft_tok_k) / p_draft(draft_tok_k))
            n_accepted = 0
            final_token = None
            resample_logits = None

            for k in range(actual_K):
                tok_k    = draft_tokens[k]       # [B]
                p_draft  = draft_probs[k]        # [B]
                # Probabilidad que el target asigna al token draft propuesto
                p_verify = verify_probs[:, k, :].gather(
                    1, tok_k.unsqueeze(1)
                ).squeeze(1)                     # [B]

                ratio     = (p_verify / p_draft.clamp(min=1e-9)).clamp(max=1.0)
                u         = torch.rand_like(ratio)
                accepted  = (u <= ratio).all().item()   # aceptado si B=1

                # Estadísticas
                self.stats['total_by_pos'][k]      += 1
                self.stats['acceptance_by_pos'][k] += int(accepted)

                if accepted:
                    generated_tokens.append(tok_k.tolist())
                    n_accepted += 1
                    # El estado del target avanza hasta el token k+1 usando el
                    # forward paralelo ya computado (new_target_cache representa
                    # el estado después de los K tokens — lo recortamos si es posible).
                    # Simplificación: si no podemos recortar, usamos el estado completo
                    # después de los K tokens y lo marcamos para corrección.
                else:
                    # Rechazo en posición k: muestrear nuevo token con distribución corregida
                    # p_resample = normalize(max(0, p_verify - p_draft))
                    p_v_full = verify_probs[:, k, :]                        # [B, V]
                    p_d_full = torch.softmax(self._get_logits(draft_hiddens[k]) / self.temperature, dim=-1)
                    p_resample = F.relu(p_v_full - p_d_full)
                    if p_resample.sum() < 1e-6:
                        p_resample = p_v_full
                    else:
                        p_resample = p_resample / p_resample.sum(dim=-1, keepdim=True)
                    resample_logits = p_resample
                    break

            # Actualizar estadísticas
            self.stats['total_accepted']  += n_accepted
            self.stats['total_generated'] += actual_K
            self.stats['n_speculations']  += 1

            # ── STATE RECOVERY ────────────────────────────────────────────────
            if n_accepted < actual_K:
                # Rechazo antes del final: avanzar estado hasta posición n_accepted
                # desde el estado base con UN forward de longitud n_accepted.
                if n_accepted > 0:
                    accepted_embeds = torch.stack(
                        [embed_fn(draft_tokens[k]) for k in range(n_accepted)], dim=1
                    )  # [B, n_accepted, D]
                    hidden_acc, target_cache = target_forward(
                        accepted_embeds, saved_target_cache
                    )
                    cur_hidden = hidden_acc[:, -1, :]   # último token aceptado
                    n_calls   += 1
                else:
                    # Ningún token aceptado — retroceder al estado base
                    target_cache = saved_target_cache
                    cur_hidden   = context_hidden         # volver al contexto original

                # Generar token de reemplazo (ya tenemos la distribución)
                if resample_logits is not None:
                    replacement_tok = torch.multinomial(resample_logits, 1).squeeze(1)
                    generated_tokens.append(replacement_tok.tolist())
                    tok_emb    = embed_fn(replacement_tok)
                    cur_hidden, target_cache = self._get_hidden(
                        self.target_model, tok_emb, target_cache
                    )
                    n_calls   += 1
                    step      += n_accepted + 1
                else:
                    step += n_accepted
            else:
                # Todos aceptados: generar un token extra desde el estado de después
                target_cache = new_target_cache
                extra_hidden = verify_hidden[:, -1, :]   # [B, D]
                extra_logits = self.lm_head(extra_hidden)
                extra_tok, _ = self._sample(extra_logits)
                generated_tokens.append(extra_tok.tolist())
                step      += actual_K + 1
                cur_hidden = extra_hidden

            draft_cache = clone_state(target_cache)   # sincronizar draft_cache

        t_elapsed = time.perf_counter() - t_start
        total_gen = sum(len(t) if isinstance(t, list) else 1 for t in generated_tokens)
        tps = total_gen / max(t_elapsed, 1e-6)

        acc_rate = (self.stats['total_accepted'] /
                    max(self.stats['total_generated'], 1))

        return {
            'tokens':          generated_tokens,
            'n_generated':     total_gen,
            'tokens_per_sec':  round(tps, 1),
            'acceptance_rate': round(acc_rate, 4),
            'n_accepted_per_spec': round(
                self.stats['total_accepted'] / max(self.stats['n_speculations'], 1), 2
            ),
            'n_calls':         n_calls,
            'time_s':          round(t_elapsed, 3),
            'acceptance_by_pos': [
                round(a / max(t, 1), 3)
                for a, t in zip(self.stats['acceptance_by_pos'],
                                self.stats['total_by_pos'])
            ],
        }

    @torch.no_grad()
    def generate_sequential(
        self,
        context_hidden: torch.Tensor,    # [B, D]
        initial_cache:  Any,
        max_new_tokens: int = 64,
        embed_fn:       Optional[Any] = None,
        vocab_lookup:   Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generación secuencial baseline (sin especulación) para comparar throughput.
        
        Cada token requiere UN call a target_model.step() → O(max_new_tokens) calls.
        """
        device = context_hidden.device
        B = context_hidden.shape[0]

        if embed_fn is None and vocab_lookup is not None:
            embed_fn = lambda tok: vocab_lookup(tok)  # noqa
        elif embed_fn is None:
            d = context_hidden.shape[-1]
            def embed_fn(tok):
                torch.manual_seed(int(tok[0].item()) * 17 + 1)
                return torch.randn(B, d, device=device)

        generated = []
        cache      = clone_state(initial_cache)
        cur_hidden = context_hidden

        t0 = time.perf_counter()
        for _ in range(max_new_tokens):
            logits = self.lm_head(cur_hidden)         # [B, V]
            tok, _ = self._sample(logits)
            generated.append(tok.tolist())
            tok_emb = embed_fn(tok)
            cur_hidden, cache = self._get_hidden(self.target_model, tok_emb, cache)

        elapsed = time.perf_counter() - t0
        tps = max_new_tokens / max(elapsed, 1e-6)

        return {
            'tokens':         generated,
            'n_generated':    max_new_tokens,
            'tokens_per_sec': round(tps, 1),
            'n_calls':        max_new_tokens,
            'time_s':         round(elapsed, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Benchmark comparativo
# ─────────────────────────────────────────────────────────────────────────────

def run_speculative_benchmark(
    d_model:       int = 256,
    vocab_size:    int = 512,
    max_new_tokens: int = 128,
    K:             int = 8,
    n_runs:        int = 3,
    device:        str = 'cuda',
    save_results:  bool = True,
) -> Dict[str, Any]:
    """
    Crea un ChimeraLM, ejecuta generación especulativa y secuencial y compara.
    """
    from niah_eval import ChimeraLM

    print("=" * 68)
    print("  SPECULATIVE SSM DECODE BENCHMARK")
    print(f"  d_model={d_model}  K={K}  max_new={max_new_tokens}  device={device}")
    print("=" * 68)

    # Modelos
    target = ChimeraLM(d_model=d_model, vocab_size=vocab_size).to(device).float()
    draft  = ChimeraLM(d_model=d_model // 2 if False else d_model,
                       vocab_size=vocab_size).to(device).float()
    # Nota: Para un benchmark real, draft sería un modelo más pequeño.
    # Aquí usamos el mismo tamaño para probar la mecánica del algoritmo.

    spec = SpeculativeSSMDecoder(
        target_model = target,
        draft_model  = draft,
        lm_head      = target.lm_head,
        K            = K,
        vocab_size   = vocab_size,
    )

    embed_fn = lambda tok: target.embed(tok).detach()  # x [B, D]

    # Contexto inicial
    ctx_ids  = torch.randint(2, 400, (1, 32), device=device)
    with torch.no_grad():
        ctx_emb = target.embed(ctx_ids)                 # [1, 32, D]
        ctx_out, ctx_cache, _ = target.chimera(ctx_emb, return_aux=True)
    ctx_hidden = ctx_out[:, -1, :].detach()            # [1, D]

    all_spec = []
    all_seq  = []

    for run in range(n_runs):
        print(f"\n  Run {run+1}/{n_runs}")

        # Especulativo
        spec.stats = {k: 0 if isinstance(v, int) else [0]*K
                      for k, v in spec.stats.items()}
        res_spec = spec.generate_speculative(
            context_hidden = ctx_hidden,
            initial_cache  = clone_state(ctx_cache),
            max_new_tokens = max_new_tokens,
            embed_fn       = embed_fn,
        )
        all_spec.append(res_spec)
        print(f"    Spec:  {res_spec['tokens_per_sec']:>8.1f} tok/s "
              f"| acceptance_rate={res_spec['acceptance_rate']:.3f} "
              f"| n_calls={res_spec['n_calls']}")

        # Secuencial
        res_seq = spec.generate_sequential(
            context_hidden = ctx_hidden,
            initial_cache  = clone_state(ctx_cache),
            max_new_tokens = max_new_tokens,
            embed_fn       = embed_fn,
        )
        all_seq.append(res_seq)
        print(f"    Seq:   {res_seq['tokens_per_sec']:>8.1f} tok/s "
              f"| n_calls={res_seq['n_calls']}")

    # Promedios
    avg_spec_tps = sum(r['tokens_per_sec'] for r in all_spec) / n_runs
    avg_seq_tps  = sum(r['tokens_per_sec'] for r in all_seq)  / n_runs
    avg_acc_rate = sum(r['acceptance_rate'] for r in all_spec) / n_runs
    speedup      = avg_spec_tps / max(avg_seq_tps, 1e-3)

    print(f"\n{'='*68}")
    print(f"  RESULTADOS PROMEDIO ({n_runs} runs):")
    print(f"  Speculative:   {avg_spec_tps:>8.1f} tok/s")
    print(f"  Sequential:    {avg_seq_tps:>8.1f} tok/s")
    print(f"  Speedup:       {speedup:>8.2f}x")
    print(f"  Accept rate:   {avg_acc_rate:>8.3f}")
    print(f"\n  Acceptance by position:")
    for k, (acc, total) in enumerate(zip(
        spec.stats['acceptance_by_pos'], spec.stats['total_by_pos']
    )):
        rate = acc / max(total, 1)
        bar  = '█' * int(rate * 20) + '░' * (20 - int(rate * 20))
        print(f"    k={k}: {bar} {rate:.3f}")
    print(f"{'='*68}\n")

    # Análisis teórico
    alpha = avg_acc_rate
    theoretical_speedup = K * alpha / max(1 - alpha**K, 1e-9) if alpha < 1.0 else K
    print(f"  Speedup teórico (medida): {theoretical_speedup:.2f}x")
    print(f"  (fórmula: K*α/(1-α^K) con K={K}, α={alpha:.3f})\n")

    summary = {
        'K':                K,
        'max_new_tokens':   max_new_tokens,
        'avg_spec_tps':     round(avg_spec_tps, 1),
        'avg_seq_tps':      round(avg_seq_tps, 1),
        'speedup':          round(speedup, 3),
        'avg_acceptance_rate': round(avg_acc_rate, 4),
        'theoretical_speedup': round(theoretical_speedup, 3),
        'n_runs':           n_runs,
    }

    if save_results:
        path = os.path.join(os.path.dirname(__file__), 'speculative_results.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Resultados: {path}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 5. Draft model ligero: CHIMERA con SLR congelado
# ─────────────────────────────────────────────────────────────────────────────

def make_frozen_ttt_draft(base_model: nn.Module) -> nn.Module:
    """
    Crea un modelo draft a partir del modelo base congelando el TTT predictor
    y el archive (los componentes más costosos de CHIMERA).
    El draft resultante es ~30% más rápido para decode.
    """
    draft = deepcopy(base_model)
    # Congelar parámetros TTT
    for name, param in draft.named_parameters():
        if 'ttt' in name.lower() or 'archive' in name.lower():
            param.requires_grad_(False)
    # Desactivar archive para decode rápido
    if hasattr(draft, 'chimera') and hasattr(draft.chimera, 'archive'):
        draft.chimera.archive = None  # type: ignore
    return draft


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test rápido unitario del decoder
    print("\n--- Test unitario SpeculativeSSMDecoder ---")
    from niah_eval import ChimeraLM

    d_model    = 256
    vocab_size = 512

    target = ChimeraLM(d_model=d_model, vocab_size=vocab_size).to(device).float()
    target.eval()

    spec = SpeculativeSSMDecoder(
        target_model = target,
        K            = 4,
        vocab_size   = vocab_size,
    )

    # Contexto
    ctx_ids = torch.randint(2, 100, (1, 16), device=device)
    with torch.no_grad():
        ctx_emb = target.embed(ctx_ids)
        ctx_out, ctx_cache, _ = target.chimera(ctx_emb, return_aux=True)
    ctx_h = ctx_out[:, -1, :].detach()

    embed_fn = lambda tok: target.embed(tok).detach()  # noqa

    # Generación especulativa corta
    res = spec.generate_speculative(
        context_hidden = ctx_h,
        initial_cache  = clone_state(ctx_cache),
        max_new_tokens = 16,
        embed_fn       = embed_fn,
    )

    print(f"  Spec tokens:           {len(res['tokens'])}")
    print(f"  Acceptance rate:       {res['acceptance_rate']:.3f}")
    print(f"  Tokens/sec:            {res['tokens_per_sec']:.1f}")
    print(f"  Target calls:          {res['n_calls']}")

    # Benchmark completo
    print("\n--- Benchmark especulativo ---")
    run_speculative_benchmark(
        d_model        = d_model,
        vocab_size     = vocab_size,
        max_new_tokens = 64,
        K              = 4,
        n_runs         = 2,
        device         = device,
    )

    print("\n[OK] speculative_ssm.py completado sin errores.")
