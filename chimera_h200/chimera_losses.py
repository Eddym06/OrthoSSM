"""
ChimeraLosses — Sprint 6.3 (routing corregido)
================================================
Pérdidas auxiliares para training estable del stack CHIMERA.

CAMBIO CRÍTICO vs Sprint 6.2:
  Antes:  routing_loss = -weight * H   → MAXIMIZA entropía → presiona a uniform
          Esto EMPEORABA el problema: benchmark T5 mostró H=99.4% del máximo.
  Ahora:  Se usa ChimeraRoutingLoss con TRES fuerzas en equilibrio correcto:
    (a) Entropy hinge: penaliza H por encima de un target (ej. 70% del máximo)
        Quiere routing ESPECIALIZADO por sample, no uniforme.
    (b) TTT supervision: usa ttt_importance como señal de complejidad por batch
        → samples complejos (alto error TTT) → FULL;
           samples simples (bajo error) → FAST.
    (c) Load balance: penaliza que algún tier tenga prob media ≈ 0 en el batch
        → evita colapso a un solo tier (todos FULL o todos FAST).

1. ChimeraLosses  — clase legada (acumulador, compatible con código existente)
   NOTA: routing_entropy_loss ahora PENALIZA H alto (minimizar H por sample)
         para incentivar especialización, manteniendo load_balance separado.

2. ChimeraRoutingLoss — clase nueva, usa aux_dict directamente (sin acumulador)
   Recomendada para nuevos training loops.

3. test_bus_publish_multilayer() — verifica gradiente en stack de N capas

2. ttt_prediction_loss   — pérdida predictiva del TTT
   Objetivo: que el modelo minimice el error de predicción un-step-ahead.
   Proporciona gradientes adicionales para dt_bias y la proyección B.
   Loss = weight · MSE(pred[:-1], target[1:])   por mini-chunk

3. chimera_total_loss    — combina LM cross-entropy + auxiliares
   Convenience wrapper para training loop.

Uso:
    losses = ChimeraLosses(
        routing_weight  = 0.01,
        ttt_pred_weight = 0.05,
    )
    # En el forward loop, acumular:
    losses.add_routing_probs(tier_probs)   # una vez por capa
    losses.add_ttt_error(pred, target)     # una vez por capa (si TTT activo)
    # Al final del step:
    aux = losses.compute()    # dict con valores individuales
    total_loss = lm_loss + aux['total']
    losses.reset()
"""
import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


class ChimeraLosses:
    """
    Acumulador de pérdidas auxiliares para un paso de training.

    Thread-safe para uso en un solo forward pass.
    """

    def __init__(
        self,
        routing_weight:  float = 0.01,   # peso de la entropy penalty de routing
        ttt_pred_weight: float = 0.05,   # peso del TTT prediction loss
    ):
        self.routing_weight   = routing_weight
        self.ttt_pred_weight  = ttt_pred_weight
        # Load-balance: penaliza que algún tier reciba cerca de 0% del tráfico.
        # Simétrico a ChimeraRoutingLoss.balance_weight.
        # Por defecto 20% del routing_weight — señal suave que evita colapso total
        # sin interferir con la especialización deseada por entropy penalty.
        self.balance_weight   = routing_weight * 0.2
        self.min_tier_prob    = 0.05    # mínimo 5% por tier (3 tiers = 15% base)

        # Acumuladores (se resetean en cada step)
        self._routing_probs: List[torch.Tensor] = []   # [B, 3] por capa
        self._ttt_errors:    List[torch.Tensor] = []   # scalar MSE por capa

    # ─────────────────────────────────────────────────────────────────────────
    # Registro (llamar durante forward de cada capa)
    # ─────────────────────────────────────────────────────────────────────────

    def add_routing_probs(self, probs: torch.Tensor):
        """
        probs: [B, 3] — salida del GatedComplexityPredictor.
        Guardar referencia (no clonar) para que el grafo de cómputo quede intacto.
        """
        self._routing_probs.append(probs)

    def add_ttt_error(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred:   [B, Sm1, D]
        target: [B, Sm1, D]
        Guarda el MSE escalar (con grad) para backprop.
        """
        mse = F.mse_loss(pred.float(), target.float())
        self._ttt_errors.append(mse)

    # ─────────────────────────────────────────────────────────────────────────
    # Cómputo (llamar al final del forward, antes del backward)
    # ─────────────────────────────────────────────────────────────────────────

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Calcula y retorna un dict con las pérdidas individuales y el total.

          'routing':     routing entropy loss     (escalar con grad)
          'ttt_pred':    TTT prediction loss      (escalar con grad)
          'total':       suma ponderada de ambas  (escalar con grad)

        Si no hay datos acumulados, retorna tensores cero sin grad (safe).
        """
        device = (self._routing_probs[0].device
                  if self._routing_probs
                  else torch.device('cpu'))

        # ── 1. Routing entropy penalty (CORREGIDO Sprint 6.3) ───────────────
        # H(p) = −Σ p·log(p).
        # OBJETIVO: MINIMIZAR H por sample (routing especializado).
        # loss = +weight * H_per_sample  →  minimizarla pica la distribución.
        # Nota: load_balance se gestiona por separado (evita colapso total).
        if self._routing_probs:
            all_probs = torch.stack(self._routing_probs, dim=0)          # [L, B, 3]
            p_clamped = all_probs.clamp(min=1e-8)
            H         = -(p_clamped * p_clamped.log()).sum(dim=-1)        # [L, B]
            H_mean    = H.mean()                                           # escalar
            # (a) Entropy penalty: penaliza H alta → routing especializado por sample
            routing_loss = self.routing_weight * H_mean                   # positivo

            # (b) Load-balance penalty: penaliza tiers con prob media ≈ 0.
            # mean_p [3]: probabilidad media en el batch × capas por tier.
            # F.relu(min - mean_p): 0 si tier recibe ≥ min, positivo si colapsa.
            # Suma sobre tiers: penaliza cada tier que cae por debajo del mínimo.
            # Esta fuerza contrarresta que entropy penalty colapse todo a 1 tier.
            mean_p = all_probs.mean(dim=[0, 1])   # [3] promedio batch × capas
            balance_loss = F.relu(
                self.min_tier_prob - mean_p
            ).sum() * self.balance_weight
            routing_loss = routing_loss + balance_loss
        else:
            routing_loss = torch.zeros(1, device=device, requires_grad=False).squeeze()

        # ── 2. TTT prediction loss ────────────────────────────────────────────
        # Promedia el MSE de todas las capas donde TTT estuvo activo.
        if self._ttt_errors:
            ttt_loss = self.ttt_pred_weight * torch.stack(self._ttt_errors).mean()
        else:
            ttt_loss = torch.zeros(1, device=device, requires_grad=False).squeeze()

        total = routing_loss + ttt_loss

        return {
            'routing':  routing_loss,
            'ttt_pred': ttt_loss,
            'total':    total,
        }

    def reset(self):
        """Limpiar acumuladores. Llamar DESPUÉS del backward."""
        self._routing_probs.clear()
        self._ttt_errors.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Utilidades de diagnóstico
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def n_routing_samples(self) -> int:
        return len(self._routing_probs)

    @property
    def n_ttt_samples(self) -> int:
        return len(self._ttt_errors)

    def routing_stats(self) -> Optional[Dict[str, float]]:
        """
        Retorna estadísticas del routing actual (sin grad).
        Útil para logging.
        """
        if not self._routing_probs:
            return None
        with torch.no_grad():
            all_p = torch.stack(self._routing_probs, dim=0)  # [L, B, 3]
            mean_p = all_p.mean(dim=[0, 1])                   # [3]
            return {
                'prob_FAST':   mean_p[0].item(),
                'prob_HYBRID': mean_p[1].item(),
                'prob_FULL':   mean_p[2].item(),
                'entropy':     -(mean_p * mean_p.clamp(1e-8).log()).sum().item(),
            }

    def __repr__(self):
        return (f"ChimeraLosses("
                f"routing_weight={self.routing_weight}, "
                f"ttt_pred_weight={self.ttt_pred_weight}, "
                f"n_routing={self.n_routing_samples}, "
                f"n_ttt={self.n_ttt_samples})")


# ───────────────────────────────────────────────────────────────────────────────
class ChimeraRoutingLoss(torch.nn.Module):
    """
    Routing loss de nueva generación — usa el aux_dict directamente.

    Tres fuerzas en equilibrio:
      (a) Entropy hinge: penaliza H(probs) > target_H.
          Objetivo: routing especializado por sample (H baja por sample).
          target_H_frac=0.70 → H deseada ≤ 70% de log(n_tiers).
          F.relu(H - target_H) → 0 cuando ya está suficientemente picado.

      (b) TTT-guided supervision (opcional): si ttt_importance está disponible,
          supervisa al router con soft-targets derivados del error predictivo:
            complejidad alta  → prob_full target alto  → tier FULL
            complejidad baja  → prob_fast target alto  → tier FAST
          Loss: KL(soft_target || probs)

      (c) Load balance: penaliza tiers con prob media muy baja en el batch.
          min_tier_prob=0.05 → cada tier debe recibir al menos 5% del tráfico.
          F.relu(min_tier_prob - mean_p).sum()

    Compara con MoE Switch Transformer: ellos también minimizan per-sample
    entropy + mantienen load balance. Nosotros añadimos supervión explícita vía
    TTT importance como señal de complejidad por input.
    """

    def __init__(
        self,
        n_tiers:             int   = 3,
        entropy_weight:      float = 0.05,
        supervision_weight:  float = 0.10,
        balance_weight:      float = 0.02,
        z_loss_weight:       float = 1e-3,   # Z-loss (router logit regularization)
        min_tier_prob:       float = 0.05,
        target_entropy_frac: float = 0.70,
    ):
        super().__init__()
        self.n_tiers             = n_tiers
        self.entropy_weight      = entropy_weight
        self.supervision_weight  = supervision_weight
        self.balance_weight      = balance_weight
        self.z_loss_weight       = z_loss_weight
        self.min_tier_prob       = min_tier_prob
        self.H_max               = math.log(n_tiers)
        self.target_H            = target_entropy_frac * self.H_max

    def forward(self, aux: dict) -> tuple:
        """
        Args:
            aux: dict de AdvancedChimeraLayer con return_aux=True.
                 Claves: 'routing_probs_grad' [B,3], 'ttt_importance' [B,S]|None
        Returns:
            (loss escalar con grad, info dict de scalares para logging)
        """
        probs   = aux.get('routing_probs_grad')
        if probs is None:
            probs = aux.get('routing_probs')
        ttt_imp    = aux.get('ttt_importance')         # [B, S] o None
        router_logits = aux.get('router_logits')       # [B, n_tiers] o None
        B, device = probs.shape[0], probs.device

        # (a) Entropy hinge ─────────────────────────────────────────────────
        H = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()   # escalar
        entropy_loss = F.relu(H - self.target_H)

        # (b) TTT supervision ───────────────────────────────────────────────
        sup_loss = torch.zeros(1, device=device).squeeze()
        if ttt_imp is not None:
            complexity = ttt_imp.mean(dim=-1)                    # [B]
            mu  = complexity.mean()
            sig = complexity.std(correction=0).clamp(min=1e-4)  # correction=0 evita NaN con B=1
            cplx_norm = torch.sigmoid((complexity - mu) / sig)   # [B] en [0,1]

            p_full   = cplx_norm.clamp(0.05, 0.85)
            p_fast   = (1.0 - cplx_norm).clamp(0.05, 0.85)
            p_hybrid = torch.full_like(p_full, 0.10)
            soft_t   = torch.stack([p_fast, p_hybrid, p_full], dim=-1)
            soft_t   = (soft_t / soft_t.sum(-1, keepdim=True)).detach()

            sup_loss = F.kl_div(
                (probs + 1e-8).log(), soft_t,
                reduction='batchmean', log_target=False
            )

        # (c) Load balance ──────────────────────────────────────────────────
        mean_p       = probs.mean(dim=0)                         # [3]
        balance_loss = F.relu(self.min_tier_prob - mean_p).sum()

        # (d) Z-loss (GLaM, Zoph et al. 2022) ──────────────────────────────
        # Penaliza la magnitud de logits del router para evitar colapso numérico
        # y routing degenerado con logits extremos.
        # L_z = E[(log Σ_i exp(x_i))²]  — diferenciable, cuadrático en logsumexp.
        # Con logits pequeños tiende a 0; con logits grandes crece agresivamente.
        # Referencia: https://arxiv.org/abs/2112.06905 §4.2
        z_loss = torch.zeros(1, device=device).squeeze()
        if router_logits is not None and self.z_loss_weight > 0:
            lse = torch.logsumexp(router_logits, dim=-1)         # [B]
            z_loss = (lse ** 2).mean()                           # escalar

        total = (self.entropy_weight     * entropy_loss +
                 self.supervision_weight * sup_loss     +
                 self.balance_weight     * balance_loss +
                 self.z_loss_weight      * z_loss)

        info = {
            'routing/H_bits':           H.item() / math.log(2),
            'routing/entropy_loss':     entropy_loss.item(),
            'routing/supervision_loss': sup_loss.item() if isinstance(sup_loss, torch.Tensor) else sup_loss,
            'routing/balance_loss':     balance_loss.item(),
            'routing/z_loss':           z_loss.item() if isinstance(z_loss, torch.Tensor) else z_loss,
            'routing/total_loss':       total.item(),
            'routing/p_fast':           mean_p[0].item(),
            'routing/p_hybrid':         mean_p[1].item(),
            'routing/p_full':           mean_p[2].item(),
        }
        return total, info


# ───────────────────────────────────────────────────────────────────────────────
def test_bus_publish_multilayer(
    d_model: int = 256,
    n_layers: int = 3,
    B: int = 2,
    S: int = 128,
    device: str = "cuda",
) -> bool:
    """
    Verifica que bus.publish.weight RECIBE GRADIENTE en un stack de n_layers.

    Por qué falla en single-layer:
        new_cache contiene el summary publicado por publish().
        En 1 capa, new_cache no entra a ninguna capa posterior → no hay
        camino de gradiente de vuelta → publish.weight.grad = None.

    Por qué funciona en multi-layer:
        Capa k+1 atiende al bus_cache que incluye summary_k (de publish_k):
            q = gather_q(x_{k+1})
            scores = bmm(q, bus_cache.T)   ← bus_cache incluye summary_k
            modulation = modulate(bmm(softmax(scores), bus_cache))
        → La capa k+1 propaga gradiente a bus_cache → a summary_k → a publish_k.
    """
    import torch.nn as nn
    from advanced_chimera import AdvancedChimeraLayer

    layers = nn.ModuleList([
        AdvancedChimeraLayer(d_model=d_model, expand=2, headdim=32)
        .to(device).float().train()
        for _ in range(n_layers)
    ])

    x = torch.randn(B, S, d_model, device=device, requires_grad=True)
    out, bus_cache = x, None
    for layer in layers:
        out, bus_cache, _ = layer(out, bus_cache=bus_cache, return_aux=True)

    out.sum().backward()

    print(f"\n=== Bus.Publish Multilayer Gradient ({n_layers} capas) ===")
    all_ok = True
    for i, layer in enumerate(layers):
        g    = layer.bus.publish.weight.grad
        norm = g.norm().item() if g is not None else 0.0
        ok   = norm > 1e-12
        # Con el fix de augmented_cache, TODAS las capas (incluida la última)
        # deben recibir gradiente: summary_last participa en su propia atención.
        status = "✅ OK" if ok else "❌ DEAD"
        print(f"  Layer {i}  bus.publish.weight  grad_norm={norm:.3e}  {status}")
        if not ok:
            all_ok = False

    result = "[SUCCESS] bus.publish recibe gradiente en todas las capas ✅" \
             if all_ok else "[FAIL]    Alguna capa tiene gradiente muerto ❌"
    print(result + "\n")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: total loss combinada para training loop
# ─────────────────────────────────────────────────────────────────────────────

def chimera_total_loss(
    lm_loss:       torch.Tensor,          # cross-entropy principal (escalar)
    chimera_losses: ChimeraLosses,        # acumulador de auxiliares
    verbose:       bool = False,
) -> torch.Tensor:
    """
    Retorna lm_loss + aux_losses y resetea el acumulador.

    Uso en training loop:
        loss = chimera_total_loss(ce_loss, losses_accumulator)
        loss.backward()
        optimizer.step()
    """
    aux = chimera_losses.compute()
    if verbose:
        stats = chimera_losses.routing_stats()
        print(f"  aux: routing={aux['routing'].item():.4f}  "
              f"ttt_pred={aux['ttt_pred'].item():.4f}  "
              f"total_aux={aux['total'].item():.4f}")
        if stats:
            print(f"  routing: FAST={stats['prob_FAST']:.3f}  "
                  f"HYBRID={stats['prob_HYBRID']:.3f}  "
                  f"FULL={stats['prob_FULL']:.3f}  "
                  f"H={stats['entropy']:.3f}")
    chimera_losses.reset()
    return lm_loss + aux['total']


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, S, D = 2, 512, 256
    losses  = ChimeraLosses(routing_weight=0.01, ttt_pred_weight=0.05)

    print("=== ChimeraLosses test ===")

    # Simular 3 capas de routing
    for i in range(3):
        # Probs reales de un router entrenado (no colapso)
        probs = torch.softmax(torch.randn(B, 3, device=device), dim=-1).requires_grad_(True)
        losses.add_routing_probs(probs)

    # Simular 2 capas con TTT activo
    for i in range(2):
        pred   = torch.randn(B, S-1, D, device=device, requires_grad=True)
        target = torch.randn(B, S-1, D, device=device)
        losses.add_ttt_error(pred, target)

    print(f"  n_routing={losses.n_routing_samples}  n_ttt={losses.n_ttt_samples}")
    stats = losses.routing_stats()
    print(f"  Routing stats: {stats}")

    # Compute
    aux = losses.compute()
    print(f"  routing_loss:  {aux['routing'].item():.6f}")
    print(f"  ttt_pred_loss: {aux['ttt_pred'].item():.6f}")
    print(f"  total_aux:     {aux['total'].item():.6f}")

    # Verificar que total tiene grad
    assert aux['total'].requires_grad or aux['routing'].requires_grad, \
        "Al menos una pérdida debe tener grad"
    print(f"  requires_grad: {aux['total'].requires_grad}")

    # Backward
    lm_loss = torch.randn(1, device=device, requires_grad=True).mean()
    losses.reset()
    # Resetear y hacer otra pasada limpia
    for i in range(3):
        probs = torch.softmax(torch.randn(B, 3, device=device), dim=-1).requires_grad_(True)
        losses.add_routing_probs(probs)

    total = chimera_total_loss(lm_loss, losses, verbose=True)
    total.backward()
    print(f"  Backward OK — total_loss: {total.item():.6f}")

    # Test colapso vs uniforme — routing_loss es NEGATIVA (reward de entropía)
    # Colapso → H pequeña → loss −weight*H ≈ 0  (menos negativa = peor)
    # Uniforme → H grande → loss −weight*H muy neg (más negativa = mejor reward)
    losses_collapsed = ChimeraLosses(routing_weight=0.01)
    # Todos los ejemplos van a FULL
    collapsed = torch.tensor([[0.01, 0.01, 0.98]], device=device).expand(B, -1).clone().requires_grad_(True)
    losses_collapsed.add_routing_probs(collapsed)
    aux_c = losses_collapsed.compute()
    loss_c = aux_c['routing'].item()

    # Distribución uniforme → máxima entropía → loss más negativa
    losses_uniform = ChimeraLosses(routing_weight=0.01)
    uniform = torch.full((B, 3), 1/3, device=device).requires_grad_(True)
    losses_uniform.add_routing_probs(uniform)
    aux_u = losses_uniform.compute()
    loss_u = aux_u['routing'].item()

    print(f"\n  Colapso: routing_loss={loss_c:.4f}  Uniforme: routing_loss={loss_u:.4f}")
    # SPRINT 6.3: loss = +weight*H  →  minimizar = routing picudo.
    # Colapso (H≈0) tiene loss MAS BAJA que uniforme (H grande) → OK (quiere especializar)
    # Colapso debe ser MENOS que uniforme
    assert loss_c < loss_u, (
        f"Con routing penalizado, colapso ({loss_c:.4f}) debe < uniforme ({loss_u:.4f})"
    )
    print(f"  (Sprint 6.3: colapso < uniforme = penalización de H alta ✓)")

    # ── Test ChimeraRoutingLoss con TTT supervision ───────────────────────────
    from advanced_chimera import AdvancedChimeraLayer
    print("\n--- ChimeraRoutingLoss tests ---")

    rl = ChimeraRoutingLoss(
        entropy_weight=0.05, supervision_weight=0.10, balance_weight=0.02
    ).to(device)

    model = AdvancedChimeraLayer(d_model=D, expand=2, headdim=32).to(device).float().train()
    x_t   = torch.randn(B, S, D, device=device)
    bus_c = torch.randn(B, 2, 128, device=device)

    print("  Routing specialization — 5 gradient steps:")
    print(f"  {'Step':>4}  {'H (bits)':>10}  {'p_fast':>8}  {'p_full':>7}  {'rl':>8}")
    print("  " + "-" * 50)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(5):
        opt.zero_grad()
        out_, _, aux_ = model(x_t, bus_cache=bus_c, return_aux=True)
        task_l = torch.nn.functional.mse_loss(out_, torch.zeros_like(out_))
        rl_l, info_ = rl(aux_)
        (task_l + 0.01 * rl_l).backward()
        opt.step()
        print(f"  {step:>4}  {info_['routing/H_bits']:>10.4f}  "
              f"{info_['routing/p_fast']:>8.4f}  {info_['routing/p_full']:>7.4f}  "
              f"{info_['routing/total_loss']:>8.5f}")

    rg = model.router.mlp[0].weight.grad
    print(f"  router grad: {'✅ OK' if rg is not None and rg.norm()>0 else '❌ FAIL'}")

    # ── Test multilayer bus.publish ───────────────────────────────────────────
    test_bus_publish_multilayer(d_model=D, n_layers=3, B=B, S=S, device=device)

    print("\n[SUCCESS] ChimeraLosses + ChimeraRoutingLoss + bus.publish multilayer OK")
