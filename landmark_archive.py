"""
OrthoSSM V10 — AsyncLightBus + Landmark Archive
=================================================
V10 changes:
  - AsyncLightBus replaces CrossLayerMemoryBus
    * 64-dim summary vector instead of full cross-attention
    * No global dependency — layers execute nearly independently
    * O(1) memory per layer instead of O(landmarks * d_model)
  
  E5: Versioned bus with gradient checkpoint support
    * Forward-pass versioning rejects stale data
    * Norm-based canary detects recompute divergence
    * Snapshot/restore for checkpoint compatibility

  - LandmarkArchive: unchanged from V9 (already optimized)
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from ortho_diagnostics import DIAG


class AsyncLightBus(nn.Module):
    """
    V10 Async Lightweight Memory Bus with Versioning (E5).

    Each layer publishes a 64-dim summary vector (instead of full landmarks).
    Upper layers can gather summaries from lower layers via simple averaging.

    E5 improvements over original:
      - Forward pass versioning: entries are tagged with version ID,
        stale data from previous forwards is automatically rejected.
      - Snapshot/restore: for gradient checkpointing, the bus state can be
        snapshotted before backward and restored during recompute.
      - Norm canary: each entry stores its L2 norm for O(1) divergence detection
        during gradient checkpoint recompute.
    """
    def __init__(self, summary_dim=64, n_layers=2):
        super().__init__()
        self.summary_dim = summary_dim
        self.n_layers = n_layers
        # E5: Versioned store replaces simple list
        self._store: Dict[int, Dict[str, Any]] = {}
        self._version: int = 0
        self._snapshot: Optional[Dict] = None
        self._recompute_mode: bool = False

    def clear(self):
        """New forward pass: increment version, clear store (E5)."""
        self._version += 1
        self._store.clear()
        self._recompute_mode = False

    def publish(self, layer_idx, summary):
        """
        Publish a summary vector for this layer.
        summary: [B, summary_dim]
        E5: Tagged with version and norm canary.
        """
        entry = {
            'value': summary.detach(),
            'version': self._version,
            'norm': summary.detach().norm().item(),
        }

        if self._recompute_mode:
            # During recompute, check consistency with original forward
            old = self._store.get(layer_idx)
            if old is not None:
                norm_diff = abs(old['norm'] - entry['norm'])
                if norm_diff > 1e-2:
                    import warnings
                    warnings.warn(
                        f"LightBus: recompute divergence at layer {layer_idx}, "
                        f"norm diff = {norm_diff:.6f}"
                    )

        self._store[layer_idx] = entry

    def gather(self, current_layer_idx, batch_size, device):
        """
        Gather summaries from all layers below current_layer_idx.
        E5: Validates version to reject stale data.
        Returns: [B, summary_dim] averaged summary, or None.
        """
        summaries = []
        for i in range(current_layer_idx):
            entry = self._store.get(i)
            if entry is not None:
                # E5: Version check — reject stale data
                if entry['version'] != self._version:
                    continue
                summaries.append(entry['value'])

        if not summaries:
            return None

        # Diagnostic: staleness = layers below that are missing/stale
        n_missing = current_layer_idx - len(summaries)
        DIAG.record_bus_staleness(current_layer_idx, n_missing)

        stacked = torch.stack(summaries, dim=0)  # [n_lower, B, summary_dim]
        return stacked.mean(dim=0)  # [B, summary_dim]

    def snapshot(self):
        """Take snapshot for gradient checkpointing (E5)."""
        self._snapshot = {
            k: dict(v) for k, v in self._store.items()
        }

    def enter_recompute(self):
        """Restore from snapshot for gradient checkpoint recompute (E5)."""
        if self._snapshot is not None:
            self._store = {
                k: dict(v) for k, v in self._snapshot.items()
            }
            self._recompute_mode = True

    @property
    def stats(self):
        """Diagnostic info for the bus."""
        return {
            'version': self._version,
            'n_entries': len(self._store),
            'recompute_mode': self._recompute_mode,
            'has_snapshot': self._snapshot is not None,
        }


class LandmarkArchive(nn.Module):
    """
    Intelligent Landmark State Archive (unchanged from V9).
    - Importance-based archiving
    - Adaptive interval
    - Weighted merge
    - Self-attention between landmarks
    """
    def __init__(self, d_model, n_cheby_heads, head_dim, max_degree=4,
                 max_landmarks=64, archive_interval=131072):
        super().__init__()
        self.d_model = d_model
        self.n_cheby_heads = n_cheby_heads
        self.head_dim = head_dim
        self.max_degree = max_degree
        self.max_landmarks = max_landmarks
        self.base_archive_interval = archive_interval

        state_flat_dim = n_cheby_heads * max_degree * head_dim

        self.state_to_embedding = nn.Sequential(
            nn.Linear(state_flat_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.importance_predictor = nn.Sequential(
            nn.Linear(state_flat_dim, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

        self.landmark_self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, batch_first=True, dropout=0.0
        )
        self.landmark_norm = nn.LayerNorm(d_model)

        self.summary_gate = nn.Linear(d_model, 1, bias=False)

        self.register_buffer('token_counter', torch.tensor(0, dtype=torch.long))
        self.register_buffer('n_archived', torch.tensor(0, dtype=torch.long))
        self.register_buffer('archived_states',
            torch.zeros(max_landmarks, n_cheby_heads, max_degree, head_dim))
        self.register_buffer('archived_token_positions',
            torch.zeros(max_landmarks, dtype=torch.long))
        self.register_buffer('archived_importance',
            torch.zeros(max_landmarks))
        self.register_buffer('complexity_accum', torch.tensor(0.0))
        self.register_buffer('complexity_count', torch.tensor(0, dtype=torch.long))

        # P2: Embedding cache — invalidated only when new state is archived.
        # Eliminates the 2-layer MLP + self-attention recompute on every forward
        # (these are unchanged when n_archived hasn't grown).
        self._emb_cache       = None   # [n, d_model] or None
        self._emb_cache_n     = -1     # number of landmarks when cache was built

    def maybe_archive(self, cheby_state, n_new_tokens, complexity_score=0.5):
        """Adaptive archiving: archive sooner when complexity is high."""
        self.token_counter += n_new_tokens
        self.complexity_accum += complexity_score * n_new_tokens
        self.complexity_count += n_new_tokens

        avg_complexity = (self.complexity_accum / max(self.complexity_count.item(), 1)).item()
        scale = max(0.3, 1.5 - avg_complexity)
        adaptive_interval = int(self.base_archive_interval * scale)

        should_archive = self.token_counter >= adaptive_interval
        if avg_complexity > 0.7 and self.complexity_count >= 512:
            should_archive = True

        if should_archive:
            state = cheby_state[0].detach() if isinstance(cheby_state, tuple) else cheby_state.detach()
            importance = self._compute_importance(state)
            self._archive_snapshot(state, importance)
            self.token_counter.zero_()
            self.complexity_accum.zero_()
            self.complexity_count.zero_()
            return True
        return False

    def _compute_importance(self, state):
        flat = state.reshape(1, -1)
        with torch.no_grad():
            imp = self.importance_predictor(flat)
        return imp.item()

    def _archive_snapshot(self, state, importance):
        n = self.n_archived.item()
        if n >= self.max_landmarks:
            self._importance_based_merge()
            n = self.n_archived.item()

        self.archived_states[n] = state.detach().clone()
        self.archived_token_positions[n] = self.token_counter.item()
        self.archived_importance[n] = importance
        self.n_archived += 1
        # P2: Invalidate embedding cache — new landmark was added
        self._emb_cache   = None
        self._emb_cache_n = -1

    def _importance_based_merge(self):
        """Vectorized importance-based hierarchical merge."""
        n = self.n_archived.item()
        merge_count = n // 2
        keep_start = merge_count * 2

        imp_a = self.archived_importance[:keep_start:2]
        imp_b = self.archived_importance[1:keep_start:2]
        states_a = self.archived_states[:keep_start:2]
        states_b = self.archived_states[1:keep_start:2]

        total_imp = imp_a + imp_b + 1e-8
        w_a = (0.3 + 0.4 * (imp_a / total_imp)).view(-1, 1, 1, 1)
        w_b = 1.0 - w_a

        new_states = torch.zeros_like(self.archived_states)
        new_pos = torch.zeros_like(self.archived_token_positions)
        new_imp = torch.zeros_like(self.archived_importance)

        new_states[:merge_count] = states_a * w_a + states_b * w_b
        new_pos[:merge_count] = self.archived_token_positions[1:keep_start:2]
        new_imp[:merge_count] = torch.maximum(imp_a, imp_b)

        remaining = n - keep_start
        if remaining > 0:
            new_states[merge_count:merge_count+remaining] = self.archived_states[keep_start:keep_start+remaining]
            new_pos[merge_count:merge_count+remaining] = self.archived_token_positions[keep_start:keep_start+remaining]
            new_imp[merge_count:merge_count+remaining] = self.archived_importance[keep_start:keep_start+remaining]

        self.archived_states.copy_(new_states)
        self.archived_token_positions.copy_(new_pos)
        self.archived_importance.copy_(new_imp)
        self.n_archived.fill_(merge_count + remaining)

    def get_landmark_embeddings(self, batch_size, device):
        """k=12 recent + 1 global summary with self-attention.

        P2: Embedding computation is cached per archive version.
        The MLP + self-attn only run when n_archived changed (new landmark
        was stored). In the vast majority of forwards (no archiving), this
        returns the pre-computed tensor in O(1) — no matmuls, no SiLU.
        """
        n = self.n_archived.item()
        if n == 0:
            return None, 0

        # P2: Return cached embeddings if the archive hasn't changed
        if self._emb_cache is not None and self._emb_cache_n == n:
            combined = self._emb_cache.to(device)
            k = min(n, 12)
            return combined.unsqueeze(0).expand(batch_size, -1, -1), k + 1

        # Cache miss — recompute embeddings
        active  = self.archived_states[:n].to(device)
        flat    = active.reshape(n, -1)
        embeds  = self.state_to_embedding(flat)

        if n >= 2:
            embeds_batched = embeds.unsqueeze(0)
            attn_out, _ = self.landmark_self_attn(
                embeds_batched, embeds_batched, embeds_batched
            )
            embeds = self.landmark_norm(embeds + attn_out.squeeze(0))

        imp_weights  = self.archived_importance[:n].to(device)
        imp_weights  = torch.softmax(imp_weights * 5.0, dim=0)
        gate_scores  = self.summary_gate(embeds).squeeze(-1) + imp_weights
        gate_weights = torch.softmax(gate_scores, dim=0).unsqueeze(-1)
        global_summary = (embeds * gate_weights).sum(dim=0, keepdim=True)

        k        = min(n, 12)
        recent   = embeds[-k:]
        combined = torch.cat([recent, global_summary], dim=0)   # [k+1, d_model]

        # P2: Store in cache (detached — no gradient retained)
        self._emb_cache   = combined.detach()
        self._emb_cache_n = n

        return combined.unsqueeze(0).expand(batch_size, -1, -1), k + 1

    def get_archive_info(self):
        n = self.n_archived.item()
        return {
            'n_landmarks': n,
            'max_landmarks': self.max_landmarks,
            'token_counter': self.token_counter.item(),
            'archive_interval': self.base_archive_interval,
            'memory_kb': (n * self.archived_states[0].numel() * 4) / 1024,
        }
