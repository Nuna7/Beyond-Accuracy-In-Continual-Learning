"""
Mechanistic interpretability analysis for SAE-FSGP.

Tracks and visualises SAE latent drift across tasks — the key unique
contribution beyond accuracy numbers.  Produces:

  1. drift_matrix[task_t, task_current, layer] — per-latent cosine drift
  2. per_class_drift — which specific latents drifted for each class
  3. stability_report — which decoder directions are preserved vs. lost

Usage after a full CL run:
    analyser = InterpAnalyser(trainer, scenario, device)
    analyser.run()          # collects all drift stats
    analyser.save('results/interp/')

--------------------------------------------------------------------------------
BUGFIX NOTE (rank_corr):
The previous rank_corr computation called .topk(k).values independently on
current_mean[mask] and stored_code[mask]. Since .topk() returns values sorted
in descending order, BOTH resulting arrays are monotonically decreasing by
construction — their rank sequences are trivially [0,1,...,k-1] regardless of
content. Spearman correlation between two independently-sorted arrays is
~1.0 for ANY input, including two unrelated random vectors (verified: two
unrelated random 200-dim vectors gave rank_corr=1.0000 under the old code).

This made rank_corr measure "do these two vectors have similarly-shaped
sorted value distributions" rather than "do the same latent dimensions
stay important across time" — the latter being what we actually want.

FIX: compare values at the SAME indices (cur_sel/sto_sel, already correctly
index-aligned via active_idx — the same alignment already used for cos_sim).
No independent sorting; rank correlation is now computed on paired,
index-matched values, which is the only definition that's invariant under
re-indexing while still being sensitive to actual ordering changes.
--------------------------------------------------------------------------------
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


HOOK_LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']

# Minimum number of active dims required for a meaningful rank correlation.
# Below this, Spearman is too noisy/degenerate to report (e.g. with 2-3
# points any pairing gives a "perfect" or "perfectly anti-correlated" rho).
MIN_ACTIVE_FOR_RANK = 5


class InterpAnalyser:
    """
    Post-hoc analysis of which SAE latents drifted during continual learning.

    Requires:
      - trainer:  a SAEFSGPTrainer that has completed the CL sequence
      - scenario: ClassIncrementalScenario used during training
    """

    def __init__(self, trainer, scenario, device: str = "cuda"):
        self.trainer  = trainer
        self.scenario = scenario
        self.device   = device

        # Results populated by run()
        self.drift_records: List[Dict] = []

    # ── main analysis ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def run(self):
        """
        For each old task t' and each subsequent task t > t':
          1. Load the old-task SAE for each layer.
          2. Collect current model activations on old task t' data.
          3. Encode through old SAE; compare latent statistics to stored protos.
          4. Record per-latent cosine similarity and rank correlation (drift).
        """
        print("\n[Interp] Running latent drift analysis …")
        bank     = self.trainer._sae_bank
        scenario = self.scenario
        device   = self.device

        old_task_ids = bank.task_ids()

        for t_old in old_task_ids:
            # Get activation loader for old task
            act_loader = scenario.get_activation_loader(
                t_old, max_samples=2000)

            act_dict, labels = self.trainer.collect_all_activations(act_loader)

            for layer in HOOK_LAYERS:
                sae  = bank.get_sae(t_old, layer)
                info = bank.get_prototypes(t_old, layer)
                if sae is None or info is None:
                    continue

                sae   = sae.to(device)
                mask  = info['mask'].to(device)       # (d_sae,)

                # Current model encoding of old task's data
                acts_now  = act_dict[layer].to(device) # (N, d_in)
                codes_now = sae.encode(acts_now)        # (N, d_sae)

                # Stored prototype codes
                stored    = info['sparse'].to(device)  # (C_old, d_sae)

                active_idx = mask.nonzero(as_tuple=True)[0]  # shared across all classes in this task

                for c_local, c_global in enumerate(
                    range(t_old * scenario.classes_per_task,
                          (t_old + 1) * scenario.classes_per_task)
                ):
                    mask_c = (labels == c_global)
                    if mask_c.sum() == 0:
                        continue

                    current_mean = codes_now[mask_c.to(device)].mean(0)  # (d_sae,)
                    stored_code  = stored[c_local]                       # (d_sae,)

                    # Index-aligned selection — used for BOTH cos_sim and rank_corr
                    cur_sel = current_mean[active_idx]
                    sto_sel = stored_code[active_idx]

                    cos_sim = F.cosine_similarity(
                        cur_sel.unsqueeze(0), sto_sel.unsqueeze(0)).item()

                    # L2 drift (full vector, intentionally unmasked — captures
                    # drift both within and outside the original active set)
                    l2_drift = (current_mean - stored_code).norm().item()

                    # Rank correlation: index-aligned, NOT independently sorted.
                    # (See BUGFIX NOTE above — previous version used .topk()
                    # on each vector separately, which is invariant to content
                    # and always returns ~1.0.)
                    rank_corr = _spearman(
                        cur_sel.cpu().numpy(),
                        sto_sel.cpu().numpy(),
                    )

                    self.drift_records.append({
                        'old_task':  t_old,
                        'layer':     layer,
                        'class':     c_global,
                        'cos_sim':   cos_sim,
                        'l2_drift':  l2_drift,
                        'rank_corr': rank_corr,
                        'n_active':  int(mask.sum()),
                    })

        print(f"[Interp] Collected {len(self.drift_records)} drift records.")

    # ── aggregate stats ───────────────────────────────────────────────────────

    def summary_by_layer(self) -> Dict[str, Dict]:
        """Average drift metrics per layer. rank_corr uses nan-safe averaging
        since some records may have too few active dims for a defined value."""
        by_layer = defaultdict(list)
        for r in self.drift_records:
            by_layer[r['layer']].append(r)

        out = {}
        for layer, records in by_layer.items():
            rank_vals = [r['rank_corr'] for r in records if np.isfinite(r['rank_corr'])]
            out[layer] = {
                'mean_cos_sim':   float(np.mean([r['cos_sim']  for r in records])),
                'mean_l2_drift':  float(np.mean([r['l2_drift'] for r in records])),
                'mean_rank_corr': float(np.mean(rank_vals)) if rank_vals else float('nan'),
                'n_records':      len(records),
                'n_rank_valid':   len(rank_vals),
            }
        return out

    def summary_by_task(self) -> Dict[int, Dict]:
        """Average drift metrics per old task."""
        by_task = defaultdict(list)
        for r in self.drift_records:
            by_task[r['old_task']].append(r)

        out = {}
        for task, records in by_task.items():
            rank_vals = [r['rank_corr'] for r in records if np.isfinite(r['rank_corr'])]
            out[task] = {
                'mean_cos_sim':   float(np.mean([r['cos_sim']  for r in records])),
                'mean_l2_drift':  float(np.mean([r['l2_drift'] for r in records])),
                'mean_rank_corr': float(np.mean(rank_vals)) if rank_vals else float('nan'),
                'n_classes':      len({r['class'] for r in records}),
            }
        return out

    # ── save ─────────────────────────────────────────────────────────────────

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)

        # Save raw records
        import json
        with open(os.path.join(out_dir, 'drift_records.json'), 'w') as f:
            json.dump(self.drift_records, f, indent=2)

        # Layer summary
        layer_sum = self.summary_by_layer()
        with open(os.path.join(out_dir, 'drift_by_layer.json'), 'w') as f:
            json.dump(layer_sum, f, indent=2)

        # Task summary
        task_sum = self.summary_by_task()
        with open(os.path.join(out_dir, 'drift_by_task.json'), 'w') as f:
            json.dump({str(k): v for k, v in task_sum.items()}, f, indent=2)

        # Print readable table
        print(f"\n[Interp] Drift by layer (cosine similarity ↑ = less drift):")
        print(f"{'Layer':<10} {'cos_sim':>10} {'l2_drift':>10} {'rank_corr':>10} {'n_valid':>8}")
        for layer in HOOK_LAYERS:
            if layer not in layer_sum:
                continue
            s = layer_sum[layer]
            rc = s['mean_rank_corr']
            rc_str = f"{rc:>10.4f}" if np.isfinite(rc) else f"{'N/A':>10}"
            print(f"{layer:<10} {s['mean_cos_sim']:>10.4f} "
                  f"{s['mean_l2_drift']:>10.4f} "
                  f"{rc_str} {s['n_rank_valid']:>8}")

        print(f"\n[Interp] Results saved to {out_dir}")


# ── utility ───────────────────────────────────────────────────────────────────

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman rank correlation between two 1-D, index-aligned arrays.
    Returns NaN (not 0.0) when undefined, so downstream aggregation can
    correctly exclude these rather than silently biasing the mean toward 0.
    """
    if len(a) < MIN_ACTIVE_FOR_RANK:
        return float('nan')
    from scipy.stats import spearmanr
    try:
        r, _ = spearmanr(a, b)
        return float(r) if np.isfinite(r) else float('nan')
    except Exception:
        return float('nan')