"""
Feature fate analysis.

For every SAE latent that was active for task t at the time it was trained,
we ask: what happened to it after all subsequent tasks?

  survived  — still active (cos_sim > hi_thresh, relative activity > act_thresh)
  drifted   — nominally active but direction changed (cos_sim < lo_thresh)
  dead      — no longer activates for this class (relative activity < dead_thresh)

Produces per-layer fate distributions and a per-class vulnerability ranking.

--------------------------------------------------------------------------------
NOTE ON rank_corr (added):
A naive Spearman rank correlation over the FULL d_sae-dimensional sparse code
(e.g. d_sae=512 with only k=8 nonzero entries) is dominated by tied ranks among
the ~98% of entries that are exactly zero in both vectors. Two completely
unrelated sparse codes still get rank_corr ~0.7-0.9 from this tie-inflation —
it reflects "both vectors are mostly zero," not "these vectors agree on order."

The fix used below: restrict the Spearman computation to active_idx (the same
mask already used for cos_sim), so we only compare the relative ordering of
features that were actually deemed important for this task. This is the same
masking pattern already used correctly for cos_sim in this file — rank_corr
was simply never computed at all here before.
--------------------------------------------------------------------------------
"""

import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── thresholds ────────────────────────────────────────────────────────────────
HI_COS   = 0.85   # cos_sim ≥ this → survived
LO_COS   = 0.50   # cos_sim < this → drifted (regardless of activity)
ACT_FRAC = 0.30   # relative activity < this → dead

# Minimum number of active dims required for a meaningful rank correlation.
# Spearman on fewer than this many points is too noisy to trust (e.g. with
# only 2-3 points, any pair of values gives a "perfect" or "perfectly
# anti-correlated" rank_corr by construction).
MIN_ACTIVE_FOR_RANK = 5


def _masked_rank_corr(cur_sel: torch.Tensor, sto_sel: torch.Tensor) -> float:
    """
    Spearman rank correlation restricted to the masked/active subspace.
    Returns NaN if there are too few active dims to compute a meaningful value.

    This is the corrected counterpart to the full-vector rank_corr previously
    computed in utils/interp.py, which suffered from tied-zero inflation when
    run on full d_sae-dimensional sparse codes.
    """
    n = cur_sel.numel()
    if n < MIN_ACTIVE_FOR_RANK:
        return float('nan')
    cur_np = cur_sel.detach().cpu().numpy()
    sto_np = sto_sel.detach().cpu().numpy()
    # If either vector is constant (all-equal values within the masked
    # subspace), Spearman is undefined — scipy returns nan, which we propagate.
    rho, _ = spearmanr(cur_np, sto_np)
    return float(rho) if np.isfinite(rho) else float('nan')


def classify_latent_fates(
    trainer,        # SAEFSGPTrainer that completed the full CL sequence
    scenario,       # ClassIncrementalScenario
    device: str,
) -> Dict:
    """
    Returns a dict:
      results[task_id][layer] = {
        'survived': int, 'drifted': int, 'dead': int, 'total': int,
        'per_class': {class_id: {'fate': str, 'cos_sim': float,
                                  'rank_corr': float, 'act_ratio': float,
                                  'n_active': int}}
      }
    """
    results = {}
    bank    = trainer._sae_bank

    for task_id in bank.task_ids():
        results[task_id] = {}

        act_loader         = scenario.get_activation_loader(task_id, max_samples=2000)
        act_dict, labels   = trainer.collect_all_activations(act_loader)
        labels_dev         = labels.to(device)

        classes_per_task   = scenario.classes_per_task
        task_start         = task_id * classes_per_task

        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            sae  = bank.get_sae(task_id, layer)
            info = bank.get_prototypes(task_id, layer)
            if sae is None or info is None:
                continue

            sae      = sae.to(device)
            mask_t   = info['mask'].to(device)           # (d_sae,) active latents
            acts_now = act_dict[layer].to(device)        # (N, d_in) CURRENT model
            codes_now = sae.encode(acts_now)              # (N, d_sae) via original SAE

            active_idx = mask_t.nonzero(as_tuple=True)[0]  # shared across all classes
            n_active   = int(active_idx.numel())

            survived, drifted, dead = 0, 0, 0
            per_class = {}

            for c_local in range(classes_per_task):
                c_global = task_start + c_local
                mask_c   = (labels_dev == c_global)
                if mask_c.sum() == 0:
                    continue

                # Current mean code for this class (via original task SAE)
                current_mean = codes_now[mask_c].mean(0)           # (d_sae,)
                stored_code  = info['sparse'][c_local].to(device)  # (d_sae,)

                # Only examine latents that were in the original task mask
                cur_sel = current_mean[active_idx]
                sto_sel = stored_code[active_idx]

                # Activity ratio: how much of original activation remains
                orig_norm  = sto_sel.norm().item()
                act_ratio  = (cur_sel.norm() / (orig_norm + 1e-8)).item()

                # Cosine similarity of masked code vectors
                cos_sim = F.cosine_similarity(
                    cur_sel.unsqueeze(0), sto_sel.unsqueeze(0)
                ).item()

                # Masked rank correlation — see _masked_rank_corr docstring.
                rank_corr = _masked_rank_corr(cur_sel, sto_sel)

                # Classify
                if act_ratio < ACT_FRAC:
                    fate = 'dead'
                    dead += 1
                elif cos_sim < LO_COS:
                    fate = 'drifted'
                    drifted += 1
                else:
                    fate = 'survived'
                    survived += 1

                per_class[c_global] = {
                    'fate':      fate,
                    'cos_sim':   round(cos_sim, 4),
                    'rank_corr': round(rank_corr, 4) if np.isfinite(rank_corr) else None,
                    'act_ratio': round(act_ratio, 4),
                    'n_active':  n_active,
                }

            results[task_id][layer] = {
                'survived':  survived,
                'drifted':   drifted,
                'dead':      dead,
                'total':     survived + drifted + dead,
                'n_active':  n_active,
                'per_class': per_class,
            }
            sae.to('cpu')

    return results


def fate_summary(results: Dict) -> Dict:
    """
    Aggregate fate counts AND mean cos_sim/rank_corr by layer across all tasks.
    Returns {layer: {survived%, drifted%, dead%, mean_cos_sim, mean_rank_corr}}
    """
    by_layer = defaultdict(lambda: {'survived': 0, 'drifted': 0, 'dead': 0, 'total': 0})
    cos_vals  = defaultdict(list)
    rank_vals = defaultdict(list)

    for task_id, layers in results.items():
        for layer, stats in layers.items():
            for k in ('survived', 'drifted', 'dead', 'total'):
                by_layer[layer][k] += stats[k]
            for c_global, info in stats.get('per_class', {}).items():
                cos_vals[layer].append(info['cos_sim'])
                if info.get('rank_corr') is not None:
                    rank_vals[layer].append(info['rank_corr'])

    summary = {}
    for layer, counts in by_layer.items():
        t = max(counts['total'], 1)
        summary[layer] = {
            'survived_pct':    round(100 * counts['survived'] / t, 1),
            'drifted_pct':     round(100 * counts['drifted']  / t, 1),
            'dead_pct':        round(100 * counts['dead']      / t, 1),
            'total':           counts['total'],
            'mean_cos_sim':    round(float(np.mean(cos_vals[layer])), 4) if cos_vals[layer] else None,
            'mean_rank_corr':  round(float(np.mean(rank_vals[layer])), 4) if rank_vals[layer] else None,
            'n_rank_samples':  len(rank_vals[layer]),
        }
    return summary


def most_vulnerable_classes(results: Dict, layer: str = 'layer4',
                             n: int = 10) -> List[Tuple]:
    """
    Return the n classes with lowest mean cos_sim at layer4 after all tasks.
    Useful for identifying which CIFAR-100 categories are hardest to preserve.
    """
    scores = []
    for task_id, layers in results.items():
        if layer not in layers:
            continue
        for c_global, info in layers[layer]['per_class'].items():
            scores.append((c_global, info['cos_sim'], info['fate']))
    scores.sort(key=lambda x: x[1])
    return scores[:n]


def save_fate_results(results: Dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Full per-class records
    serialisable = {}
    for tid, layers in results.items():
        serialisable[str(tid)] = {}
        for layer, stats in layers.items():
            serialisable[str(tid)][layer] = {
                k: v for k, v in stats.items() if k != 'per_class'
            }
            serialisable[str(tid)][layer]['per_class'] = {
                str(cg): v for cg, v in stats.get('per_class', {}).items()
            }

    with open(os.path.join(out_dir, 'feature_fate.json'), 'w') as f:
        json.dump(serialisable, f, indent=2)

    summary = fate_summary(results)
    with open(os.path.join(out_dir, 'fate_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Human-readable table
    print(f"\n{'─'*70}")
    print(f"  Feature Fate Summary (across all tasks and classes)")
    print(f"{'─'*70}")
    print(f"  {'Layer':<10} {'Survived':>10} {'Drifted':>10} {'Dead':>8}"
          f"  {'cos_sim':>9}  {'rank_corr':>10}")
    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        if layer in summary:
            s = summary[layer]
            cos_str  = f"{s['mean_cos_sim']:.4f}"   if s['mean_cos_sim']   is not None else "  N/A "
            rank_str = f"{s['mean_rank_corr']:.4f}" if s['mean_rank_corr'] is not None else "  N/A "
            print(f"  {layer:<10} {s['survived_pct']:>9.1f}% "
                  f"{s['drifted_pct']:>9.1f}% {s['dead_pct']:>7.1f}%"
                  f"  {cos_str:>9}  {rank_str:>10}")
    print(f"{'─'*70}")
    print(f"  Results saved → {out_dir}")