import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

HI_COS   = 0.85
LO_COS   = 0.50
ACT_FRAC = 0.30

MIN_ACTIVE_FOR_RANK = 5

def _masked_rank_corr(cur_sel: torch.Tensor, sto_sel: torch.Tensor) -> float:
    n = cur_sel.numel()
    if n < MIN_ACTIVE_FOR_RANK:
        return float('nan')
    cur_np = cur_sel.detach().cpu().numpy()
    sto_np = sto_sel.detach().cpu().numpy()

    rho, _ = spearmanr(cur_np, sto_np)
    return float(rho) if np.isfinite(rho) else float('nan')

def classify_latent_fates(
    trainer,
    scenario,
    device: str,
) -> Dict:
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
            mask_t   = info['mask'].to(device)
            acts_now = act_dict[layer].to(device)
            codes_now = sae.encode(acts_now)

            active_idx = mask_t.nonzero(as_tuple=True)[0]
            n_active   = int(active_idx.numel())

            survived, drifted, dead = 0, 0, 0
            per_class = {}

            for c_local in range(classes_per_task):
                c_global = task_start + c_local
                mask_c   = (labels_dev == c_global)
                if mask_c.sum() == 0:
                    continue

                current_mean = codes_now[mask_c].mean(0)
                stored_code  = info['sparse'][c_local].to(device)

                cur_sel = current_mean[active_idx]
                sto_sel = stored_code[active_idx]

                orig_norm  = sto_sel.norm().item()
                act_ratio  = (cur_sel.norm() / (orig_norm + 1e-8)).item()

                cos_sim = F.cosine_similarity(
                    cur_sel.unsqueeze(0), sto_sel.unsqueeze(0)
                ).item()

                rank_corr = _masked_rank_corr(cur_sel, sto_sel)

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
    scores = []
    for _, layers in results.items():
        if layer not in layers:
            continue
        for c_global, info in layers[layer]['per_class'].items():
            scores.append((c_global, info['cos_sim'], info['fate']))
    scores.sort(key=lambda x: x[1])
    return scores[:n]

def save_fate_results(results: Dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

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
