import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

def build_interference_matrix(R: np.ndarray) -> np.ndarray:
    T = R.shape[0]
    I = np.full((T, T), np.nan)
    for i in range(T):
        for j in range(i+1, T):
            before = R[j-1, i] if j > 0 and not np.isnan(R[j-1, i]) else np.nan
            after  = R[j, i]   if not np.isnan(R[j, i]) else np.nan
            if not np.isnan(before) and not np.isnan(after):
                I[i, j] = max(0.0, before - after)
    return I

def load_R_matrix(json_path: str) -> np.ndarray:
    with open(json_path) as f:
        data = json.load(f)
    R_raw = data['R']
    T = len(R_raw)
    R = np.full((T, T), np.nan)
    for i, row in enumerate(R_raw):
        for j, val in enumerate(row):
            if val is not None:
                R[i, j] = val
    return R

def worst_interference_pairs(I: np.ndarray, n: int = 5) -> List:
    pairs = []
    T = I.shape[0]
    for i in range(T):
        for j in range(i+1, T):
            if not np.isnan(I[i, j]):
                pairs.append((i, j, float(I[i, j])))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:n]

_PARAM_TO_LAYER = {
    'layer1': 'layer1', 'layer2': 'layer2',
    'layer3': 'layer3', 'layer4': 'layer4',
}

def _layer_of(param_name: str) -> Optional[str]:
    for k, v in _PARAM_TO_LAYER.items():
        if k in param_name:
            return v
    return None

def ewc_fisher_per_output_unit(ewc_trainer) -> Dict[str, np.ndarray]:
    fisher     = ewc_trainer._fisher
    layer_fish = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}

    for name, F_diag in fisher.items():
        layer = _layer_of(name)
        if layer is None:
            continue

        if F_diag.dim() == 4:
            importance = F_diag.sum(dim=(1, 2, 3)).cpu().numpy()
        elif F_diag.dim() == 1:
            importance = F_diag.cpu().numpy()
        elif F_diag.dim() == 2:
            importance = F_diag.sum(dim=1).cpu().numpy()
        else:
            continue
        layer_fish[layer].append(importance)

    result = {}
    for layer, arrays in layer_fish.items():
        if arrays:

            min_len = min(a.shape[0] for a in arrays)
            stacked = np.stack([a[:min_len] for a in arrays], axis=0)
            result[layer] = stacked.mean(axis=0)
    return result

def sae_discriminativity_per_unit(posthoc_bank: dict, task_id: int = 0
                                   ) -> Dict[str, np.ndarray]:
    result = {}
    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        key = (task_id, layer)
        if key not in posthoc_bank:
            continue
        entry      = posthoc_bank[key]
        discrim    = torch.tensor(entry['discrim_scores'])
        sae_state  = entry['sae_state']

        W_dec = sae_state.get('decoder.weight',
                sae_state.get('decoder.linear.weight', None))
        if W_dec is None:
            continue

        importance = (W_dec.abs() @ discrim).numpy()
        result[layer] = importance
    return result

def ewc_sae_alignment(ewc_fisher: Dict[str, np.ndarray],
                      sae_discrim: Dict[str, np.ndarray]) -> Dict[str, dict]:
    results = {}
    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        if layer not in ewc_fisher or layer not in sae_discrim:
            continue
        f = ewc_fisher[layer]
        s = sae_discrim[layer]
        min_len = min(len(f), len(s))
        if min_len < 5:
            continue
        r, p = spearmanr(f[:min_len], s[:min_len])
        results[layer] = {'spearman_r': float(r), 'p_value': float(p), 'n': min_len}
    return results

def save_comparison_results(interference: np.ndarray,
                             ewc_alignment: Optional[Dict],
                             out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'interference_matrix.npy'), interference)
    worst = worst_interference_pairs(interference)
    with open(os.path.join(out_dir, 'worst_interference.json'), 'w') as f:
        json.dump([{'task_i': a, 'task_j': b, 'drop_pct': round(c, 2)}
                   for a, b, c in worst], f, indent=2)

    print(f"\n  Top interference pairs:")
    for i, j, drop in worst[:5]:
        print(f"    task {i} → task {j}: −{drop:.1f}%")

    if ewc_alignment:
        with open(os.path.join(out_dir, 'ewc_sae_alignment.json'), 'w') as f:
            json.dump(ewc_alignment, f, indent=2)
        print(f"\n  EWC Fisher vs SAE discriminativity (Spearman ρ):")
        for layer, stats in ewc_alignment.items():
            print(f"    {layer}: ρ={stats['spearman_r']:.3f}  p={stats['p_value']:.3e}")
        avg_r = np.mean([v['spearman_r'] for v in ewc_alignment.values()])
        print(f"    mean ρ = {avg_r:.3f}  "
              f"{'(weak — EWC does not protect interpretable features)'  if avg_r < 0.3 else ''}")
