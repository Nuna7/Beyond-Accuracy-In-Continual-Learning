"""
method_compare.py — Cross-method mechanistic analysis.

Two analyses:

1. Task interference matrix
   R[i,j] = accuracy drop on task i caused by training on task j
   Built from the saved R matrix in results JSON.

2. EWC vs SAE alignment
   Core question: do EWC's high-Fisher parameters correspond to
   high-discriminativity SAE latent directions?

   For each layer we compute:
     - EWC Fisher importance per weight (already stored in EWCTrainer._fisher)
     - SAE discriminativity score per latent (from posthoc SAE bank)
     - The column-wise Fisher sum of weight matrices → per-output-unit importance
     - Spearman correlation between Fisher importance and SAE discriminativity

   Finding: if correlation is low, EWC is protecting weights that are NOT
   aligned with interpretable feature directions — which explains why it
   gives lower accuracy than methods that explicitly preserve feature space.
"""

import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── 1. Task interference matrix ───────────────────────────────────────────────

def build_interference_matrix(R: np.ndarray) -> np.ndarray:
    """
    interference[i, j] = accuracy of task i BEFORE training on task j
                        - accuracy of task i AFTER training on task j
    Positive = task j caused forgetting of task i.

    R shape: (T, T), R[after_task, task_id], NaN where not measured.
    """
    T = R.shape[0]
    I = np.full((T, T), np.nan)
    for i in range(T):        # old task
        for j in range(i+1, T):  # causative task
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
    """Return top-n (task_i, task_j, drop%) pairs by forgetting severity."""
    pairs = []
    T = I.shape[0]
    for i in range(T):
        for j in range(i+1, T):
            if not np.isnan(I[i, j]):
                pairs.append((i, j, float(I[i, j])))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:n]


# ── 2. EWC Fisher vs SAE discriminativity alignment ──────────────────────────

# Map ResNet-18 named parameters to their hook layer
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
    """
    For each hook layer, sum the Fisher importance of all weight parameters
    that feed into each output unit (channel).

    Returns {layer: array of shape (d_out,)} where d_out is the channel
    dimension of that layer's output.
    """
    fisher     = ewc_trainer._fisher    # {param_name: tensor}
    layer_fish = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}

    for name, F_diag in fisher.items():
        layer = _layer_of(name)
        if layer is None:
            continue
        # Conv weight: (out_ch, in_ch, kH, kW) — sum over in/kernel dims
        if F_diag.dim() == 4:
            importance = F_diag.sum(dim=(1, 2, 3)).cpu().numpy()  # (out_ch,)
        elif F_diag.dim() == 1:
            importance = F_diag.cpu().numpy()                      # BN/bias
        elif F_diag.dim() == 2:
            importance = F_diag.sum(dim=1).cpu().numpy()           # Linear
        else:
            continue
        layer_fish[layer].append(importance)

    result = {}
    for layer, arrays in layer_fish.items():
        if arrays:
            # Average across all parameter arrays in this layer
            # Truncate/pad to min length for simplicity
            min_len = min(a.shape[0] for a in arrays)
            stacked = np.stack([a[:min_len] for a in arrays], axis=0)
            result[layer] = stacked.mean(axis=0)
    return result


def sae_discriminativity_per_unit(posthoc_bank: dict, task_id: int = 0
                                   ) -> Dict[str, np.ndarray]:
    """
    From the post-hoc SAE bank, extract per-latent discriminativity scores
    and project them back to the activation dimension via the decoder.

    Returns {layer: array of shape (d_in,)} — importance per activation unit.
    """
    result = {}
    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        key = (task_id, layer)
        if key not in posthoc_bank:
            continue
        entry      = posthoc_bank[key]
        discrim    = torch.tensor(entry['discrim_scores'])  # (d_sae,)
        sae_state  = entry['sae_state']

        # decoder.weight: (d_in, d_sae)
        W_dec = sae_state.get('decoder.weight',
                sae_state.get('decoder.linear.weight', None))
        if W_dec is None:
            continue
        # For each activation unit i, its importance ≈ sum_j |W_dec[i,j]| * discrim[j]
        importance = (W_dec.abs() @ discrim).numpy()   # (d_in,)
        result[layer] = importance
    return result


def ewc_sae_alignment(ewc_fisher: Dict[str, np.ndarray],
                      sae_discrim: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """
    Compute Spearman correlation between EWC Fisher importance and
    SAE discriminativity importance, per layer.

    Returns {layer: {'spearman_r': float, 'p_value': float, 'n': int}}
    """
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

    # Interference matrix
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
