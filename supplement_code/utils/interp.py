import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

HOOK_LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']

MIN_ACTIVE_FOR_RANK = 5

class InterpAnalyser:
    def __init__(self, trainer, scenario, device: str = "cuda"):
        self.trainer  = trainer
        self.scenario = scenario
        self.device   = device

        self.drift_records: List[Dict] = []

    @torch.no_grad()
    def run(self):
        print("\n[Interp] Running latent drift analysis …")
        bank     = self.trainer._sae_bank
        scenario = self.scenario
        device   = self.device

        old_task_ids = bank.task_ids()

        for t_old in old_task_ids:

            act_loader = scenario.get_activation_loader(
                t_old, max_samples=2000)

            act_dict, labels = self.trainer.collect_all_activations(act_loader)

            for layer in HOOK_LAYERS:
                sae  = bank.get_sae(t_old, layer)
                info = bank.get_prototypes(t_old, layer)
                if sae is None or info is None:
                    continue

                sae   = sae.to(device)
                mask  = info['mask'].to(device)

                acts_now  = act_dict[layer].to(device)
                codes_now = sae.encode(acts_now)

                stored    = info['sparse'].to(device)

                active_idx = mask.nonzero(as_tuple=True)[0]

                for c_local, c_global in enumerate(
                    range(t_old * scenario.classes_per_task,
                          (t_old + 1) * scenario.classes_per_task)
                ):
                    mask_c = (labels == c_global)
                    if mask_c.sum() == 0:
                        continue

                    current_mean = codes_now[mask_c.to(device)].mean(0)
                    stored_code  = stored[c_local]

                    cur_sel = current_mean[active_idx]
                    sto_sel = stored_code[active_idx]

                    cos_sim = F.cosine_similarity(
                        cur_sel.unsqueeze(0), sto_sel.unsqueeze(0)).item()

                    l2_drift = (current_mean - stored_code).norm().item()

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

    def summary_by_layer(self) -> Dict[str, Dict]:
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

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)

        import json
        with open(os.path.join(out_dir, 'drift_records.json'), 'w') as f:
            json.dump(self.drift_records, f, indent=2)

        layer_sum = self.summary_by_layer()
        with open(os.path.join(out_dir, 'drift_by_layer.json'), 'w') as f:
            json.dump(layer_sum, f, indent=2)

        task_sum = self.summary_by_task()
        with open(os.path.join(out_dir, 'drift_by_task.json'), 'w') as f:
            json.dump({str(k): v for k, v in task_sum.items()}, f, indent=2)

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

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < MIN_ACTIVE_FOR_RANK:
        return float('nan')
    from scipy.stats import spearmanr
    try:
        r, _ = spearmanr(a, b)
        return float(r) if np.isfinite(r) else float('nan')
    except Exception:
        return float('nan')
