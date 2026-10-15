import os, sys, json, torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from sae.compat import AutoEncoderTopK
from models.resnet_hook import build_resnet18

TARGET_LAYER = 'layer4'
HI_COS   = 0.85
LO_COS   = 0.50
ACT_FRAC = 0.30
HOOK_LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']
HOOK_DIMS   = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}

def _infer_k(d_in, state):
    if 'k' in state:
        k_val = state['k']
        return int(k_val.item()) if hasattr(k_val, 'item') else int(k_val)
    return max(8, d_in // 16)

def reconstruct_sae(bank_data, task_id, layer, device):
    key = (task_id, layer)
    if key not in bank_data:
        return None
    state = bank_data[key]['sae_state']
    d_sae = state['encoder.weight'].shape[0]
    d_in  = state['encoder.weight'].shape[1]
    sae   = AutoEncoderTopK(d_in, d_sae, k=_infer_k(d_in, state))
    sae.load_state_dict(state)
    sae.eval().to(device)
    return sae

def reconstruct_mask(bank_data, task_id, layer):
    key = (task_id, layer)
    if key not in bank_data:
        return None
    codes = torch.tensor(bank_data[key]['sparse_codes'])

    mask = (codes.abs() > 0).any(0)
    return mask

@torch.no_grad()
def collect_all_acts(snap_path, loader, dataset, device):
    ckpt = torch.load(snap_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']

    if 'fc.weight' in sd:
        n_classes = int(sd['fc.weight'].shape[0])
    else:
        n_classes = int(ckpt.get('seen_classes', 100))

    model = build_resnet18(dataset, num_classes=n_classes)
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    bufs = {l: [] for l in HOOK_LAYERS}
    labs = []
    for imgs, labels in loader:
        model(imgs.to(device))
        for l in HOOK_LAYERS:
            bufs[l].append(model.hooks[l].cpu())
        labs.append(labels)
    model.to('cpu'); del model

    return {l: torch.cat(v) for l, v in bufs.items()}, torch.cat(labs)

def classify_fates_from_bank(bank_data, snap_dir, scenario,
                              dataset, seed, device='cuda'):
    from scipy.stats import spearmanr

    results = {}
    snap_final = os.path.join(snap_dir, 'task9.pt')

    num_tasks        = scenario.num_tasks
    classes_per_task = scenario.classes_per_task

    for task_id in range(num_tasks):
        results[task_id] = {}

        act_loader = scenario.get_activation_loader(task_id, max_samples=2000)
        act_dict, labels = collect_all_acts(snap_final, act_loader, dataset, device)
        labels_dev = labels.to(device)
        task_start = task_id * classes_per_task

        for layer in HOOK_LAYERS:
            sae  = reconstruct_sae(bank_data, task_id, layer, device)
            mask = reconstruct_mask(bank_data, task_id, layer)
            if sae is None or mask is None:
                continue

            mask_dev   = mask.to(device)
            active_idx = mask_dev.nonzero(as_tuple=True)[0]
            n_active   = int(active_idx.numel())

            acts_now  = act_dict[layer].to(device)
            codes_now = sae.encode(acts_now)

            stored_codes = torch.tensor(
                bank_data[(task_id, layer)]['sparse_codes']).to(device)

            survived, drifted, dead = 0, 0, 0
            per_class = {}

            for c_local in range(classes_per_task):
                c_global = task_start + c_local
                mask_c   = (labels_dev == c_global)
                if mask_c.sum() == 0:
                    continue

                current_mean = codes_now[mask_c].mean(0)
                stored_code  = stored_codes[c_local]

                cur_sel = current_mean[active_idx]
                sto_sel = stored_code[active_idx]

                orig_norm  = sto_sel.norm().item()
                act_ratio  = (cur_sel.norm() / (orig_norm + 1e-8)).item()

                cos_sim = F.cosine_similarity(
                    cur_sel.unsqueeze(0), sto_sel.unsqueeze(0)).item()

                if n_active >= 5:
                    rho, _ = spearmanr(
                        cur_sel.detach().cpu().numpy(), sto_sel.detach().cpu().numpy())
                    rank_corr = float(rho) if np.isfinite(rho) else None
                else:
                    rank_corr = None

                if act_ratio < ACT_FRAC:
                    fate = 'dead';     dead     += 1
                elif cos_sim < LO_COS:
                    fate = 'drifted';  drifted  += 1
                else:
                    fate = 'survived'; survived += 1

                per_class[c_global] = {
                    'fate': fate, 'cos_sim': round(cos_sim, 4),
                    'rank_corr': round(rank_corr, 4) if rank_corr else None,
                    'act_ratio': round(act_ratio, 4), 'n_active': n_active,
                }

            results[task_id][layer] = {
                'survived': survived, 'drifted': drifted,
                'dead': dead, 'total': survived + drifted + dead,
                'n_active': n_active, 'per_class': per_class,
            }
            sae.to('cpu')

        print(f"  task {task_id:2d} done | "
              + " | ".join(
                  f"{l}: {results[task_id].get(l,{}).get('survived',0)}S "
                  f"{results[task_id].get(l,{}).get('drifted',0)}Dr "
                  f"{results[task_id].get(l,{}).get('dead',0)}D"
                  for l in HOOK_LAYERS))

    return results

def fate_summary(results):
    by_layer = defaultdict(lambda: {'survived':0,'drifted':0,'dead':0,'total':0})
    cos_vals  = defaultdict(list)
    rank_vals = defaultdict(list)
    for _, layers in results.items():
        for layer, stats in layers.items():
            for k in ('survived','drifted','dead','total'):
                by_layer[layer][k] += stats[k]
            for info in stats.get('per_class', {}).values():
                cos_vals[layer].append(info['cos_sim'])
                if info.get('rank_corr'):
                    rank_vals[layer].append(info['rank_corr'])
    summary = {}
    for layer, counts in by_layer.items():
        t = max(counts['total'], 1)
        summary[layer] = {
            'survived_pct':   round(100*counts['survived']/t, 1),
            'drifted_pct':    round(100*counts['drifted']/t,  1),
            'dead_pct':       round(100*counts['dead']/t,     1),
            'total':          counts['total'],
            'mean_cos_sim':   round(float(np.mean(cos_vals[layer])), 4),
            'mean_rank_corr': round(float(np.mean(rank_vals[layer])), 4)
                              if rank_vals[layer] else None,
        }
    return summary

def run(results_dir, dataset, seed, data_root, num_tasks=10, device='cuda', method='sae_fsgp'):
    from data.cifar_cl import ClassIncrementalScenario

    scenario = ClassIncrementalScenario(
        dataset=dataset, num_tasks=num_tasks,
        data_root=data_root, seed=42)

    bank_path = os.path.join(results_dir, 'analysis',
                             f'{dataset}_t{num_tasks}_s{seed}',
                             f'{method}', 'posthoc_sae', 'sae_bank.pt')
    bank_data = torch.load(bank_path, map_location='cpu', weights_only=False)

    snap_dir = os.path.join(results_dir, 'snapshots',
                            f'{method}_{dataset}_t{num_tasks}_s{seed}')

    print(f"\n[FeatureFate] {dataset} T={num_tasks} seed={seed}")
    print("="*60)
    print(f"{'Layer':<8} {'Survived':>10} {'Drifted':>10} {'Dead':>8}")
    print("-"*40)

    results = classify_fates_from_bank(
        bank_data, snap_dir, scenario, dataset, seed, device)

    summary = fate_summary(results)
    for layer in HOOK_LAYERS:
        if layer in summary:
            s = summary[layer]
            print(f"{layer:<8} {s['survived_pct']:>9.1f}% "
                  f"{s['drifted_pct']:>9.1f}% {s['dead_pct']:>7.1f}%  "
                  f"(cos={s['mean_cos_sim']:.3f})")

    out_dir = os.path.join(results_dir, 'analysis',
                           f'{dataset}_t{num_tasks}_s{seed}', method)
    os.makedirs(out_dir, exist_ok=True)

    serialisable = {}
    for tid, layers in results.items():
        serialisable[str(tid)] = {}
        for layer, stats in layers.items():
            serialisable[str(tid)][layer] = {
                k: v for k, v in stats.items() if k != 'per_class'}
            serialisable[str(tid)][layer]['per_class'] = {
                str(cg): v for cg, v in stats.get('per_class', {}).items()}

    with open(os.path.join(out_dir, 'feature_fate.json'), 'w') as f:
        json.dump(serialisable, f, indent=2)
    with open(os.path.join(out_dir, 'fate_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved → {out_dir}/fate_summary.json")
    _plot_fate_summary(summary, dataset, seed,
                   os.path.join(results_dir, 'figures'), method)
    return summary

def _plot_fate_summary(summary, dataset, seed, fig_dir, method):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    layers = [l for l in HOOK_LAYERS if l in summary]
    surv   = [summary[l]['survived_pct'] for l in layers]
    drift  = [summary[l]['drifted_pct']  for l in layers]
    dead   = [summary[l]['dead_pct']     for l in layers]

    x = np.arange(len(layers))
    w = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, surv,  w, label='Survived', color='#2ca02c', alpha=0.85)
    ax.bar(x,     drift, w, label='Drifted',  color='#ff7f0e', alpha=0.85)
    ax.bar(x + w, dead,  w, label='Dead',     color='#d62728', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('layer', 'L') for l in layers], fontsize=11)
    ax.set_ylabel('Percentage of class-level SAE latents (%)', fontsize=10)
    ax.set_title(f'Feature Fate by Layer — {method}, {dataset.upper()}, seed {seed}',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)

    for i, layer in enumerate(layers):
        cos = summary[layer].get('mean_cos_sim', 0)
        ax.text(x[i], max(surv[i], drift[i], dead[i]) + 2.5,
                f'cos={cos:.2f}', ha='center', fontsize=7, color='#444')

    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)

    out_path = os.path.join(fig_dir, f'fig1_fate_{method}_{dataset}_s{seed}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Figure saved → {out_path}")
    plt.close()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--dataset',     default='cifar100')
    p.add_argument('--seed',        type=int, default=0)
    p.add_argument('--data_root',   default='/path/to/')
    p.add_argument('--num_tasks',   type=int, default=10)
    p.add_argument('--device',      default='cuda')

    p.add_argument('--method', default='sae_fsgp',
               choices=['finetune', 'ewc', 'lwf', 'fecam', 'sae_fsgp', 'dense_fsgp', 'efc', 'dsal'])

    args = p.parse_args()
    run(args.results_dir, args.dataset, args.seed,
        args.data_root, args.num_tasks, args.device, args.method)
