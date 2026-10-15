import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.cifar_cl         import ClassIncrementalScenario
from models.resnet_hook    import build_resnet18

from analysis.posthoc_sae   import (train_posthoc_saes, load_posthoc_bank,
                                     _compute_discriminativity, collect_activations,
                                     load_snapshot)
from analysis.method_compare import (build_interference_matrix, load_R_matrix,
                                      worst_interference_pairs,
                                      sae_discriminativity_per_unit,
                                      ewc_sae_alignment,
                                      save_comparison_results)

METHODS = ['finetune', 'ewc', 'lwf', 'fecam', 'sae_fsgp', 'dense_fsgp', "dsal","efc"]

def find_result_json(results_dir, method, dataset, T, seed):
    p = os.path.join(results_dir, f"{method}_{dataset}_t{T}_s{seed}.json")
    return p if os.path.exists(p) else None

def find_snapshot_dir(results_dir, method, dataset, T, seed):

    tagged = os.path.join(results_dir, 'snapshots',
                          f"{method}_{dataset}_t{T}_s{seed}")
    if os.path.isdir(tagged) and any(
            f.endswith('.pt') for f in os.listdir(tagged)):
        return tagged
    return None

@torch.no_grad()
def compute_posthoc_drift(snap_dir: str, scenario, dataset: str,
                           device: str = 'cuda') -> Dict:
    from scipy.stats import spearmanr

    snap_paths = sorted(Path(snap_dir).glob('task*.pt'),
                        key=lambda p: int(p.stem.replace('task', '')))
    if len(snap_paths) < 2:
        return {}

    final_snap = snap_paths[-1]
    T          = len(snap_paths)
    LAYERS     = ['layer1', 'layer2', 'layer3', 'layer4']

    model_final = load_snapshot(str(final_snap), dataset, device)
    model_final.eval()

    layer_cos, layer_rc = {l: [] for l in LAYERS}, {l: [] for l in LAYERS}

    for t, snap_path in enumerate(snap_paths[:-1]):
        model_t = load_snapshot(str(snap_path), dataset, device)
        model_t.eval()

        if hasattr(scenario, 'get_activation_loader'):
            loader_t = scenario.get_activation_loader(t, max_samples=1000)
        else:
            loader_t, _ = scenario.get_task_loaders(t)

        acts_t,   labels = collect_activations(model_t,     loader_t, device)
        acts_fin, _      = collect_activations(model_final, loader_t, device)

        classes_per_task = scenario.classes_per_task
        task_start       = t * classes_per_task
        labels_cpu       = labels

        for layer in LAYERS:
            cos_vals, rc_vals = [], []
            for c_local in range(classes_per_task):
                c_global = task_start + c_local
                mask     = (labels_cpu == c_global)
                if mask.sum() < 2:
                    continue
                mu_t   = acts_t[layer][mask].float().mean(0)
                mu_fin = acts_fin[layer][mask].float().mean(0)

                cos_sim  = F.cosine_similarity(
                    mu_t.unsqueeze(0), mu_fin.unsqueeze(0)).item()
                rc, _    = spearmanr(
                    mu_t.cpu().numpy(), mu_fin.cpu().numpy())

                cos_vals.append(cos_sim)
                rc_vals.append(float(rc) if np.isfinite(rc) else 0.0)

            if cos_vals:
                layer_cos[layer].append(np.mean(cos_vals))
                layer_rc[layer].append(np.mean(rc_vals))

        model_t.to('cpu'); del model_t

    model_final.to('cpu')

    result = {}
    for layer in LAYERS:
        if layer_cos[layer]:
            result[layer] = {
                'mean_cos_sim':  round(float(np.mean(layer_cos[layer])), 4),
                'mean_rank_corr': round(float(np.mean(layer_rc[layer])),  4),
            }
    return result

def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"  SAE-FSGP Mechanistic Analysis")
    print(f"  dataset={args.dataset}  T={args.num_tasks}  seed={args.seed}")
    print(f"  device={device}")
    print(f"{'='*60}")

    tag = f"{args.dataset}_t{args.num_tasks}_s{args.seed}"
    analysis_dir = os.path.join(args.results_dir, 'analysis', tag)
    os.makedirs(analysis_dir, exist_ok=True)

    scenario = ClassIncrementalScenario(
        dataset     = args.dataset,
        num_tasks   = args.num_tasks,
        data_root   = args.data_root,
        seed        = 42,
        batch_size  = 256,
        num_workers = args.num_workers,
    )

    report_path = os.path.join(analysis_dir, 'analysis_report.json')
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        report.setdefault('methods', {})
        print(f"  Loaded existing report ({len(report['methods'])} methods already present)")
    else:
        report = {'methods': {}}
    report['dataset']   = args.dataset
    report['num_tasks'] = args.num_tasks
    report['seed']      = args.seed

    print("\n[1/4] Interference matrices …")
    for method in args.methods:
        rj = find_result_json(args.results_dir, method,
                              args.dataset, args.num_tasks, args.seed)
        if rj is None:
            continue
        R   = load_R_matrix(rj)
        I   = build_interference_matrix(R)
        out = os.path.join(analysis_dir, method)
        os.makedirs(out, exist_ok=True)
        save_comparison_results(I, None, out)
        avg_inter = float(np.nanmean(I[~np.isnan(I)]))
        report['methods'].setdefault(method, {})['avg_interference'] = avg_inter
        print(f"  {method:<14}: avg interference = {avg_inter:.2f}%")

    print("\n[2/4] Post-hoc SAE training on backbone snapshots …")
    for method in args.methods:
        snap_dir = find_snapshot_dir(args.results_dir, method,
                                      args.dataset, args.num_tasks, args.seed)
        if snap_dir is None:
            print(f"  {method:<14}: no snapshots found, skipping")
            continue
        print(f"  {method:<14}: snapshots at {snap_dir}")
        ph_out = os.path.join(analysis_dir, method)
        train_posthoc_saes(
            snap_dir   = snap_dir,
            result_dir = ph_out,
            scenario   = scenario,
            dataset    = args.dataset,
            device     = device,
            expansion  = args.sae_expansion,
            epochs     = args.sae_epochs,
            verbose    = True,
        )
        report['methods'].setdefault(method, {})['posthoc_sae'] = ph_out

    print("\n[3/4] Per-method backbone drift (from snapshots) …")

    drift_comparison = dict(report.get('drift_comparison', {}))
    for method in args.methods:
        snap_dir = find_snapshot_dir(args.results_dir, method,
                                      args.dataset, args.num_tasks, args.seed)
        if snap_dir is None:
            print(f"  {method:<14}: no snapshots, skipping drift")
            continue

        interp_path = os.path.join(
            args.results_dir, 'interp',
            f"sae_fsgp_{args.dataset}_t{args.num_tasks}_seed{args.seed}",
            'drift_by_layer.json')
        if method == 'sae_fsgp' and os.path.exists(interp_path):
            with open(interp_path) as f:
                raw = json.load(f)

            normalised = {}
            for layer, stats in raw.items():
                normalised[layer] = {
                    'mean_cos_sim':   stats.get('cos_sim', stats.get('mean_cos_sim', float('nan'))),
                    'mean_rank_corr': stats.get('rank_corr', stats.get('mean_rank_corr', float('nan'))),
                }
            drift_comparison[method] = normalised
            print(f"  {method:<14}: loaded from existing interp data")
            continue

        print(f"  {method:<14}: computing drift from {Path(snap_dir).name} …")
        try:
            drift = compute_posthoc_drift(snap_dir, scenario, args.dataset, device)
            if drift:
                drift_comparison[method] = drift

                out_path = os.path.join(analysis_dir, method, 'drift_by_layer.json')
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, 'w') as f:
                    json.dump(drift, f, indent=2)
        except Exception as e:
            print(f"    Failed: {e}")

    if drift_comparison:
        print(f"\n  Layer4 cosine similarity (↑ = less backbone drift):")
        print(f"  {'Method':<16} {'cos_sim':>10} {'rank_corr':>12}")
        for m, dl in drift_comparison.items():
            if 'layer4' in dl:
                cs = dl['layer4'].get('mean_cos_sim',   float('nan'))
                rc = dl['layer4'].get('mean_rank_corr', float('nan'))
                print(f"  {m:<16} {cs:>10.4f} {rc:>12.4f}")
    report['drift_comparison'] = drift_comparison

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✓ Full report saved → {report_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir',   default='./results')
    p.add_argument('--dataset',       default='cifar100')
    p.add_argument('--num_tasks',     type=int, default=10)
    p.add_argument('--seed',          type=int, default=0)
    p.add_argument('--data_root',     default='./data_cache')
    p.add_argument('--num_workers',   type=int, default=4)
    p.add_argument('--sae_expansion', type=int, default=8)
    p.add_argument('--sae_epochs',    type=int, default=20)
    p.add_argument('--methods', nargs='+', default=METHODS,
                  help='Subset of methods to (re)process. Defaults to all 6. '
                       'Scoping this to just the methods you actually changed '
                       'avoids redoing posthoc SAE training for the others -- '
                       'the report/drift files now merge with what is already '
                       'on disk, so a scoped run will not delete other '
                       'methods\' previously-computed entries.')
    return p.parse_args()

if __name__ == '__main__':
    run(parse_args())
