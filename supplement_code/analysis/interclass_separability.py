import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.cifar_cl         import ClassIncrementalScenario
from analysis.posthoc_sae  import collect_activations, load_snapshot

def find_snapshot_dir(results_dir, method, dataset, T, seed):
    tagged = os.path.join(results_dir, 'snapshots', f"{method}_{dataset}_t{T}_s{seed}")
    if os.path.isdir(tagged) and any(f.endswith('.pt') for f in os.listdir(tagged)):
        return tagged
    return None

@torch.no_grad()
def build_final_prototypes(snap_dir, scenario, dataset, device, layer='layer4'):
    snap_paths = sorted(Path(snap_dir).glob('task*.pt'),
                        key=lambda p: int(p.stem.replace('task', '')))
    if not snap_paths:
        return {}
    final_snap = snap_paths[-1]
    model = load_snapshot(str(final_snap), dataset, device)
    model.eval()

    num_tasks = scenario.num_tasks
    prototypes = {}
    for t in range(num_tasks):
        if hasattr(scenario, 'get_activation_loader'):
            loader = scenario.get_activation_loader(t, max_samples=2000)
        else:
            loader, _ = scenario.get_task_loaders(t)

        acts, labels = collect_activations(model, loader, device)
        feats = acts[layer].float()
        for c in labels.unique():
            mask = (labels == c)
            mu = feats[mask].mean(0)
            prototypes[int(c)] = F.normalize(mu, dim=0).cpu()

    model.to('cpu'); del model
    return prototypes

def analyse_separability(prototypes: dict, top_k_pairs: int = 5):
    classes = sorted(prototypes.keys())
    P = torch.stack([prototypes[c] for c in classes])
    sim = P @ P.T
    C = sim.shape[0]

    off_diag_mask = ~torch.eye(C, dtype=torch.bool)
    off_diag_vals = sim[off_diag_mask]

    sim_no_diag = sim.clone()
    sim_no_diag.fill_diagonal_(-2.0)
    nearest_confuser_sim, nearest_confuser_idx = sim_no_diag.max(dim=1)

    P_centered = P - P.mean(dim=0, keepdim=True)
    svals = torch.linalg.svdvals(P_centered)
    p = (svals ** 2) / ((svals ** 2).sum() + 1e-12)
    effective_rank = float(1.0 / (p ** 2).sum())

    triu_idx = torch.triu_indices(C, C, offset=1)
    triu_vals = sim[triu_idx[0], triu_idx[1]]
    top_vals, top_pos = torch.topk(triu_vals, min(top_k_pairs, triu_vals.numel()))
    worst_pairs = [
        {'class_a': classes[triu_idx[0][i]], 'class_b': classes[triu_idx[1][i]],
         'cos_sim': float(top_vals[j])}
        for j, i in enumerate(top_pos)
    ]

    return {
        'num_classes':            C,
        'nominal_dim':             int(P.shape[1]),
        'mean_offdiag_sim':        float(off_diag_vals.mean()),
        'std_offdiag_sim':         float(off_diag_vals.std()),
        'max_offdiag_sim':         float(off_diag_vals.max()),
        'mean_nearest_confuser':   float(nearest_confuser_sim.mean()),
        'worst_nearest_confuser':  float(nearest_confuser_sim.max()),
        'effective_rank':          effective_rank,
        'worst_pairs':             worst_pairs,
    }

def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scenario = ClassIncrementalScenario(
        dataset=args.dataset, num_tasks=args.num_tasks,
        data_root=args.data_root, seed=42,
        batch_size=256, num_workers=args.num_workers)

    tag = f"{args.dataset}_t{args.num_tasks}_s{args.seed}"
    out_dir = os.path.join(args.results_dir, 'analysis', tag, 'separability')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"interclass_separability_seed{args.seed}.json")

    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"Loaded existing separability file ({len(results)} methods already present)")
    else:
        results = {}

    print(f"\n{'Method':<12}{'mean off-diag':>15}{'worst confuser':>16}"
          f"{'eff. rank':>12}{'nominal dim':>13}")
    print("-" * 68)
    for method in args.methods:
        snap_dir = find_snapshot_dir(args.results_dir, method,
                                     args.dataset, args.num_tasks, args.seed)
        if snap_dir is None:
            print(f"{method:<12}  (no snapshots found, skipped)")
            continue

        protos = build_final_prototypes(snap_dir, scenario, args.dataset, device)
        if not protos:
            print(f"{method:<12}  (no prototypes built, skipped)")
            continue
        stats = analyse_separability(protos)
        results[method] = stats

        print(f"{method:<12}{stats['mean_offdiag_sim']:>15.4f}"
              f"{stats['worst_nearest_confuser']:>16.4f}"
              f"{stats['effective_rank']:>12.1f}"
              f"{stats['nominal_dim']:>13d}")

    print(f"\n{'Method':<12}Worst confusable class pairs (top 3)")
    print("-" * 68)
    for method, stats in results.items():
        pairs_str = ", ".join(
            f"({p['class_a']},{p['class_b']})={p['cos_sim']:.3f}"
            for p in stats['worst_pairs'][:3])
        print(f"{method:<12}{pairs_str}")

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")
    return results

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--dataset',     default='cifar100')
    p.add_argument('--num_tasks',   type=int, default=10)
    p.add_argument('--seed',        type=int, default=0)
    p.add_argument('--data_root',   default='./data_cache')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--methods', nargs='+',
                   default=['finetune', 'ewc', 'lwf', 'fecam', 'sae_fsgp', 'dense_fsgp', "dsal","efc"])
    return p.parse_args()

if __name__ == '__main__':
    run(parse_args())

