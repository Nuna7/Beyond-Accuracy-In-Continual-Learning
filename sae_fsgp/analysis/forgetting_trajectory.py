"""
forgetting_trajectory_fixed.py — fixes BUG 1 (wrong latent selection).
Root cause of the original bug: `discrim.topk(n_top)` selected latents by
GLOBAL discriminativity score, which is unrelated to which latents are
actually non-zero in a SPECIFIC class's stored prototype code. For a
TopK-SAE (k=32), a mean class code has ~32 non-zero entries out of
d_sae total; picking by global score almost always lands on latents that
are zero for the target class, producing the "all activations are 0.0"
trajectories seen in the original output.
Fix: select the top-k latents directly from the STORED PROTOTYPE CODE
for the target class. This guarantees non-zero activation at t=0 by
construction (we are selecting from a vector that is non-zero exactly
where the SAE assigned mass for this class).
Also adds a second target class (a "stable" class from a LATE task, not
just any task-0 class) so the figure shows a genuine forgotten-vs-stable
contrast within the same plot.
Usage (on KutumLabGPU):
  python analysis/forgetting_trajectory_fixed.py \
      --results_dir ./results --dataset cifar100 --seed 0 \
      --data_root /media/20TB/vision/CL/data_cache
"""

import os, sys, json, torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.resnet_hook import build_resnet18
from sae.compat import AutoEncoderTopK

TARGET_LAYER = 'layer4'
N_TOP_LATENTS = 8 # fewer lines -> cleaner figure

@torch.no_grad()
def collect_l4(model, loader, device):
    bufs, labs = [], []
    for imgs, labels in loader:
        model(imgs.to(device))
        bufs.append(model.hooks[TARGET_LAYER].cpu())
        labs.append(labels)
    return torch.cat(bufs), torch.cat(labs)

def _infer_k(d_in, state):
    """Prefer the k actually used during training. If it was saved as a buffer
    (check: print(list(state.keys())) once -- the sae.k.item() pattern in your
    other scripts suggests it's a tensor, meaning load_state_dict should restore
    it, but this makes reconstruction correct either way). Falls back to the
    layer-dependent formula from sae_fsgp.py's _sae_cfg (topk_by_layer=True
    case) -- hardcoding k=32 for every layer was only ever correct for layer4."""
    if 'k' in state:
        k_val = state['k']
        return int(k_val.item()) if hasattr(k_val, 'item') else int(k_val)
    return max(8, d_in // 16)

def run(results_dir, dataset, seed, data_root, device='cuda', n_top=N_TOP_LATENTS, method='sae_fsgp', num_tasks=10):
    n_top = int(n_top)
    #classes_per_task = 10 if dataset == 'cifar100' else 20
    total_classes = 100 if dataset == 'cifar100' else 200
    classes_per_task = total_classes // num_tasks
    # ── load feature_fate to find forgotten + stable target classes ───────────
    fate_path = os.path.join(results_dir, 'analysis', f'{dataset}_t{num_tasks}_s{seed}', f'{method}', 'feature_fate.json')
    with open(fate_path) as f:
        fate_data = json.load(f)
    # Most forgotten task-0 class at layer4 (lowest act_ratio)
    task0_l4 = fate_data['0']['layer4']['per_class']
    forgotten_class = int(min(task0_l4, key=lambda c: task0_l4[c]['act_ratio']))
    forgotten_cos = task0_l4[str(forgotten_class)]['cos_sim']
    forgotten_act = task0_l4[str(forgotten_class)]['act_ratio']
    # Stable class from task 8 (recently trained, guaranteed alive)
    task8_l4 = fate_data['8']['layer4']['per_class']
    survived_t8 = [(int(c), v) for c, v in task8_l4.items()
                   if v['fate'] == 'survived' and v['act_ratio'] > 0.4]
    survived_t8.sort(key=lambda x: x[1]['cos_sim'], reverse=True)
    if survived_t8:
        stable_class = survived_t8[0][0]
        stable_cos = task8_l4[str(stable_class)]['cos_sim']
        stable_act = task8_l4[str(stable_class)]['act_ratio']
    else:
        stable_class = int(max(task8_l4, key=lambda c: task8_l4[c]['act_ratio']))
        stable_cos = task8_l4[str(stable_class)]['cos_sim']
        stable_act = task8_l4[str(stable_class)]['act_ratio']
    print(f"\n[TrajFix] {dataset} seed {seed}")
    print(f" Forgotten: class {forgotten_class} "
          f"(cos={forgotten_cos:.3f}, act_ratio={forgotten_act:.3f})")
    print(f" Stable: class {stable_class} "
          f"(cos={stable_cos:.3f}, act_ratio={stable_act:.3f})")
    # ── load SAE bank ─────────────────────────────────────────────────────────
    bank_path = os.path.join(results_dir, 'analysis', f'{dataset}_t{num_tasks}_s{seed}', f'{method}', 'posthoc_sae', 'sae_bank.pt')
    bank = torch.load(bank_path, map_location='cpu', weights_only=False)

    def get_sae(task_id):
        key = (task_id, TARGET_LAYER)
        state = bank[key]['sae_state']
        d_sae = state['encoder.weight'].shape[0]
        d_in = state['encoder.weight'].shape[1]
        sae = AutoEncoderTopK(d_in, d_sae, k=_infer_k(d_in, state))
        sae.load_state_dict(state)
        return sae.eval().to(device)

    def get_top_latents_from_proto(task_id, class_global):
        """KEY FIX: select latents from the STORED PROTOTYPE, not global discrim."""
        key = (task_id, TARGET_LAYER)
        codes = torch.tensor(bank[key]['sparse_codes']) # (C, d_sae)
        c_local = class_global - task_id * classes_per_task
        proto = codes[c_local]
        vals, idx = proto.abs().topk(n_top)
        nz = vals > 1e-6
        top_idx = idx[nz].tolist()
        if len(top_idx) < 3:
            top_idx = proto.abs().topk(n_top).indices.tolist()
        print(f" Latents for class {class_global}: {top_idx[:6]} "
              f"(max_act={float(vals[0]):.4f})")
        return top_idx, proto

    sae_t0 = get_sae(0)
    sae_t8 = get_sae(8)
    lats_forgotten, proto_forgotten = get_top_latents_from_proto(0, forgotten_class)
    lats_stable, proto_stable = get_top_latents_from_proto(8, stable_class)
    # ── load data scenario ───────────────────────────────────────────────────
    from data.cifar_cl import ClassIncrementalScenario
    #scenario = ClassIncrementalScenario(dataset=dataset, num_tasks=10,
    #                                    data_root=data_root, seed=42)
    # snap_dir = os.path.join(results_dir, 'snapshots',
    #                         f'{method}_{dataset}_t10_s{seed}')
    scenario = ClassIncrementalScenario(dataset=dataset, num_tasks=num_tasks, data_root=data_root, seed=42)
    snap_dir = os.path.join(results_dir, 'snapshots', f'{method}_{dataset}_t{num_tasks}_s{seed}')
    
    snap_paths = sorted(Path(snap_dir).glob('task*.pt'),
                        key=lambda p: int(p.stem.replace('task', '')))

    # def load_model(snap_path):
    #     ckpt = torch.load(str(snap_path), map_location='cpu', weights_only=False)
    #     model = build_resnet18(dataset, num_classes=ckpt['seen_classes'])
    #     model.load_state_dict(ckpt['state_dict'])
    #     return model.eval().to(device)

    def load_model(snap_path):
        ckpt = torch.load(str(snap_path), map_location='cpu', weights_only=False)
        sd = ckpt['state_dict']
        if 'fc.weight' in sd:
            n_classes = int(sd['fc.weight'].shape[0])
        else:
            n_classes = int(ckpt.get('seen_classes', 100))
        model = build_resnet18(dataset, num_classes=n_classes)
        model.load_state_dict(sd, strict=False)
        return model.eval().to(device)

    traj_f, traj_s = {}, {}
    for snap_path in snap_paths:
        t = int(snap_path.stem.replace('task', ''))
        model = load_model(snap_path)
        # Forgotten class trajectory (task-0 SAE)
        loader_f = scenario.get_activation_loader(
            forgotten_class // classes_per_task, max_samples=1000)
        acts_f, labs_f = collect_l4(model, loader_f, device)
        mask_f = (labs_f == forgotten_class)
        if mask_f.sum() > 0:
            mean_f = acts_f[mask_f].float().mean(0).to(device)
            code_f = sae_t0.encode(mean_f.unsqueeze(0)).squeeze(0)
            cos_f = F.cosine_similarity(
                code_f.unsqueeze(0), proto_forgotten.to(device).unsqueeze(0)).item()
            traj_f[t] = {str(l): float(code_f[l]) for l in lats_forgotten}
            traj_f[t]['__cos'] = cos_f
        # Stable class trajectory (task-8 SAE)
        stable_task = stable_class // classes_per_task
        loader_s = scenario.get_activation_loader(stable_task, max_samples=1000)
        acts_s, labs_s = collect_l4(model, loader_s, device)
        mask_s = (labs_s == stable_class)
        if mask_s.sum() > 0:
            mean_s = acts_s[mask_s].float().mean(0).to(device)
            code_s = sae_t8.encode(mean_s.unsqueeze(0)).squeeze(0)
            cos_s = F.cosine_similarity(
                code_s.unsqueeze(0), proto_stable.to(device).unsqueeze(0)).item()
            traj_s[t] = {str(l): float(code_s[l]) for l in lats_stable}
            traj_s[t]['__cos'] = cos_s
        model.to('cpu'); del model
        print(f" t={t}: forgotten_cos={traj_f.get(t,{}).get('__cos',0):.3f} "
              f"stable_cos={traj_s.get(t,{}).get('__cos',0):.3f}")
    # ── save ─────────────────────────────────────────────────────────────────
    out = {
        'forgotten_class': forgotten_class, 'stable_class': stable_class,
        'lats_forgotten': lats_forgotten, 'lats_stable': lats_stable,
        'traj_forgotten': {str(t): {str(k): v for k, v in d.items()}
                           for t, d in traj_f.items()},
        'traj_stable': {str(t): {str(k): v for k, v in d.items()}
                           for t, d in traj_s.items()},
    }
    out_dir = os.path.join(results_dir, 'analysis', 'forgetting_trajectory',
                            f'{method}_{dataset}_t{num_tasks}_s{seed}')
                           #f'{method}_{dataset}_t10_s{seed}')
        
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{method}_{dataset}_s{seed}_fixed.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f" Stats saved -> {out_dir}/{method}_{dataset}_s{seed}_fixed.json")
    _plot(out, dataset, seed, os.path.join(results_dir, 'figures'), method)
    return out
    
def _plot(data, dataset, seed, fig_dir, method):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    COLOURS = {'forgotten': '#d62728', 'stable': '#2ca02c'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f'Per-latent forgetting trajectory — {dataset.upper()} seed {seed}',
                 fontsize=11)
    tasks = sorted(int(t) for t in data['traj_forgotten'])
    for ax, key, lats, colour, label_prefix in [
        (axes[0], 'traj_forgotten', data['lats_forgotten'],
         COLOURS['forgotten'], f"Forgotten (class {data['forgotten_class']})"),
        (axes[1], 'traj_stable', data['lats_stable'],
         COLOURS['stable'], f"Stable (class {data['stable_class']})"),
    ]:
        traj = data[key]
        t_has_cos = [t for t in tasks if str(t) in traj]
        cos_vals = [traj[str(t)].get('__cos', 0) for t in t_has_cos]
        used_label = False
        for lat_idx in lats:
            vals = [traj.get(str(t), {}).get(str(lat_idx), 0) for t in tasks]
            lbl = 'Individual SAE latents' if not used_label else '_nolegend_'
            used_label = True
            ax.plot(tasks, vals, color=colour, alpha=0.45, linewidth=1.4, label=lbl)
        ax2 = ax.twinx()
        ax2.plot(t_has_cos, cos_vals, 'k--', linewidth=2.2,
                 label='cos_sim to stored prototype')
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel('cos_sim to prototype', fontsize=8)
        ax2.tick_params(axis='y', labelsize=8)
        ax.set_xlabel('Task checkpoint t', fontsize=9)
        ax.set_ylabel('SAE latent activation', fontsize=9)
        if cos_vals:
            ax.set_title(f'{label_prefix}\n'
                         f'cos_sim: t0={cos_vals[0]:.3f} -> t9={cos_vals[-1]:.3f}',
                         fontsize=9, color=colour)
        ax.set_xticks(tasks)
        ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, f'fig_trajectory_{method}_{dataset}_s{seed}_fixed.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f' Figure -> {out_path}')
    plt.close()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='./results_v2')
    p.add_argument('--dataset', default='cifar100')
    p.add_argument('--num_tasks', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data_root', default='/media/20TB/vision/CL/data_cache')
    p.add_argument('--device', default='cuda')
    #p.add_argument('--method', default='sae_fsgp', choices=['sae_fsgp','dense_fsgp'])
    p.add_argument('--method', default='sae_fsgp',
               choices=['finetune', 'ewc', 'lwf', 'fecam', 'sae_fsgp', 'dense_fsgp', 'dsal', 'efc', 'ada_gauss'])
    args = p.parse_args()
    run(args.results_dir, args.dataset, args.seed, args.data_root, args.device, N_TOP_LATENTS,
      args.method, args.num_tasks)