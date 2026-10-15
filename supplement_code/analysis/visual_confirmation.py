import os, sys, json, torch
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.resnet_hook import build_resnet18
from sae.compat import AutoEncoderTopK

TARGET_LAYER = 'layer4'
TOP_K_IMAGES = 8
N_TOP_LATENTS = TOP_K_IMAGES

@torch.no_grad()
def collect_with_images(model, loader, sae, device):
    all_imgs, all_labs, all_codes = [], [], []
    model.eval(); sae.eval()
    for imgs, labels in loader:
        model(imgs.to(device))
        acts = model.hooks[TARGET_LAYER].float()
        codes = sae.encode(acts)
        all_imgs.append(imgs.cpu())
        all_labs.append(labels.cpu())
        all_codes.append(codes.cpu())
    return torch.cat(all_imgs), torch.cat(all_labs), torch.cat(all_codes)

def _infer_k(d_in, state):
    if 'k' in state:
        k_val = state['k']
        return int(k_val.item()) if hasattr(k_val, 'item') else int(k_val)
    return max(8, d_in // 16)

def load_sae_from_bank(bank, task_id, layer, device):
    state = bank[(task_id, layer)]['sae_state']
    d_sae = state['encoder.weight'].shape[0]
    d_in = state['encoder.weight'].shape[1]
    sae = AutoEncoderTopK(d_in, d_sae, k=_infer_k(d_in, state))
    sae.load_state_dict(state)
    return sae.eval().to(device)

def run(results_dir, dataset, seed, data_root, device='cuda', n_top=N_TOP_LATENTS,
        method='sae_fsgp', num_tasks=10):
    n_images = N_TOP_LATENTS
    n_images = int(n_images)
    total_classes = 100 if dataset == 'cifar100' else 200
    classes_per_task = total_classes // num_tasks

    fate_path = os.path.join(results_dir, 'analysis', f'{dataset}_t{num_tasks}_s{seed}', f'{method}', 'feature_fate.json')
    with open(fate_path) as f:
        fate_data = json.load(f)

    task0_l4 = fate_data['0']['layer4']['per_class']
    forgotten_class = int(min(task0_l4, key=lambda c: task0_l4[c]['act_ratio']))
    fc = task0_l4[str(forgotten_class)]

    stable_class = stable_task_id = None
    for try_task in [8, 7, 6, 5]:
        tl4 = fate_data[str(try_task)]['layer4']['per_class']
        cands = [(int(c), v) for c, v in tl4.items()
                 if v['fate'] == 'survived' and v['act_ratio'] > 0.4]
        if cands:
            cands.sort(key=lambda x: x[1]['cos_sim'], reverse=True)
            stable_class = cands[0][0]
            stable_task_id = try_task
            sc = tl4[str(stable_class)]
            break
    if stable_class is None:

        tl4 = fate_data['9']['layer4']['per_class']
        stable_class = int(max(tl4, key=lambda c: tl4[c]['act_ratio']))
        stable_task_id = 9
        sc = tl4[str(stable_class)]
    print(f"\n[VisFixd] {dataset} seed {seed}")
    print(f" Forgotten: class {forgotten_class} task=0 "
          f"fate={fc['fate']} cos={fc['cos_sim']:.3f} act_ratio={fc['act_ratio']:.3f}")
    print(f" Stable: class {stable_class} task={stable_task_id} "
          f"fate={sc['fate']} cos={sc['cos_sim']:.3f} act_ratio={sc['act_ratio']:.3f}")

    bank_path = os.path.join(results_dir, 'analysis', f'{dataset}_t{num_tasks}_s{seed}', f'{method}', 'posthoc_sae', 'sae_bank.pt')
    bank = torch.load(bank_path, map_location='cpu', weights_only=False)
    sae_f = load_sae_from_bank(bank, 0, TARGET_LAYER, device)
    sae_s = load_sae_from_bank(bank, stable_task_id, TARGET_LAYER, device)

    def top_proto_latent(task_id, class_global):
        codes = torch.tensor(bank[(task_id, TARGET_LAYER)]['sparse_codes'])
        c_local = class_global - task_id * classes_per_task
        proto = codes[c_local]
        lat_idx = int(proto.abs().argmax())
        mag = float(proto.abs().max())
        return lat_idx, mag
    lat_f, mag_f = top_proto_latent(0, forgotten_class)
    lat_s, mag_s = top_proto_latent(stable_task_id, stable_class)
    print(f" Forgotten latent {lat_f}: proto_magnitude={mag_f:.4f}")
    print(f" Stable latent {lat_s}: proto_magnitude={mag_s:.4f}")

    snap_dir = os.path.join(results_dir, 'snapshots', f'{method}_{dataset}_t{num_tasks}_s{seed}')

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

    snap_t0 = Path(snap_dir) / 'task0.pt'
    snap_t9 = Path(snap_dir) / 'task9.pt'
    from data.cifar_cl import ClassIncrementalScenario
    scenario = ClassIncrementalScenario(dataset=dataset, num_tasks=num_tasks, data_root=data_root, seed=42)
    results = {}
    for label, class_id, task_id, sae, lat_idx, snap_early in [
        ('forgotten', forgotten_class, 0, sae_f, lat_f, snap_t0),
        ('stable', stable_class, stable_task_id, sae_s, lat_s, snap_t9),
    ]:
        loader = scenario.get_clean_activation_loader(
            class_id // classes_per_task, max_samples=5000)

        m_early = load_model(snap_early)
        imgs_all, labs_all, codes_early = collect_with_images(
            m_early, loader, sae, device)
        m_early.to('cpu'); del m_early

        m_final = load_model(snap_t9)
        _, _, codes_final = collect_with_images(
            m_final, loader, sae, device)
        m_final.to('cpu'); del m_final
        mask_c = (labs_all == class_id)
        imgs_c = imgs_all[mask_c]
        act_earl = codes_early[mask_c][:, lat_idx]
        act_fin = codes_final[mask_c][:, lat_idx]

        rank_earl = act_earl.argsort(descending=True)[:n_images]
        rank_fin = act_fin.argsort(descending=True)[:n_images]
        mean_t0 = float(act_earl[rank_earl].mean())
        mean_t9 = float(act_fin[rank_earl].mean())
        retained = mean_t9 / (mean_t0 + 1e-8)
        results[label] = {
            'class_id': class_id, 'task_id': task_id,
            'latent_idx': lat_idx,
            'mean_act_t0': mean_t0, 'mean_act_t9': mean_t9,
            'act_retained': round(retained, 4),
            'imgs_by_t0': imgs_c[rank_earl],
            'imgs_by_t9': imgs_c[rank_fin],
        }
        print(f" [{label}] class={class_id} lat={lat_idx} "
              f"t_early={mean_t0:.3f} → t9={mean_t9:.3f} "
              f"({retained*100:.0f}% retained)")

    out_dir = os.path.join(results_dir, 'analysis', 'visual_confirmation')
    os.makedirs(out_dir, exist_ok=True)
    stats = {k: {ky: v for ky, v in d.items()
                 if not ky.startswith('imgs')}
             for k, d in results.items()}

    stats_path = os.path.join(out_dir, f'{method}_{dataset}_s{seed}_stats_fixed.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f" Stats → {stats_path}")
    _plot(results, dataset, seed, out_dir, method)
    return stats

def _plot(data, dataset, seed, fig_dir, method):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if dataset == "cifar100":
        MEAN = (0.5071, 0.4867, 0.4408)
        STD  = (0.2675, 0.2565, 0.2761)
    else:
        MEAN = (0.4802, 0.4481, 0.3975)
        STD  = (0.2770, 0.2691, 0.2821)

    def denorm(t):
        for c, (m, s) in enumerate(zip(MEAN, STD)):
            t[c] = t[c] * s + m
        return t.clamp(0, 1)
    f = data['forgotten']
    s = data['stable']
    n = f['imgs_by_t0'].shape[0]
    f_ret_str = ('0% — LATENT SILENCED'
                 if f['act_retained'] < 0.02 else f"{f['act_retained']*100:.0f}%")
    row_meta = [
        (f['imgs_by_t0'], '#d62728',
         f"▲ FORGOTTEN class {f['class_id']} · latent {f['latent_idx']} · "
         f"top images @ task-0 snapshot (act={f['mean_act_t0']:.3f})"),
        (f['imgs_by_t9'], '#d62728',
         f" Same images rescored @ task-9 → "
         f"act={f['mean_act_t9']:.3f} ({f_ret_str} retained)"),
        (s['imgs_by_t0'], '#2ca02c',
         f"▼ STABLE class {s['class_id']} · latent {s['latent_idx']} (task {s['task_id']}) · "
         f"top images @ task-9 snapshot (act={s['mean_act_t0']:.3f})"),
        (s['imgs_by_t9'], '#2ca02c',
         f" Same images rescored @ task-9 → "
         f"act={s['mean_act_t9']:.3f} ({s['act_retained']*100:.0f}% retained)"),
    ]
    fig, axes = plt.subplots(4, n, figsize=(n * 1.8, 8))
    for row_idx, (imgs, col, row_title) in enumerate(row_meta):
        for col_idx in range(min(n, imgs.shape[0])):
            ax = axes[row_idx, col_idx]
            img = denorm(imgs[col_idx].clone())
            ax.imshow(img.permute(1, 2, 0).numpy(), interpolation='nearest')
            ax.axis('off')
        axes[row_idx, 0].set_ylabel(row_title, fontsize=7, color=col,
                                    rotation=90, labelpad=3, va='center')
    fig.suptitle(
        f"SAE Latent Visual Fate — {dataset.upper()} seed {seed}\n"
        f"FORGOTTEN class {f['class_id']}, latent {f['latent_idx']}: "
        f"activation {f['mean_act_t0']:.3f} → {f['mean_act_t9']:.3f} "
        f"({f_ret_str} retained)\n"
        f"STABLE class {s['class_id']}, latent {s['latent_idx']}: "
        f"activation {s['mean_act_t0']:.3f} → {s['mean_act_t9']:.3f} "
        f"({s['act_retained']*100:.0f}% retained)",
        fontsize=8.5)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = os.path.join(fig_dir, f'{method}_{dataset}_s{seed}_visual_fixed.png')
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f" Figure → {out_path}")
    plt.close()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--dataset', default='cifar100')
    p.add_argument('--num_tasks', type=int, default=10)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--data_root', default='/path/to')
    p.add_argument('--device', default='cuda')

    p.add_argument('--method', default='sae_fsgp',
               choices=['finetune', 'ewc', 'lwf', 'fecam', 'sae_fsgp', 'dense_fsgp', 'efc', 'dsal'])
    args = p.parse_args()
    run(args.results_dir, args.dataset, args.seed, args.data_root, args.device, N_TOP_LATENTS,
      args.method, args.num_tasks)
