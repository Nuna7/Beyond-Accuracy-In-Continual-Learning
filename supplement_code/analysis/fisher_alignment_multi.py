import os, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from models.resnet_hook import build_resnet18
from analysis.method_compare import (ewc_fisher_per_output_unit,
                                     sae_discriminativity_per_unit,
                                     ewc_sae_alignment)

def _num_classes_from_ckpt(ckpt):
    return ckpt['state_dict']['fc.weight'].shape[0]

class _FisherHolder:
    def __init__(self, fisher_dict):
        self._fisher = fisher_dict

def compute_fisher_diagonal(model, loader, device, n_samples=5000):
    model.eval()
    fisher = {n: torch.zeros_like(p)
             for n, p in model.named_parameters()
             if p.requires_grad and 'fc' not in n}
    seen = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        model.zero_grad()
        logits = model(imgs)
        probs = F.softmax(logits.detach(), dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(1)
        loss = F.cross_entropy(logits, sampled)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and n in fisher:
                fisher[n] += p.grad.detach().pow(2) * imgs.size(0)
        seen += imgs.size(0)
        if seen >= n_samples:
            break
    for n in fisher:
        fisher[n] /= max(seen, 1)
    model.zero_grad()
    return fisher

def run_multi_fisher_alignment(results_dir, dataset, seed, data_root, methods,
                               num_tasks=10, device='cuda'):
    from data.cifar_cl import ClassIncrementalScenario

    scenario = ClassIncrementalScenario(dataset=dataset, num_tasks=num_tasks,
                                        data_root=data_root, seed=42,
                                        batch_size=256)
    tag = f'{dataset}_t{num_tasks}_s{seed}'
    analysis_dir = os.path.join(results_dir, 'analysis', tag)

    fisher_loader = scenario.get_cumulative_test_loader(num_tasks - 1)

    summary = {}
    raw_arrays = {}

    for method in methods:
        snap_dir = os.path.join(results_dir, 'snapshots',
                                f'{method}_{dataset}_t{num_tasks}_s{seed}')
        bank_path = os.path.join(analysis_dir, method, 'posthoc_sae', 'sae_bank.pt')
        if not os.path.isdir(snap_dir) or not os.path.exists(bank_path):
            print(f"  {method:<12}: missing snapshots or SAE bank, skipping")
            continue

        snap_paths = sorted(Path(snap_dir).glob('task*.pt'),
                            key=lambda p: int(p.stem.replace('task', '')))
        if not snap_paths:
            continue
        final_snap = snap_paths[-1]
        ckpt = torch.load(str(final_snap), map_location='cpu', weights_only=False)
        model = build_resnet18(dataset, num_classes=_num_classes_from_ckpt(ckpt))
        model.load_state_dict(ckpt['state_dict'])
        model.to(device)

        print(f"  {method:<12}: computing fresh Fisher diagonal on final backbone …")
        fisher = compute_fisher_diagonal(model, fisher_loader, device)
        holder = _FisherHolder(fisher)
        fisher_per_unit = ewc_fisher_per_output_unit(holder)

        bank = torch.load(bank_path, map_location='cpu', weights_only=False)
        sae_disc = sae_discriminativity_per_unit(bank, task_id=0)

        alignment = ewc_sae_alignment(fisher_per_unit, sae_disc)
        summary[method] = alignment

        raw_arrays[method] = {}
        for layer in fisher_per_unit:
            if layer in sae_disc:
                n = min(len(fisher_per_unit[layer]), len(sae_disc[layer]))
                raw_arrays[method][layer] = {
                    'fisher': fisher_per_unit[layer][:n].tolist(),
                    'sae_disc': sae_disc[layer][:n].tolist(),
                }

        model.to('cpu'); del model

        if alignment:
            mean_r = float(np.mean([v['spearman_r'] for v in alignment.values()]))
            print(f"    mean rho = {mean_r:.3f}  " +
                  " | ".join(f"{l}={s['spearman_r']:.3f}" for l, s in alignment.items()))

    out_path = os.path.join(analysis_dir, 'fisher_sae_alignment_multi.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    raw_path = os.path.join(analysis_dir, 'fisher_sae_alignment_multi_raw.json')
    with open(raw_path, 'w') as f:
        json.dump(raw_arrays, f)

    print(f"\n  Summary -> {out_path}")
    print(f"  Raw points (for plotting) -> {raw_path}")
    return summary

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--dataset', default='cifar100')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data_root', default='/path/to/')
    p.add_argument('--num_tasks', type=int, default=10)
    p.add_argument('--device', default='cuda')
    p.add_argument('--methods', nargs='+',
                   default=['finetune', 'ewc', 'lwf', 'fecam', 'sae_fsgp',
                           'dense_fsgp', 'efc', 'dsal'])
    args = p.parse_args()
    run_multi_fisher_alignment(args.results_dir, args.dataset, args.seed,
                               args.data_root, args.methods, args.num_tasks,
                               args.device)
