import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from models.resnet_hook import build_resnet18
from sae.compat import AutoEncoderTopK

TARGET_LAYER = 'layer4'
HI_COS = 0.85
LO_COS = 0.50
ACT_FRAC = 0.30
DECODABLE_F1_MIN = 0.05

RIDGE_LAMBDA = 1e-2
FIT_FRAC = 0.8

N_PERMUTATIONS_PROBE = 10

def _num_classes_from_ckpt(ckpt):
    return ckpt['state_dict']['fc.weight'].shape[0]

def fit_linear_translation(X_source: torch.Tensor, X_target: torch.Tensor,
                           ridge: float = RIDGE_LAMBDA) -> Tuple[torch.Tensor, torch.Tensor]:
    N, d = X_source.shape
    X_aug = torch.cat([X_source, torch.ones(N, 1, device=X_source.device)], dim=1)
    A = X_aug.T @ X_aug + ridge * torch.eye(d + 1, device=X_source.device)
    B = X_aug.T @ X_target
    sol = torch.linalg.solve(A, B)
    W, b = sol[:-1, :], sol[-1, :]
    return W, b

def apply_translation(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return X @ W + b

def train_linear_probe(X_train, y_train, X_val, y_val, device, epochs=200, lr=0.05):
    if y_train.sum() == 0 or y_train.sum() == y_train.numel():

        return 0.0

    d = X_train.shape[1]
    probe = nn.Linear(d, 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    y_train_f = y_train.float().to(device)

    for _ in range(epochs):
        opt.zero_grad()
        logits = probe(X_train).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_train_f)
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = (torch.sigmoid(probe(X_val).squeeze(-1)) > 0.5).float().cpu()
        y_val_f = y_val.float().cpu()
        tp = ((preds == 1) & (y_val_f == 1)).sum().item()
        fp = ((preds == 1) & (y_val_f == 0)).sum().item()
        fn = ((preds == 0) & (y_val_f == 1)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
    return f1

def permutation_null_f1(X_train, y_train, X_val, y_val, device,
                        n_perm: int = N_PERMUTATIONS_PROBE,
                        epochs: int = 200, lr: float = 0.05,
                        seed: int = 0) -> List[float]:
    null_f1s = []
    y_train_cpu = y_train.cpu()
    for i in range(n_perm):
        g = torch.Generator().manual_seed(seed * 1000 + i)
        perm = torch.randperm(y_train_cpu.numel(), generator=g)
        y_shuffled = y_train_cpu[perm].to(y_train.device)
        f1 = train_linear_probe(X_train, y_shuffled, X_val, y_val, device,
                                epochs=epochs, lr=lr)
        null_f1s.append(f1)
    return null_f1s

@torch.no_grad()
def _collect_l4(snap_path, dataset, loader, device):
    ckpt = torch.load(str(snap_path), map_location='cpu', weights_only=False)
    model = build_resnet18(dataset, num_classes=_num_classes_from_ckpt(ckpt))
    model.load_state_dict(ckpt['state_dict'])
    model.eval().to(device)
    bufs, labs = [], []
    for imgs, labels in loader:
        model(imgs.to(device))
        bufs.append(model.hooks[TARGET_LAYER].cpu())
        labs.append(labels)
    model.to('cpu'); del model
    return torch.cat(bufs), torch.cat(labs)

def _load_sae(bank, task_id, layer, device):
    state = bank[(task_id, layer)]['sae_state']
    d_sae = state['encoder.weight'].shape[0]
    d_in = state['encoder.weight'].shape[1]
    k = state['k'].item() if hasattr(state.get('k'), 'item') else max(8, d_in // 16)
    sae = AutoEncoderTopK(d_in, d_sae, k=k)
    sae.load_state_dict(state)
    return sae.eval().to(device)

@torch.no_grad()
def concept_activity_targets(features, sae, concept_code):
    codes = sae.encode(features)

    active_latents = concept_code > 0

    if active_latents.sum() == 0:
        return torch.zeros(
            features.shape[0],
            dtype=torch.long,
            device=features.device
        )

    concept_score = codes[:, active_latents].sum(dim=1)

    return (concept_score > 0).long()

def run_recoverability(results_dir, dataset, seed, data_root, method,
                       num_tasks=10, device='cuda',
                       fit_frac=FIT_FRAC, n_perm=N_PERMUTATIONS_PROBE,
                       max_samples_per_task=None, out_name='recoverability_v3.json'):
    from data.cifar_cl import ClassIncrementalScenario

    total_classes = 100 if dataset == 'cifar100' else 200
    classes_per_task = total_classes // num_tasks

    bank_path = os.path.join(results_dir, 'analysis', f'{dataset}_t{num_tasks}_s{seed}',
                             method, 'posthoc_sae', 'sae_bank.pt')
    bank = torch.load(bank_path, map_location='cpu', weights_only=False)

    fate_path = os.path.join(results_dir, 'analysis', f'{dataset}_t{num_tasks}_s{seed}',
                             method, 'feature_fate.json')
    with open(fate_path) as f:
        fate_data = json.load(f)

    snap_dir = os.path.join(results_dir, 'snapshots', f'{method}_{dataset}_t{num_tasks}_s{seed}')
    final_task = num_tasks - 1

    scenario = ClassIncrementalScenario(dataset=dataset, num_tasks=num_tasks,
                                        data_root=data_root, seed=42)

    results = {}
    for task_id in range(num_tasks - 1):
        s = final_task - task_id
        if s == 0:
            continue
        layer_fate = fate_data.get(str(task_id), {}).get(TARGET_LAYER)
        if layer_fate is None:
            continue

        fit_loader, eval_loader = scenario.get_split_activation_loaders(
            task_id, fit_frac=fit_frac, max_samples=max_samples_per_task,
            split_seed=seed)

        snap_t = Path(snap_dir) / f'task{task_id}.pt'
        snap_ts = Path(snap_dir) / f'task{final_task}.pt'

        feats_t_fit, labels_fit = _collect_l4(
            snap_t, dataset, fit_loader, device
        )

        feats_ts_fit, _ = _collect_l4(
            snap_ts, dataset, fit_loader, device
        )

        feats_t_eval, labels_eval = _collect_l4(
            snap_t, dataset, eval_loader, device
        )

        feats_ts_eval, _ = _collect_l4(
            snap_ts, dataset, eval_loader, device
        )

        feats_t_fit = feats_t_fit.to(device)
        feats_ts_fit = feats_ts_fit.to(device)
        feats_t_eval = feats_t_eval.to(device)
        feats_ts_eval = feats_ts_eval.to(device)

        W, b = fit_linear_translation(feats_ts_fit, feats_t_fit)
        translated_eval = apply_translation(feats_ts_eval, W, b)

        sae = _load_sae(bank, task_id, TARGET_LAYER, device)
        info = bank[(task_id, TARGET_LAYER)]

        task_start = task_id * classes_per_task
        per_class = {}
        for c_key, c_info in layer_fate['per_class'].items():
            c_global = int(c_key)
            if c_info['fate'] == 'survived':
                continue

            c_local_idx = c_global - task_start
            stored_code = torch.tensor(info['sparse_codes'][c_local_idx]).to(device)
            mask_class_fit  = (labels_fit.to(device) == c_global)
            mask_class_eval = (labels_eval.to(device) == c_global)
            if mask_class_fit.sum() < 2 or mask_class_eval.sum() < 2:
                continue

            translated_mean = translated_eval[mask_class_eval].mean(0, keepdim=True)
            translated_code = sae.encode(translated_mean).squeeze(0)
            cos_recovered = F.cosine_similarity(
                translated_code.unsqueeze(0), stored_code.unsqueeze(0)).item()

            if cos_recovered >= HI_COS:
                per_class[c_global] = {'outcome': 'recovered',
                                       'cos_after_translation': round(cos_recovered, 4)}
                continue

            concept_active_fit = concept_activity_targets(
                feats_t_fit,
                sae,
                stored_code
            )

            concept_active_eval = concept_activity_targets(
                feats_t_eval,
                sae,
                stored_code
            )

            if (
                concept_active_fit.sum() == 0
                or concept_active_fit.sum() == concept_active_fit.numel()
                or concept_active_eval.sum() == 0
                or concept_active_eval.sum() == concept_active_eval.numel()
            ):

                continue

            f1_true = train_linear_probe(
                feats_ts_fit,
                concept_active_fit,
                feats_ts_eval,
                concept_active_eval,
                device
            )

            null_f1s = permutation_null_f1(
                feats_ts_fit,
                concept_active_fit,
                feats_ts_eval,
                concept_active_eval,
                device,
                n_perm=n_perm,
                seed=task_id * 100_000 + c_global
            )

            null_mean = float(np.mean(null_f1s)) if null_f1s else 0.0
            null_p95 = float(np.percentile(null_f1s, 95)) if null_f1s else 0.0
            threshold = max(DECODABLE_F1_MIN, null_p95)

            outcome = 'decodable' if f1_true > threshold else 'lost'

            per_class[c_global] = {
                'outcome': outcome,
                'probe_f1': round(f1_true, 4),
                'probe_null_mean': round(null_mean, 4),
                'probe_null_p95': round(null_p95, 4),
                'probe_delta': round(f1_true - null_mean, 4),
                'probe_threshold_used': round(threshold, 4),
                'cos_after_translation': round(cos_recovered, 4),
            }

        sae.to('cpu')
        results[task_id] = {'s': s, 'per_class': per_class}
        n_checked = len(per_class)
        n_recovered = sum(1 for v in per_class.values() if v['outcome'] == 'recovered')
        n_decodable = sum(1 for v in per_class.values() if v['outcome'] == 'decodable')
        n_lost = sum(1 for v in per_class.values() if v['outcome'] == 'lost')
        print(f"  task {task_id} (s={s}): {n_checked} checked | "
              f"recovered={n_recovered} decodable={n_decodable} lost={n_lost}")

    results['_meta'] = {
        'fit_frac': fit_frac,
        'n_permutations_probe': n_perm,
        'decodable_f1_min': DECODABLE_F1_MIN,
        'hi_cos': HI_COS,
        'ridge_lambda': RIDGE_LAMBDA,
        'split_seed': seed,
        'max_samples_per_task': max_samples_per_task,

        'version': 'v3_held_out_split_concept_activity_probe',
        'probe_target': 'original_task_SAE_concept_activity',
    }

    out_dir = os.path.join(results_dir, 'analysis', f'{dataset}_t{num_tasks}_s{seed}', method)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved -> {out_path}")
    return results

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--dataset', default='cifar100')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data_root', default='/path/to/')
    p.add_argument('--num_tasks', type=int, default=10)
    p.add_argument('--device', default='cuda')
    p.add_argument('--method', default='sae_fsgp',
                   choices=['finetune', 'ewc', 'lwf', 'fecam', 'sae_fsgp',
                           'dense_fsgp', 'efc', 'dsal'])
    p.add_argument('--fit_frac', type=float, default=FIT_FRAC,
                   help='fraction of each task pool used to FIT the translation/probe')
    p.add_argument('--n_perm', type=int, default=N_PERMUTATIONS_PROBE,
                   help='number of label-shuffled null probes per concept')
    p.add_argument('--max_samples_per_task', type=int, default=None,
                   help='cap on pool size before the fit/eval split; default = use all '
                        'of that task\'s train images')
    p.add_argument('--out_name', default='recoverability.json',
                   help='output filename inside analysis/{tag}/{method}/ -- keep this '
                        'different from the old recoverability.json so old results '
                        'are preserved for comparison')
    args = p.parse_args()
    run_recoverability(args.results_dir, args.dataset, args.seed, args.data_root,
                       args.method, args.num_tasks, args.device,
                       args.fit_frac, args.n_perm, args.max_samples_per_task,
                       args.out_name)
