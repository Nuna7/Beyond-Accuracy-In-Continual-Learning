# """
# analysis/recoverability_null_control.py

# Permutation-null control for the recoverability check in recoverability.py.

# Question: does the ridge-regularized translation fit "recover" a concept
# because it captures a genuine temporal correspondence between checkpoint-t
# and checkpoint-(t+s) representations, or would ANY sufficiently flexible
# linear map inflate cosine similarity regardless of which target it's
# compared against?

# For every concept recoverability.py checks (dead/drifted under
# feature_fate.py), we fit the SAME translation (checkpoint-(t+s) ->
# checkpoint-t, on task t's train split) as the main script, but instead of
# comparing the translated representation only against ITS OWN class's
# stored SAE concept vector, we ALSO compare it against several other,
# MISMATCHED classes' stored concept vectors from the same task. If the real
# (correct-correspondence) comparison shows substantially higher cosine
# similarity than the mismatched ones, recovery reflects genuine
# correspondence, not an artifact of the translation's flexibility. If real
# and mismatched comparisons look similar, the recoverability result is
# suspect and should not be reported as-is.

# Reuses the exact same feature-extraction and translation-fitting logic as
# recoverability.py for an apples-to-apples comparison; only the final
# comparison step differs (one real + several permuted targets instead of
# just the real one).

# No retraining required. Same runtime profile as recoverability.py (the
# expensive part is loading snapshots and extracting features, which this
# script does once per task, same as the original).
# """
# import os
# import json
# import random
# import torch
# import torch.nn.functional as F
# from pathlib import Path

# import os, sys, json, argparse
# import numpy as np
# import torch
# import torch.nn.functional as F
# from pathlib import Path
# from typing import Dict

# sys.path.insert(0, str(Path(__file__).parent.parent))

# from models.resnet_hook import build_resnet18
# from sae.compat import AutoEncoderTopK

# TARGET_LAYER = 'layer4'
# RIDGE_LAMBDA = 1e-2
# N_PERMUTATIONS = 5   # average over several mismatched targets per concept,
#                       # not just one, to reduce noise from an unlucky draw


# # ── translation fit (identical to recoverability.py) ─────────────────────────

# def fit_linear_translation(X_source, X_target, ridge=RIDGE_LAMBDA):
#     N, d = X_source.shape
#     X_aug = torch.cat([X_source, torch.ones(N, 1, device=X_source.device)], dim=1)
#     A = X_aug.T @ X_aug + ridge * torch.eye(d + 1, device=X_source.device)
#     B = X_aug.T @ X_target
#     sol = torch.linalg.solve(A, B)
#     return sol[:-1, :], sol[-1, :]


# def apply_translation(X, W, b):
#     return X @ W + b


# # ── snapshot / feature helpers (identical to recoverability.py, including
# #    the fc-width fix from the DS-AL checkpoint issue) ───────────────────────

# @torch.no_grad()
# def _collect_l4(snap_path, dataset, loader, device):
#     ckpt = torch.load(str(snap_path), map_location='cpu', weights_only=False)
#     num_classes = ckpt['state_dict']['fc.weight'].shape[0]
#     model = build_resnet18(dataset, num_classes=num_classes)
#     model.load_state_dict(ckpt['state_dict'])
#     model.eval().to(device)
#     bufs, labs = [], []
#     for imgs, labels in loader:
#         model(imgs.to(device))
#         bufs.append(model.hooks[TARGET_LAYER].cpu())
#         labs.append(labels)
#     model.to('cpu'); del model
#     return torch.cat(bufs), torch.cat(labs)


# def _load_sae(bank, task_id, layer, device):
#     state = bank[(task_id, layer)]['sae_state']
#     d_sae = state['encoder.weight'].shape[0]
#     d_in = state['encoder.weight'].shape[1]
#     k = state['k'].item() if hasattr(state.get('k'), 'item') else max(8, d_in // 16)
#     sae = AutoEncoderTopK(d_in, d_sae, k=k)
#     sae.load_state_dict(state)
#     return sae.eval().to(device)


# # ── main entry ────────────────────────────────────────────────────────────────

# def run_null_control(results_dir, dataset, seed, data_root, method,
#                      num_tasks=10, device='cuda', seed_perm=0):
#     from data.cifar_cl import ClassIncrementalScenario

#     random.seed(seed_perm)
#     total_classes = 100 if dataset == 'cifar100' else 200
#     classes_per_task = total_classes // num_tasks

#     bank_path = os.path.join(results_dir, 'analysis', f'{dataset}_t{num_tasks}_s{seed}',
#                              method, 'posthoc_sae', 'sae_bank.pt')
#     bank = torch.load(bank_path, map_location='cpu', weights_only=False)

#     fate_path = os.path.join(results_dir, 'analysis', f'{dataset}_t{num_tasks}_s{seed}',
#                              method, 'feature_fate.json')
#     with open(fate_path) as f:
#         fate_data = json.load(f)

#     snap_dir = os.path.join(results_dir, 'snapshots', f'{method}_{dataset}_t{num_tasks}_s{seed}')
#     final_task = num_tasks - 1

#     scenario = ClassIncrementalScenario(dataset=dataset, num_tasks=num_tasks,
#                                         data_root=data_root, seed=42)

#     results = {}
#     for task_id in range(num_tasks - 1):
#         s = final_task - task_id
#         if s == 0:
#             continue
#         layer_fate = fate_data.get(str(task_id), {}).get(TARGET_LAYER)
#         if layer_fate is None:
#             continue

#         checked_classes = [int(c) for c, v in layer_fate['per_class'].items()
#                            if v['fate'] != 'survived']
#         if len(checked_classes) < 2:
#             continue   # need at least 2 non-survived classes in this task to permute against

#         train_loader = scenario.get_activation_loader(task_id, max_samples=2000)
#         eval_loader = scenario.get_clean_activation_loader(task_id, max_samples=2000)

#         snap_t = Path(snap_dir) / f'task{task_id}.pt'
#         snap_ts = Path(snap_dir) / f'task{final_task}.pt'

#         feats_t_train, labels_train = _collect_l4(snap_t, dataset, train_loader, device)
#         feats_ts_train, _ = _collect_l4(snap_ts, dataset, train_loader, device)
#         feats_t_eval, labels_eval = _collect_l4(snap_t, dataset, eval_loader, device)
#         feats_ts_eval, _ = _collect_l4(snap_ts, dataset, eval_loader, device)

#         feats_t_train, feats_ts_train = feats_t_train.to(device), feats_ts_train.to(device)
#         feats_t_eval, feats_ts_eval = feats_t_eval.to(device), feats_ts_eval.to(device)

#         W, b = fit_linear_translation(feats_ts_train, feats_t_train)
#         translated_eval = apply_translation(feats_ts_eval, W, b)

#         sae = _load_sae(bank, task_id, TARGET_LAYER, device)
#         info = bank[(task_id, TARGET_LAYER)]
#         task_start = task_id * classes_per_task

#         per_class = {}
#         for c_global in checked_classes:
#             c_local_idx = c_global - task_start
#             stored_code_real = torch.tensor(info['sparse_codes'][c_local_idx]).to(device)
#             mask_class_eval = (labels_eval.to(device) == c_global)
#             if mask_class_eval.sum() < 2:
#                 continue

#             translated_mean = translated_eval[mask_class_eval].mean(0, keepdim=True)
#             translated_code = sae.encode(translated_mean).squeeze(0)

#             cos_real = F.cosine_similarity(
#                 translated_code.unsqueeze(0), stored_code_real.unsqueeze(0)).item()

#             other_classes = [c for c in checked_classes if c != c_global]
#             n_perm = min(N_PERMUTATIONS, len(other_classes))
#             perm_targets = random.sample(other_classes, n_perm)
#             cos_null_vals = []
#             for c_mismatch in perm_targets:
#                 mismatch_idx = c_mismatch - task_start
#                 stored_code_mismatch = torch.tensor(
#                     info['sparse_codes'][mismatch_idx]).to(device)
#                 cos_null_vals.append(F.cosine_similarity(
#                     translated_code.unsqueeze(0), stored_code_mismatch.unsqueeze(0)).item())
#             cos_null_mean = sum(cos_null_vals) / len(cos_null_vals)

#             per_class[c_global] = {
#                 'cos_real': round(cos_real, 4),
#                 'cos_null_mean': round(cos_null_mean, 4),
#                 'cos_null_vals': [round(v, 4) for v in cos_null_vals],
#                 'delta': round(cos_real - cos_null_mean, 4),
#             }

#         sae.to('cpu')
#         results[task_id] = {'s': s, 'per_class': per_class}
#         n = len(per_class)
#         if n:
#             mean_real = sum(v['cos_real'] for v in per_class.values()) / n
#             mean_null = sum(v['cos_null_mean'] for v in per_class.values()) / n
#             print(f"  task {task_id} (s={s}): n={n} | "
#                   f"mean_real={mean_real:.3f} mean_null={mean_null:.3f} "
#                   f"delta={mean_real - mean_null:+.3f}")

#     out_dir = os.path.join(results_dir, 'analysis', f'{dataset}_t{num_tasks}_s{seed}', method)
#     os.makedirs(out_dir, exist_ok=True)
#     out_path = os.path.join(out_dir, 'recoverability_null_control.json')
#     with open(out_path, 'w') as f:
#         json.dump({str(k): v for k, v in results.items()}, f, indent=2)
#     print(f"  Saved -> {out_path}")
#     return results


# if __name__ == '__main__':
#     import argparse
#     p = argparse.ArgumentParser()
#     p.add_argument('--results_dir', default='./results_sweep')
#     p.add_argument('--dataset', default='cifar100')
#     p.add_argument('--seed', type=int, default=0)
#     p.add_argument('--data_root', default='/media/20TB/vision/CL/data_cache')
#     p.add_argument('--num_tasks', type=int, default=10)
#     p.add_argument('--device', default='cuda')
#     p.add_argument('--method', default='sae_fsgp',
#                    choices=['finetune', 'ewc', 'lwf', 'fecam', 'sae_fsgp',
#                            'dense_fsgp', 'efc', 'dsal'])
#     p.add_argument('--seed_perm', type=int, default=0,
#                    help='seed for random selection of mismatched targets')
#     args = p.parse_args()
#     run_null_control(args.results_dir, args.dataset, args.seed, args.data_root,
#                      args.method, args.num_tasks, args.device, args.seed_perm)

"""
analysis/recoverability_null_control.py  (v2 — held-out split)

Permutation-null control for the TRANSLATION step in recoverability.py:
does the ridge-regularized translation fit "recover" a concept because it
captures a genuine temporal correspondence between checkpoint-t and
checkpoint-T representations, or would ANY sufficiently flexible linear
map inflate cosine similarity regardless of which target it's compared
against?

For every concept recoverability.py checks (dead/drifted under
feature_fate.py), we fit the SAME translation (final checkpoint -> task-t
checkpoint, on task t's FIT split) as the main script, but instead of
comparing the translated representation only against ITS OWN class's
stored SAE concept vector, we ALSO compare it against several other,
MISMATCHED classes' stored concept vectors from the same task. If the
real (correct-correspondence) comparison shows substantially higher
cosine similarity than the mismatched ones, recovery reflects genuine
correspondence, not an artifact of the translation's flexibility. If real
and mismatched comparisons look similar, the recoverability result is
suspect and should not be reported as-is.

────────────────────────────────────────────────────────────────────────
CHANGE vs. the original recoverability_null_control.py
────────────────────────────────────────────────────────────────────────
Same underlying bug as recoverability.py v1: `get_activation_loader` and
`get_clean_activation_loader` return the SAME image indices (both seeded
by `np.random.default_rng(task_id)`), so "train" and "eval" for the
translation fit were the same images, just with/without augmentation.
Fixed by switching to `scenario.get_split_activation_loaders(task_id,
fit_frac=..., split_seed=seed)`, which returns two disjoint index sets.
Uses the same fit_frac/split_seed convention as recoverability.py v2 so a
given (dataset, seed) run's null control is checking the exact same
fit/held-out partition as its recoverability run.

No retraining required. Same runtime profile as recoverability.py v2 (the
expensive part is loading snapshots and extracting features, which this
script does once per task, same as before).
"""
import os
import json
import random
import torch
import torch.nn.functional as F
from pathlib import Path

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.resnet_hook import build_resnet18
from sae.compat import AutoEncoderTopK

TARGET_LAYER = 'layer4'
RIDGE_LAMBDA = 1e-2
FIT_FRAC = 0.8
N_PERMUTATIONS = 5   # average over several mismatched targets per concept,
                      # not just one, to reduce noise from an unlucky draw


# ── translation fit (identical to recoverability.py) ─────────────────────────

def fit_linear_translation(X_source, X_target, ridge=RIDGE_LAMBDA):
    N, d = X_source.shape
    X_aug = torch.cat([X_source, torch.ones(N, 1, device=X_source.device)], dim=1)
    A = X_aug.T @ X_aug + ridge * torch.eye(d + 1, device=X_source.device)
    B = X_aug.T @ X_target
    sol = torch.linalg.solve(A, B)
    return sol[:-1, :], sol[-1, :]


def apply_translation(X, W, b):
    return X @ W + b


# ── snapshot / feature helpers (identical to recoverability.py, including
#    the fc-width fix from the DS-AL checkpoint issue) ───────────────────────

@torch.no_grad()
def _collect_l4(snap_path, dataset, loader, device):
    ckpt = torch.load(str(snap_path), map_location='cpu', weights_only=False)
    num_classes = ckpt['state_dict']['fc.weight'].shape[0]
    model = build_resnet18(dataset, num_classes=num_classes)
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


# ── main entry ────────────────────────────────────────────────────────────────

def run_null_control(results_dir, dataset, seed, data_root, method,
                     num_tasks=10, device='cuda', seed_perm=0,
                     fit_frac=FIT_FRAC, max_samples_per_task=None,
                     out_name='recoverability_null_control_v2.json'):
    from data.cifar_cl import ClassIncrementalScenario

    random.seed(seed_perm)
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

        checked_classes = [int(c) for c, v in layer_fate['per_class'].items()
                           if v['fate'] != 'survived']
        if len(checked_classes) < 2:
            continue   # need at least 2 non-survived classes in this task to permute against

        # Same split convention as recoverability.py v2 -- uses `seed`
        # (the experiment seed), not `seed_perm` (which only controls the
        # mismatched-target draw below), so a given run's null control
        # checks the same fit/held-out partition as its recoverability run.
        fit_loader, eval_loader = scenario.get_split_activation_loaders(
            task_id, fit_frac=fit_frac, max_samples=max_samples_per_task,
            split_seed=seed)

        snap_t = Path(snap_dir) / f'task{task_id}.pt'
        snap_ts = Path(snap_dir) / f'task{final_task}.pt'

        feats_t_fit, labels_fit   = _collect_l4(snap_t, dataset, fit_loader, device)
        feats_ts_fit, _           = _collect_l4(snap_ts, dataset, fit_loader, device)
        feats_ts_eval, labels_eval = _collect_l4(snap_ts, dataset, eval_loader, device)

        feats_t_fit, feats_ts_fit = feats_t_fit.to(device), feats_ts_fit.to(device)
        feats_ts_eval = feats_ts_eval.to(device)

        W, b = fit_linear_translation(feats_ts_fit, feats_t_fit)
        translated_eval = apply_translation(feats_ts_eval, W, b)

        sae = _load_sae(bank, task_id, TARGET_LAYER, device)
        info = bank[(task_id, TARGET_LAYER)]
        task_start = task_id * classes_per_task

        per_class = {}
        for c_global in checked_classes:
            c_local_idx = c_global - task_start
            stored_code_real = torch.tensor(info['sparse_codes'][c_local_idx]).to(device)
            mask_class_eval = (labels_eval.to(device) == c_global)
            if mask_class_eval.sum() < 2:
                continue

            translated_mean = translated_eval[mask_class_eval].mean(0, keepdim=True)
            translated_code = sae.encode(translated_mean).squeeze(0)

            cos_real = F.cosine_similarity(
                translated_code.unsqueeze(0), stored_code_real.unsqueeze(0)).item()

            other_classes = [c for c in checked_classes if c != c_global]
            n_perm = min(N_PERMUTATIONS, len(other_classes))
            perm_targets = random.sample(other_classes, n_perm)
            cos_null_vals = []
            for c_mismatch in perm_targets:
                mismatch_idx = c_mismatch - task_start
                stored_code_mismatch = torch.tensor(
                    info['sparse_codes'][mismatch_idx]).to(device)
                cos_null_vals.append(F.cosine_similarity(
                    translated_code.unsqueeze(0), stored_code_mismatch.unsqueeze(0)).item())
            cos_null_mean = sum(cos_null_vals) / len(cos_null_vals)

            per_class[c_global] = {
                'cos_real': round(cos_real, 4),
                'cos_null_mean': round(cos_null_mean, 4),
                'cos_null_vals': [round(v, 4) for v in cos_null_vals],
                'delta': round(cos_real - cos_null_mean, 4),
            }

        sae.to('cpu')
        results[task_id] = {'s': s, 'per_class': per_class}
        n = len(per_class)
        if n:
            mean_real = sum(v['cos_real'] for v in per_class.values()) / n
            mean_null = sum(v['cos_null_mean'] for v in per_class.values()) / n
            print(f"  task {task_id} (s={s}): n={n} | "
                  f"mean_real={mean_real:.3f} mean_null={mean_null:.3f} "
                  f"delta={mean_real - mean_null:+.3f}")

    results['_meta'] = {
        'fit_frac': fit_frac, 'split_seed': seed, 'seed_perm': seed_perm,
        'n_mismatch_permutations': N_PERMUTATIONS,
        'max_samples_per_task': max_samples_per_task,
        'version': 'v2_held_out_split',
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
    p.add_argument('--results_dir', default='./results_sweep')
    p.add_argument('--dataset', default='cifar100')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data_root', default='/media/20TB/vision/CL/data_cache')
    p.add_argument('--num_tasks', type=int, default=10)
    p.add_argument('--device', default='cuda')
    p.add_argument('--method', default='sae_fsgp',
                   choices=['finetune', 'ewc', 'lwf', 'fecam', 'sae_fsgp',
                           'dense_fsgp', 'efc', 'dsal', 'ada_gauss'])
    p.add_argument('--seed_perm', type=int, default=0,
                   help='seed for random selection of mismatched targets')
    p.add_argument('--fit_frac', type=float, default=FIT_FRAC)
    p.add_argument('--max_samples_per_task', type=int, default=None)
    p.add_argument('--out_name', default='recoverability_null_control_v2.json')
    args = p.parse_args()
    run_null_control(args.results_dir, args.dataset, args.seed, args.data_root,
                     args.method, args.num_tasks, args.device, args.seed_perm,
                     args.fit_frac, args.max_samples_per_task, args.out_name)