# """
# analysis/recoverability.py

# Extends feature_fate.py's survived/drifted/dead taxonomy with a
# recoverability/decodability layer, adapted from Filus et al. 2026
# ("Lost or Hidden?") Eq. 3 + their concept taxonomy, applied to this
# codebase's class-incremental cold-start setting (which their paper does
# not study -- they are task-incremental with a separate head per task).

# For each concept (a class-level SAE latent from an earlier task) that
# feature_fate.py already flagged as 'dead' or 'drifted' at some later
# checkpoint t+s:

#   1. Fit a LINEAR TRANSLATION T: h_{t+s}(x) -> h_t(x) via ridge-regularised
#      least squares on task-t's TRAIN split (Filus Eq. 3). This uses the
#      current (t+s) and original (t) backbone snapshots on the SAME images.
#   2. Apply T to the TEST split, re-encode through task-t's own SAE, and
#      recheck the concept against feature_fate's existing thresholds.
#      If it now clears them -> 'recovered'.
#   3. If not recovered, train a small linear probe on RAW (untranslated)
#      t+s features to predict the concept's binarised activity pattern
#      (from t). Evaluate F1 on held-out data.
#        F1 > DECODABLE_F1_MIN -> 'decodable'
#        else                  -> 'lost'

# This directly tests whether Filus et al.'s central finding (most apparent
# forgetting is recoverable misalignment, not erasure) transfers from their
# task-incremental, 4-classic-method, 2-10-task setting to this codebase's
# class-incremental cold-start, 9-method (including 2024 EFCIL SOTA), up-to-
# 20-task setting -- which neither Filus et al. nor Masip et al. test.

# Whether it transfers cleanly, partially, or not at all is an empirical
# question this script answers -- not something assumed going in.
# """
# import os, sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent))
# import json
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from pathlib import Path
# from typing import Dict, Tuple

# from models.resnet_hook import build_resnet18
# from sae.compat import AutoEncoderTopK

# TARGET_LAYER = 'layer4'
# HI_COS = 0.85          # matches feature_fate.py's 'survived' threshold
# LO_COS = 0.50           # matches feature_fate.py's 'drifted' threshold
# ACT_FRAC = 0.30
# DECODABLE_F1_MIN = 0.05  # Filus et al. use strict F1>0; we use a small
#                          # floor to absorb estimation noise from a single
#                          # linear probe fit rather than their more heavily
#                          # cross-validated setup -- reported as a knob,
#                          # not hidden.
# RIDGE_LAMBDA = 1e-2      # regularisation for the translation fit


# def _num_classes_from_ckpt(ckpt):
#     """
#     Build using the fc width actually saved in the checkpoint, not
#     ckpt['seen_classes'] -- they usually agree, but DS-AL's incremental
#     steps never call expand_classifier() (it doesn't use self.model.fc at
#     all after task 0), so its snapshots after task 0 have a STALE fc sized
#     to task 0's class count while seen_classes correctly reports the true
#     cumulative count. Since layer4 hooks are populated BEFORE fc runs in
#     forward(), fc's size never affects anything either script reads -- so
#     this sidesteps the mismatch entirely, no retraining required.
#     """
#     return ckpt['state_dict']['fc.weight'].shape[0]

# # ── linear translation (Filus Eq. 3) ─────────────────────────────────────────

# def fit_linear_translation(X_source: torch.Tensor, X_target: torch.Tensor,
#                            ridge: float = RIDGE_LAMBDA) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Closed-form ridge least squares: min_W,b E||W x_source + b - x_target||^2.
#     Returns (W, b). X_source, X_target: (N, d) on the same device.
#     VERIFIED against a known linear relationship (max param error 1e-4).
#     """
#     N, d = X_source.shape
#     X_aug = torch.cat([X_source, torch.ones(N, 1, device=X_source.device)], dim=1)  # (N, d+1)
#     A = X_aug.T @ X_aug + ridge * torch.eye(d + 1, device=X_source.device)          # (d+1, d+1)
#     B = X_aug.T @ X_target                                                          # (d+1, d)
#     sol = torch.linalg.solve(A, B)                                                  # (d+1, d)
#     W, b = sol[:-1, :], sol[-1, :]
#     return W, b


# def apply_translation(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     return X @ W + b


# # ── linear probe for decodability ────────────────────────────────────────────

# def train_linear_probe(X_train, y_train, X_val, y_val, device, epochs=200, lr=0.05):
#     """
#     Single-layer logistic regression, predicting binarised concept activity
#     from raw (untranslated) features. Returns F1 on the held-out split.
#     Kept as a plain torch module (no sklearn dependency assumed).
#     """
#     if y_train.sum() == 0 or y_train.sum() == y_train.numel():
#         # Degenerate: concept never/always active in the training split --
#         # a probe can't learn anything meaningful here either way.
#         return 0.0

#     d = X_train.shape[1]
#     probe = nn.Linear(d, 1).to(device)
#     opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
#     y_train_f = y_train.float().to(device)

#     for _ in range(epochs):
#         opt.zero_grad()
#         logits = probe(X_train).squeeze(-1)
#         loss = F.binary_cross_entropy_with_logits(logits, y_train_f)
#         loss.backward()
#         opt.step()

#     with torch.no_grad():
#         preds = (torch.sigmoid(probe(X_val).squeeze(-1)) > 0.5).float().cpu()
#         y_val_f = y_val.float().cpu()
#         tp = ((preds == 1) & (y_val_f == 1)).sum().item()
#         fp = ((preds == 1) & (y_val_f == 0)).sum().item()
#         fn = ((preds == 0) & (y_val_f == 1)).sum().item()
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#         f1 = (2 * precision * recall / (precision + recall)
#               if (precision + recall) > 0 else 0.0)
#     return f1


# # ── snapshot / feature helpers ───────────────────────────────────────────────

# @torch.no_grad()
# def _collect_l4(snap_path, dataset, loader, device):
#     ckpt = torch.load(str(snap_path), map_location='cpu', weights_only=False)
#     model = build_resnet18(dataset, num_classes=_num_classes_from_ckpt(ckpt))
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

# def run_recoverability(results_dir, dataset, seed, data_root, method,
#                        num_tasks=10, device='cuda'):
#     from data.cifar_cl import ClassIncrementalScenario

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
#     for task_id in range(num_tasks - 1):  # nothing to recover for the final task itself
#         s = final_task - task_id
#         if s == 0:
#             continue
#         layer_fate = fate_data.get(str(task_id), {}).get(TARGET_LAYER)
#         if layer_fate is None:
#             continue

#         PER_CLASS_TARGET = 200
#         classes_per_task = scenario.classes_per_task
#         max_samples = PER_CLASS_TARGET * classes_per_task
#         # Snapshots at t and t+s, evaluated on task-t's TRAIN (fit) and TEST (eval) splits
#         train_loader = scenario.get_activation_loader(task_id, max_samples=max_samples)
#         # get_clean_activation_loader gives a second, non-overlapping augmentation
#         # pass over the SAME indices -- used here purely as a held-out eval pass,
#         # not a strict disjoint test set (this codebase's activation loaders are
#         # capped/sampled from the train split; a fully disjoint held-out split
#         # would need a dedicated loader this codebase doesn't currently expose).
#         eval_loader = scenario.get_clean_activation_loader(task_id, max_samples=max_samples)

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
#         for c_local, c_info in layer_fate['per_class'].items():
#             c_global = int(c_local)
#             if c_info['fate'] == 'survived':
#                 continue  # already fine, nothing to check

#             c_local_idx = c_global - task_start
#             stored_code = torch.tensor(info['sparse_codes'][c_local_idx]).to(device)
#             mask_class_train = (labels_train.to(device) == c_global)
#             mask_class_eval = (labels_eval.to(device) == c_global)
#             if mask_class_train.sum() < 2 or mask_class_eval.sum() < 2:
#                 continue

#             # 1. Recovered? re-encode translated eval features, recheck cos_sim
#             translated_mean = translated_eval[mask_class_eval].mean(0, keepdim=True)
#             translated_code = sae.encode(translated_mean).squeeze(0)
#             cos_recovered = F.cosine_similarity(
#                 translated_code.unsqueeze(0), stored_code.unsqueeze(0)).item()

#             if cos_recovered >= HI_COS:
#                 per_class[c_global] = {'outcome': 'recovered', 'cos_after_translation': cos_recovered}
#                 continue

#             # 2. Decodable? linear probe on RAW (untranslated) t+s features
#             active_train = mask_class_train.long()
#             active_eval = mask_class_eval.long()
#             f1 = train_linear_probe(feats_ts_train, active_train,
#                                     feats_ts_eval, active_eval, device)

#             outcome = 'decodable' if f1 > DECODABLE_F1_MIN else 'lost'
#             per_class[c_global] = {'outcome': outcome, 'probe_f1': round(f1, 4),
#                                    'cos_after_translation': round(cos_recovered, 4)}

#         sae.to('cpu')
#         results[task_id] = {'s': s, 'per_class': per_class}
#         n_checked = len(per_class)
#         n_recovered = sum(1 for v in per_class.values() if v['outcome'] == 'recovered')
#         n_decodable = sum(1 for v in per_class.values() if v['outcome'] == 'decodable')
#         n_lost = sum(1 for v in per_class.values() if v['outcome'] == 'lost')
#         print(f"  task {task_id} (s={s}): {n_checked} checked | "
#               f"recovered={n_recovered} decodable={n_decodable} lost={n_lost}")

#     out_dir = os.path.join(results_dir, 'analysis', f'{dataset}_t{num_tasks}_s{seed}', method)
#     os.makedirs(out_dir, exist_ok=True)
#     out_path = os.path.join(out_dir, 'recoverability.json')
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
#                            'dense_fsgp', 'efc', 'dsal', 'ada_gauss'])
#     args = p.parse_args()
#     run_recoverability(args.results_dir, args.dataset, args.seed, args.data_root,
#                        args.method, args.num_tasks, args.device)

"""
analysis/recoverability.py  (v2 — held-out split + permutation-null decodability)

Extends feature_fate.py's survived/drifted/dead taxonomy with a
recoverability/decodability layer, adapted from Filus et al. 2026
("Lost or Hidden?") Eq. 3 + their concept taxonomy, applied to this
codebase's class-incremental cold-start setting (which their paper does
not study -- they are task-incremental with a separate head per task).

For each concept (a class-level SAE latent from an earlier task) that
feature_fate.py already flagged as 'dead' or 'drifted' at the final
checkpoint T:

  1. Fit a LINEAR TRANSLATION T: h_T(x) -> h_t(x) via ridge-regularised
     least squares (Filus Eq. 3), using ONLY the FIT split of task t's
     images (see CHANGE 1 below).
  2. Apply T to the HELD-OUT split, re-encode through task-t's own SAE,
     and recheck the concept against feature_fate's existing thresholds.
     If it now clears them -> 'recovered'.
  3. If not recovered, train a small linear probe on RAW (untranslated)
     T-checkpoint features (FIT split) to predict the concept's binarised
     activity pattern (from t), and evaluate F1 on the HELD-OUT split.
       F1 > decodability threshold -> 'decodable'
       else                        -> 'lost'
     The threshold is now a PERMUTATION NULL, not a bare constant (see
     CHANGE 2 below).

This directly tests whether Filus et al.'s central finding (most apparent
forgetting is recoverable misalignment, not erasure) transfers from their
task-incremental, 4-classic-method, 2-10-task setting to this codebase's
class-incremental cold-start, 8-method (including 2024 EFCIL SOTA), setting
-- which neither Filus et al. nor Masip et al. test.

SCOPE NOTE (previously a bug, now just documented): this script checks
each task-t concept against the FINAL checkpoint T only, not every
intermediate checkpoint t+1, t+2, .... That's a deliberate scope decision
to keep the analysis and its aggregation tractable, not a limitation of
the method -- report it as "recoverability/decodability at the final
checkpoint", not "at t+s for general s".

────────────────────────────────────────────────────────────────────────
CHANGES vs. the original recoverability.py
────────────────────────────────────────────────────────────────────────
CHANGE 1 -- disjoint fit/held-out split (was a real leakage bug):
    The old code fit the translation AND the probe on
    `get_activation_loader(task_id)` and evaluated both on
    `get_clean_activation_loader(task_id)`. Both call
    `np.random.default_rng(task_id)` internally with an identical
    max_samples, so they return the SAME underlying image indices --
    "eval" was just the same images minus augmentation, not a held-out
    set. That inflates both the translation's cosine similarity and the
    probe's F1: a sufficiently flexible fit can partially memorise the
    eval set precisely because it *is* the fit set.
    Fix: `scenario.get_split_activation_loaders(task_id, fit_frac=0.8, ...)`
    returns two DISJOINT index sets, both using the clean (non-augmented)
    transform, deterministic in (task_id, split_seed) so features from
    different checkpoints still line up on the same images.

CHANGE 2 -- permutation-null decodability criterion (was an arbitrary
    threshold): `F1 > 0.05` is not a principled bar -- with d=512 features
    and a few hundred held-out samples, plain sampling noise can clear it.
    Fix: for every concept that reaches the probe step, additionally fit
    N_PERMUTATIONS_PROBE probes with the TRAINING labels shuffled (held-out
    labels untouched), producing a null F1 distribution. A concept is
    'decodable' only if its true F1 exceeds max(DECODABLE_F1_MIN, the
    null's 95th percentile). DECODABLE_F1_MIN is kept as a sanity floor.
    Per-concept output now also stores probe_null_mean, probe_null_p95,
    and probe_delta = probe_f1 - probe_null_mean.

Everything else (translation math, SAE loading, the 'recovered' check
itself, snapshot loading incl. the DS-AL fc-width fix) is unchanged from
the original script.
"""
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
HI_COS = 0.85            # matches feature_fate.py's 'survived' threshold
LO_COS = 0.50             # matches feature_fate.py's 'drifted' threshold
ACT_FRAC = 0.30
DECODABLE_F1_MIN = 0.05   # sanity floor only -- the real bar is the
                          # permutation null (see CHANGE 2 above)
RIDGE_LAMBDA = 1e-2       # regularisation for the translation fit
FIT_FRAC = 0.8            # fraction of each task's images used to FIT the
                          # translation/probe; the rest is held out for eval
N_PERMUTATIONS_PROBE = 10 # null-distribution probes per concept


def _num_classes_from_ckpt(ckpt):
    """
    Build using the fc width actually saved in the checkpoint, not
    ckpt['seen_classes'] -- they usually agree, but DS-AL's incremental
    steps never call expand_classifier() (it doesn't use self.model.fc at
    all after task 0), so its snapshots after task 0 have a STALE fc sized
    to task 0's class count while seen_classes correctly reports the true
    cumulative count. Since layer4 hooks are populated BEFORE fc runs in
    forward(), fc's size never affects anything either script reads -- so
    this sidesteps the mismatch entirely, no retraining required.
    """
    return ckpt['state_dict']['fc.weight'].shape[0]


# ── linear translation (Filus Eq. 3) ─────────────────────────────────────────

def fit_linear_translation(X_source: torch.Tensor, X_target: torch.Tensor,
                           ridge: float = RIDGE_LAMBDA) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form ridge least squares: min_W,b E||W x_source + b - x_target||^2.
    Returns (W, b). X_source, X_target: (N, d) on the same device, same
    (disjoint-from-eval) FIT split.
    """
    N, d = X_source.shape
    X_aug = torch.cat([X_source, torch.ones(N, 1, device=X_source.device)], dim=1)  # (N, d+1)
    A = X_aug.T @ X_aug + ridge * torch.eye(d + 1, device=X_source.device)          # (d+1, d+1)
    B = X_aug.T @ X_target                                                          # (d+1, d)
    sol = torch.linalg.solve(A, B)                                                  # (d+1, d)
    W, b = sol[:-1, :], sol[-1, :]
    return W, b


def apply_translation(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return X @ W + b


# ── linear probe for decodability ────────────────────────────────────────────

def train_linear_probe(X_train, y_train, X_val, y_val, device, epochs=200, lr=0.05):
    """
    Single-layer logistic regression, predicting binarised concept activity
    from raw (untranslated) features. Returns F1 on the (disjoint) held-out
    split. Kept as a plain torch module (no sklearn dependency assumed).
    """
    if y_train.sum() == 0 or y_train.sum() == y_train.numel():
        # Degenerate: concept never/always active in the fit split -- a
        # probe can't learn anything meaningful here either way.
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
    """
    Label-permutation null for the decodability probe (CHANGE 2).

    Shuffles the TRAINING labels only -- y_val (the held-out ground truth
    used to score every fit) is never touched -- and refits the SAME probe
    architecture, same epochs/lr (same capacity/budget, or the null would
    be unfairly weak) n_perm times. This gives a null distribution of F1
    scores a probe reaches with no real train/held-out correspondence,
    just from fitting noise at this feature dimensionality / sample size.
    A concept only counts as genuinely decodable if its true F1 clears
    this null, not an arbitrary constant.
    """
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


# ── snapshot / feature helpers ───────────────────────────────────────────────

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
    """
    Build a binary concept-activity target for each image.

    concept_code is the stored SAE encoding of the class mean activation
    from the original task.

    An image is positive when at least one latent belonging to the
    original concept's active support fires in that image.
    """
    codes = sae.encode(features)

    # Active latent support of the class-level concept.
    active_latents = concept_code > 0

    if active_latents.sum() == 0:
        return torch.zeros(
            features.shape[0],
            dtype=torch.long,
            device=features.device
        )

    concept_score = codes[:, active_latents].sum(dim=1)

    return (concept_score > 0).long()

# ── main entry ────────────────────────────────────────────────────────────────

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
    for task_id in range(num_tasks - 1):  # nothing to recover for the final task itself
        s = final_task - task_id
        if s == 0:
            continue
        layer_fate = fate_data.get(str(task_id), {}).get(TARGET_LAYER)
        if layer_fate is None:
            continue

        # CHANGE 1: disjoint fit/held-out split, deterministic per
        # (task_id, seed) so old- and final-checkpoint features line up.
        fit_loader, eval_loader = scenario.get_split_activation_loaders(
            task_id, fit_frac=fit_frac, max_samples=max_samples_per_task,
            split_seed=seed)

        snap_t = Path(snap_dir) / f'task{task_id}.pt'
        snap_ts = Path(snap_dir) / f'task{final_task}.pt'

        # FIT split: need both checkpoints (translation target + source).
        # feats_t_fit, labels_fit = _collect_l4(snap_t, dataset, fit_loader, device)
        # feats_ts_fit, _         = _collect_l4(snap_ts, dataset, fit_loader, device)
        # # EVAL split: only need the final checkpoint (source) + ground-truth
        # # labels -- the old checkpoint's held-out features were computed by
        # # the original script but never actually used past label extraction.
        # feats_ts_eval, labels_eval = _collect_l4(snap_ts, dataset, eval_loader, device)

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

        #feats_t_fit, feats_ts_fit = feats_t_fit.to(device), feats_ts_fit.to(device)
        #feats_ts_eval = feats_ts_eval.to(device)

        W, b = fit_linear_translation(feats_ts_fit, feats_t_fit)
        translated_eval = apply_translation(feats_ts_eval, W, b)

        sae = _load_sae(bank, task_id, TARGET_LAYER, device)
        info = bank[(task_id, TARGET_LAYER)]

        task_start = task_id * classes_per_task
        per_class = {}
        for c_key, c_info in layer_fate['per_class'].items():
            c_global = int(c_key)
            if c_info['fate'] == 'survived':
                continue  # already fine, nothing to check

            c_local_idx = c_global - task_start
            stored_code = torch.tensor(info['sparse_codes'][c_local_idx]).to(device)
            mask_class_fit  = (labels_fit.to(device) == c_global)
            mask_class_eval = (labels_eval.to(device) == c_global)
            if mask_class_fit.sum() < 2 or mask_class_eval.sum() < 2:
                continue

            # 1. Recovered? re-encode translated held-out features, recheck cos_sim
            translated_mean = translated_eval[mask_class_eval].mean(0, keepdim=True)
            translated_code = sae.encode(translated_mean).squeeze(0)
            cos_recovered = F.cosine_similarity(
                translated_code.unsqueeze(0), stored_code.unsqueeze(0)).item()

            if cos_recovered >= HI_COS:
                per_class[c_global] = {'outcome': 'recovered',
                                       'cos_after_translation': round(cos_recovered, 4)}
                continue

            # 2. Decodable? linear probe on RAW (untranslated) final-checkpoint
            #    features, fit split -> held-out split, against a
            #    permutation null (CHANGE 2).
            # active_fit  = mask_class_fit.long()
            # active_eval = mask_class_eval.long()
            # f1_true = train_linear_probe(feats_ts_fit, active_fit,
            #                              feats_ts_eval, active_eval, device)
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

            # Degenerate concept activity cannot support a meaningful probe.

            if (
                concept_active_fit.sum() == 0
                or concept_active_fit.sum() == concept_active_fit.numel()
                or concept_active_eval.sum() == 0
                or concept_active_eval.sum() == concept_active_eval.numel()
            ):
                
                # per_class[c_global] = {
                #     'outcome': 'lost',
                #     'probe_f1': 0.0,
                #     'probe_null_mean': 0.0,
                #     'probe_null_p95': 0.0,
                #     'probe_delta': 0.0,
                #     'probe_threshold_used': DECODABLE_F1_MIN,
                #     'cos_after_translation': round(cos_recovered, 4),
                # }
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
            # null_f1s = permutation_null_f1(
            #     feats_ts_fit, active_fit, feats_ts_eval, active_eval, device,
            #     n_perm=n_perm, seed=task_id * 100_000 + c_global)
            # null_mean = float(np.mean(null_f1s)) if null_f1s else 0.0
            # null_p95 = float(np.percentile(null_f1s, 95)) if null_f1s else 0.0
            # threshold = max(DECODABLE_F1_MIN, null_p95)

            # outcome = 'decodable' if f1_true > threshold else 'lost'
            # per_class[c_global] = {
            #     'outcome': outcome,
            #     'probe_f1': round(f1_true, 4),
            #     'probe_null_mean': round(null_mean, 4),
            #     'probe_null_p95': round(null_p95, 4),
            #     'probe_delta': round(f1_true - null_mean, 4),
            #     'probe_threshold_used': round(threshold, 4),
            #     'cos_after_translation': round(cos_recovered, 4),
            # }

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
        #'version': 'v2_held_out_split_permutation_null',
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
    p.add_argument('--results_dir', default='./results_sweep')
    p.add_argument('--dataset', default='cifar100')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data_root', default='/media/20TB/vision/CL/data_cache')
    p.add_argument('--num_tasks', type=int, default=10)
    p.add_argument('--device', default='cuda')
    p.add_argument('--method', default='sae_fsgp',
                   choices=['finetune', 'ewc', 'lwf', 'fecam', 'sae_fsgp',
                           'dense_fsgp', 'efc', 'dsal', 'ada_gauss'])
    p.add_argument('--fit_frac', type=float, default=FIT_FRAC,
                   help='fraction of each task pool used to FIT the translation/probe')
    p.add_argument('--n_perm', type=int, default=N_PERMUTATIONS_PROBE,
                   help='number of label-shuffled null probes per concept')
    p.add_argument('--max_samples_per_task', type=int, default=None,
                   help='cap on pool size before the fit/eval split; default = use all '
                        'of that task\'s train images')
    p.add_argument('--out_name', default='recoverability_v3.json',
                   help='output filename inside analysis/{tag}/{method}/ -- keep this '
                        'different from the old recoverability.json so old results '
                        'are preserved for comparison')
    args = p.parse_args()
    run_recoverability(args.results_dir, args.dataset, args.seed, args.data_root,
                       args.method, args.num_tasks, args.device,
                       args.fit_frac, args.n_perm, args.max_samples_per_task,
                       args.out_name)