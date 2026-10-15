import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Optional

@torch.no_grad()
def accuracy(model, loader: DataLoader, device: str = "cuda") -> float:
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / max(total, 1)

@torch.no_grad()
def accuracy_topk(model, loader: DataLoader,
                  k: int = 5, device: str = "cuda") -> float:
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        _, topk = model(imgs).topk(k, dim=1)
        correct += topk.eq(labels.view(-1, 1).expand_as(topk)).any(1).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / max(total, 1)

def build_accuracy_matrix(
    trainer,
    scenario,
    eval_after_each_task: bool = True,
) -> np.ndarray:
    T = scenario.num_tasks
    R = np.full((T, T), np.nan)

    for res in trainer._results:
        t = res['task_id']
        R[t, t] = res.get('test_acc', np.nan)
    return R

def compute_metrics(R: np.ndarray) -> Dict[str, float]:
    T = R.shape[0]
    last_row = R[T - 1, :]

    avg_acc = float(np.nanmean(last_row))

    if T <= 1:
        return {'avg_acc': avg_acc, 'forgetting': 0.0, 'bwt': 0.0}

    forgetting_vals, bwt_vals = [], []
    for j in range(T - 1):
        col = R[:, j]

        max_acc = float(np.nanmax(col[: T - 1]))
        final   = float(R[T - 1, j]) if not np.isnan(R[T - 1, j]) else max_acc
        diag    = float(R[j, j])     if not np.isnan(R[j, j])     else 0.0
        forgetting_vals.append(max_acc - final)
        bwt_vals.append(final - diag)

    return {
        'avg_acc':    avg_acc,
        'forgetting': float(np.mean(forgetting_vals)),
        'bwt':        float(np.mean(bwt_vals)),
    }

def count_stored_params(trainer) -> Dict[str, int]:
    totals = {}

    method = type(trainer).__name__

    if method == 'EWCTrainer':
        fisher_params = sum(v.numel() for v in trainer._fisher.values())
        optima_params = sum(v.numel() for v in trainer._optima.values())
        totals['fisher_params']    = fisher_params
        totals['optima_params']    = optima_params
        totals['total_extra']      = fisher_params + optima_params

    elif method == 'LwFTrainer':
        if trainer._old_model is not None:
            model_params = sum(p.numel() for p in trainer._old_model.parameters())
        else:
            model_params = 0
        totals['old_model_params'] = model_params
        totals['total_extra']      = model_params

    elif method == 'FeCAMTrainer':
        mean_params = sum(v.numel() for v in trainer._means.values())
        cov_params  = sum(v.numel() for v in trainer._covs.values())
        totals['mean_params']   = mean_params
        totals['cov_params']    = cov_params
        totals['total_extra']   = mean_params + cov_params

    elif method in ('SAEFSGPTrainer',):
        sae_params = sum(
            p.numel()
            for sae in trainer._sae_bank._saes.values()
            for p in sae.parameters()
        )
        proto_dense  = sum(
            v['dense'].numel()
            for v in trainer._sae_bank._prototypes.values()
        )
        proto_sparse = sum(
            v['sparse'].numel() + v['mask'].numel()
            for v in trainer._sae_bank._prototypes.values()
        )
        old_model_params = sum(
            p.numel()
            for m in trainer._old_models
            for p in m.parameters()
        )
        totals['sae_params']        = sae_params
        totals['proto_dense']       = proto_dense
        totals['proto_sparse']      = proto_sparse
        totals['old_model_params']  = old_model_params
        totals['total_extra']       = sae_params + proto_dense + proto_sparse + old_model_params

    else:
        totals['total_extra'] = 0

    return totals

def print_summary(method_name: str, per_task_accs: List[float],
                  metrics: Dict[str, float], storage: Dict[str, int]):
    print(f"\n{'─'*60}")
    print(f"  Method : {method_name}")
    print(f"  Per-task acc : {[f'{a:.1f}' for a in per_task_accs]}")
    print(f"  Avg Acc      : {metrics['avg_acc']:.2f}%")
    print(f"  Forgetting   : {metrics['forgetting']:.2f}%")
    print(f"  BWT          : {metrics['bwt']:.2f}%")
    print(f"  Extra params : {storage.get('total_extra', 0):,}")
    print(f"{'─'*60}")
