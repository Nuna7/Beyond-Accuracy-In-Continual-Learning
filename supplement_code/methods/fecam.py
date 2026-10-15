import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional

from .base_trainer import BaseTrainer

def ledoit_wolf_shrinkage(X: torch.Tensor, shrink: float = 1.0) -> torch.Tensor:
    n, d = X.shape
    S = (X.T @ X) / n
    mu = torch.trace(S) / d

    delta2  = ((X.T @ X / n).pow(2).sum() - torch.trace(S).pow(2) / d) / n
    gamma   = delta2 / ((S - mu * torch.eye(d, device=S.device)).pow(2).sum() + 1e-8)
    gamma   = gamma.clamp(0.0, 1.0) * shrink

    I = torch.eye(d, device=S.device, dtype=S.dtype)
    return (1 - gamma) * S + gamma * mu * I

class FeCAMTrainer(BaseTrainer):

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self._means:  Dict[int, torch.Tensor] = {}
        self._covs:   Dict[int, torch.Tensor] = {}
        self._task0_done = False

    def _tukey(self, x: torch.Tensor) -> torch.Tensor:
        lam = self.cfg.method.fecam_tukey_lambda
        return x.sign() * x.abs().pow(lam)

    @torch.no_grad()
    def _extract_features(self, loader: DataLoader) -> tuple:
        self.model.eval()
        all_feats, all_labels = [], []
        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            self.model(imgs)
            feats = self._tukey(self.model.hooks['layer4'].cpu())
            all_feats.append(feats)
            all_labels.append(labels)
        return torch.cat(all_feats), torch.cat(all_labels)

    def _update_stats(self, loader: DataLoader, task_id: int,
                      n_classes_this_task: int):
        feats, labels = self._extract_features(loader)
        start = task_id * n_classes_this_task

        for c_local in range(n_classes_this_task):
            c_global = start + c_local
            mask = labels == c_global
            if mask.sum() < 2:
                continue
            X = feats[mask]
            mu = X.mean(0)
            Z  = X - mu
            cov = ledoit_wolf_shrinkage(
                Z, self.cfg.method.fecam_cov_shrink)
            self._means[c_global] = mu
            self._covs[c_global]  = cov

        print(f"  [FeCAM] Stats updated for task {task_id} "
              f"({n_classes_this_task} new classes, "
              f"{len(self._means)} total)")

    def before_task(self, task_id: int, train_loader: DataLoader):
        if task_id >= self.cfg.method.fecam_freeze_after + 1:
            self.model.freeze_backbone()
            if task_id == 1:
                print("  [FeCAM] Backbone frozen after task 0")

    def after_task(self, task_id: int, train_loader: DataLoader):

        classes_per_task = self._last_classes_this_task
        self._update_stats(train_loader, task_id, classes_per_task)

    @torch.no_grad()
    def _eval_accuracy(self, loader: DataLoader) -> float:
        if not self._means:
            return super()._eval_accuracy(loader)

        self.model.eval()
        correct = total = 0
        classes = sorted(self._means.keys())

        device = self.device
        mu_stack  = torch.stack([self._means[c] for c in classes]).to(device)
        cov_inv   = []
        for c in classes:
            cov = self._covs[c].to(device).float()

            try:
                cov_inv.append(torch.linalg.inv(
                    cov + 1e-5 * torch.eye(cov.size(0), device=device)))
            except Exception:
                cov_inv.append(torch.eye(cov.size(0), device=device))

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            self.model(imgs)
            feats = self._tukey(self.model.hooks['layer4'])

            dists = []
            for i, c in enumerate(classes):
                diff = feats - mu_stack[i]
                Σinv = cov_inv[i]

                d2 = (diff @ Σinv * diff).sum(-1)
                dists.append(d2)

            dists = torch.stack(dists, dim=1)
            preds = dists.argmin(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        return 100.0 * correct / total
