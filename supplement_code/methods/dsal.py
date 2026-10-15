import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from .base_trainer import BaseTrainer

def _activation(name: str):
    return {'relu': torch.relu, 'tanh': torch.tanh}[name]

class DSALTrainer(BaseTrainer):

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self._X_B: Optional[torch.Tensor] = None
        self._A_M: Optional[torch.Tensor] = None
        self._B_M: Optional[torch.Tensor] = None
        self._A_C: Optional[torch.Tensor] = None
        self._B_C: Optional[torch.Tensor] = None
        self._W_M: Optional[torch.Tensor] = None
        self._W_C: Optional[torch.Tensor] = None
        self._backbone_frozen = False

    @torch.no_grad()
    def _extract_features(self, loader: DataLoader):
        self.model.eval()
        all_f, all_y = [], []
        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            self.model(imgs)
            all_f.append(self.model.hooks['layer4'].cpu())
            all_y.append(labels)
        return torch.cat(all_f), torch.cat(all_y)

    def _init_buffer(self, d_in: int):
        gen = torch.Generator().manual_seed(self.cfg.seed)
        d_buf = self.cfg.method.dsal_buffer_size
        self._X_B = (torch.randn(d_in, d_buf, generator=gen) / (d_in ** 0.5))

    def _buffer_acts(self, feats: torch.Tensor, sigma_name: str):
        raw = feats.float() @ self._X_B.to(feats.device)
        return _activation(sigma_name)(raw)

    def _solve(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        d = A.shape[0]
        gamma = self.cfg.method.dsal_gamma
        R = torch.linalg.inv(A + gamma * torch.eye(d, device=A.device))
        return R @ B

    def _init_stream(self, X: torch.Tensor, Y_onehot: torch.Tensor):
        """First-task fit (Eq. 4/5): A = X^T X, B = X^T Y."""
        A = X.T @ X
        B = X.T @ Y_onehot
        return A, B

    def _grow_stream(self, A, B, X_k: torch.Tensor, Y_k_onehot: torch.Tensor):
        """
        Add a new task's contribution. A grows by X_k^T X_k (same dim).
        B grows in COLUMNS: new task's classes never appear in old samples'
        labels and vice-versa, so old columns of B need no update -- just
        concatenate the new block. Equivalent to Eq. 11/12's block-diagonal
        label structure (Eq. 6), reformulated (see docstring).
        """
        A = A + X_k.T @ X_k
        new_cols = X_k.T @ Y_k_onehot
        B = torch.cat([B, new_cols], dim=1)
        return A, B

    def _init_analytic_head(self, task_id: int, train_loader: DataLoader,
                            classes_this_task: int):
        feats, labels = self._extract_features(train_loader)
        feats, labels = feats.to(self.device), labels.to(self.device)
        self._init_buffer(feats.shape[1])
        self._X_B = self._X_B.to(self.device)

        act_main = self.cfg.method.dsal_activation_main
        act_comp = self.cfg.method.dsal_activation
        X_M = self._buffer_acts(feats, act_main)
        X_C = self._buffer_acts(feats, act_comp)

        Y = torch.zeros(feats.size(0), classes_this_task, device=self.device)
        Y[torch.arange(feats.size(0)), labels] = 1.0

        self._A_M, self._B_M = self._init_stream(X_M, Y)
        self._W_M = self._solve(self._A_M, self._B_M)

        Y_tilde = Y - X_M @ self._W_M
        self._A_C, self._B_C = self._init_stream(X_C, Y_tilde)
        self._W_C = self._solve(self._A_C, self._B_C)

        self.model.freeze_backbone()
        self._backbone_frozen = True
        print(f"  [DS-AL] Analytic head initialised on task 0 "
              f"(d_buffer={self._X_B.shape[1]}), backbone frozen.")

    def _incremental_step(self, task_id: int, train_loader: DataLoader,
                          task_start: int, classes_this_task: int):
        feats, labels = self._extract_features(train_loader)
        feats = feats.to(self.device)
        local_labels = (labels - task_start).to(self.device)

        act_main = self.cfg.method.dsal_activation_main
        act_comp = self.cfg.method.dsal_activation
        X_M = self._buffer_acts(feats, act_main)
        X_C = self._buffer_acts(feats, act_comp)

        Y_k = torch.zeros(feats.size(0), classes_this_task, device=self.device)
        Y_k[torch.arange(feats.size(0)), local_labels] = 1.0

        self._A_M, self._B_M = self._grow_stream(self._A_M, self._B_M, X_M, Y_k)
        self._W_M = self._solve(self._A_M, self._B_M)

        W_M_new_cols = self._W_M[:, task_start:task_start + classes_this_task]
        Y_tilde_new = Y_k - X_M @ W_M_new_cols

        self._A_C, self._B_C = self._grow_stream(self._A_C, self._B_C, X_C, Y_tilde_new)
        self._W_C = self._solve(self._A_C, self._B_C)

    @torch.no_grad()
    def _eval_accuracy(self, loader: DataLoader) -> float:
        if self._W_M is None:
            return super()._eval_accuracy(loader)
        self.model.eval()
        C_ratio = self.cfg.method.dsal_compensation_ratio
        act_main = self.cfg.method.dsal_activation_main
        act_comp = self.cfg.method.dsal_activation
        correct = total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.model(imgs)
            feats = self.model.hooks['layer4'].float()
            X_M = self._buffer_acts(feats, act_main)
            X_C = self._buffer_acts(feats, act_comp)
            y_hat = X_M @ self._W_M + C_ratio * (X_C @ self._W_C)
            preds = y_hat.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return 100.0 * correct / max(total, 1)

    def train_task(self, task_id, train_loader, test_loader, classes_this_task):
        self.task_id = task_id
        self.seen_classes += classes_this_task
        task_start = self.seen_classes - classes_this_task
        t0 = time.time()

        if task_id == 0:

            self.seen_classes -= classes_this_task
            metrics = BaseTrainer.train_task(
                self, task_id, train_loader, test_loader, classes_this_task)

            if getattr(self.cfg, 'save_snapshots', False) and self._snapshot_paths:
                last_path = self._snapshot_paths[-1]
                ckpt = torch.load(last_path, map_location='cpu', weights_only=False)
                ckpt['dsal_X_B'] = self._X_B.cpu()
                ckpt['dsal_W_M'] = self._W_M.cpu()
                ckpt['dsal_W_C'] = self._W_C.cpu()
                torch.save(ckpt, last_path)
        else:
            if not self._backbone_frozen:
                raise RuntimeError("DS-AL: backbone was never frozen after task 0 -- "
                                   "was after_task/_init_analytic_head skipped?")
            self._incremental_step(task_id, train_loader, task_start, classes_this_task)
            acc = self._eval_accuracy(test_loader)
            metrics = {'task_id': task_id, 'train_loss': 0.0,
                      'test_acc': acc, 'time_s': time.time() - t0}
            self._results.append(metrics)
            print(f"  -> Task {task_id} | acc (DS-AL analytic): {acc:.2f}% | "
                  f"time: {metrics['time_s']:.1f}s  (no gradient steps)")

            if getattr(self.cfg, 'save_snapshots', False):
                method_tag = getattr(self.cfg.method, 'original_method_name',
                                     getattr(self.cfg.method, 'name', 'dsal'))
                run_tag = (f"{method_tag}_{self.cfg.data.dataset}"
                          f"_t{self.cfg.data.num_tasks}_s{self.cfg.seed}")
                snap_dir = os.path.join(self.cfg.save_dir, 'snapshots', run_tag)
                os.makedirs(snap_dir, exist_ok=True)
                snap_path = os.path.join(snap_dir, f'task{task_id}.pt')

                torch.save({
                    'task_id': task_id,
                    'state_dict': self.model.state_dict(),
                    'seen_classes': self.seen_classes,
                    'class_order': getattr(self, '_class_order', None),
                    'dsal_X_B': self._X_B.cpu(),
                    'dsal_W_M': self._W_M.cpu(),
                    'dsal_W_C': self._W_C.cpu(),
                }, snap_path)
                self._snapshot_paths.append(snap_path)
        return metrics

    def after_task(self, task_id: int, train_loader: DataLoader):

        if task_id == 0:
            self._init_analytic_head(task_id, train_loader, self._last_classes_this_task)
