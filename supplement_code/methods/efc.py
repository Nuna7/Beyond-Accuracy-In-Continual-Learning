import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional

from .base_trainer import BaseTrainer

class EFCTrainer(BaseTrainer):

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self._proto_means: Dict[int, torch.Tensor] = {}
        self._proto_chol:  Dict[int, torch.Tensor] = {}
        self._efm: Optional[torch.Tensor] = None
        self._efm_reg: Optional[torch.Tensor] = None
        self._old_model: Optional[nn.Module] = None
        self._last_imgs: Optional[torch.Tensor] = None
        self._task_start = 0
        self._task_ncls = 0

    @torch.no_grad()
    def _extract_features(self, loader: DataLoader, model: Optional[nn.Module] = None):
        """(feats, labels) using layer4 (post-GAP) activations. model=None -> self.model."""
        m = model if model is not None else self.model
        m.eval()
        m.to(self.device)
        all_f, all_y = [], []
        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            m(imgs)
            all_f.append(m.hooks['layer4'].cpu())
            all_y.append(labels)
        return torch.cat(all_f), torch.cat(all_y)

    @torch.no_grad()
    def _compute_efm(self, loader: DataLoader) -> torch.Tensor:
        self.model.eval()
        fc = self.model.fc
        Wt = fc.weight.detach()
        n = Wt.shape[1]
        m = Wt.shape[0]
        term1 = torch.zeros(n, n, device=self.device)
        term2 = torch.zeros(n, n, device=self.device)
        S     = torch.zeros(m, device=self.device)
        N     = 0

        for imgs, _ in loader:
            imgs = imgs.to(self.device)
            self.model(imgs)
            f = self.model.hooks['layer4'].float()
            z = f @ Wt.T + fc.bias.detach()
            P = F.softmax(z, dim=1)

            S_batch = P.sum(0)
            Wp_batch = P @ Wt

            W_scaled = Wt * S_batch.unsqueeze(1)
            term1 += W_scaled.T @ Wt
            term2 += Wp_batch.T @ Wp_batch
            S += S_batch
            N += f.size(0)

        E = (term1 - term2) / max(N, 1)
        return E.cpu()

    @torch.no_grad()
    def _update_prototypes_sdc(self, feats_new: torch.Tensor, feats_old: torch.Tensor):
        if not self._proto_means or self._efm is None:
            return
        classes = sorted(self._proto_means.keys())
        P = torch.stack([self._proto_means[c] for c in classes]).to(self.device).float()
        E = self._efm.to(self.device).float()

        F_old = feats_old.float()
        F_new = feats_new.float()
        delta = F_new - F_old

        FE = F_old @ E
        a  = (FE * F_old).sum(-1)
        PE = P @ E
        b  = (PE * P).sum(-1)
        cross = FE @ P.T
        d = a.unsqueeze(1) + b.unsqueeze(0) - 2 * cross
        d = d.clamp(min=0)

        d_min = d.min(0, keepdim=True).values
        d_max = d.max(0, keepdim=True).values
        d_norm = (d - d_min) / (d_max - d_min + 1e-8)

        sigma = self.cfg.method.efc_proto_sigma
        W = torch.exp(-d_norm / (2 * sigma ** 2))

        weighted_delta = W.T @ delta
        denom = W.sum(0).clamp(min=1e-8).unsqueeze(1)
        new_means = P + weighted_delta / denom

        for i, c in enumerate(classes):
            self._proto_means[c] = new_means[i].cpu()

    @torch.no_grad()
    def _store_new_prototypes(self, feats: torch.Tensor, labels: torch.Tensor,
                              task_start: int, n_cls: int):
        eps_scale = self.cfg.method.efc_cov_eps
        for c_local in range(n_cls):
            c_global = task_start + c_local
            mask = labels == c_global
            if mask.sum() < 2:
                continue
            X = feats[mask].float()
            mu = X.mean(0)
            Z = X - mu
            cov = (Z.T @ Z) / max(X.size(0) - 1, 1)
            eps = eps_scale * cov.diagonal().mean().clamp(min=1e-6)
            cov = cov + eps * torch.eye(cov.size(0))
            try:
                chol = torch.linalg.cholesky(cov)
            except Exception:
                chol = torch.linalg.cholesky(cov + 1e-2 * torch.eye(cov.size(0)))
            self._proto_means[c_global] = mu.cpu()
            self._proto_chol[c_global]  = chol.cpu()

    def _sample_prototypes(self, n: int):
        classes = list(self._proto_means.keys())
        idx = torch.randint(0, len(classes), (n,))
        feats, labels = [], []
        d = next(iter(self._proto_means.values())).shape[0]
        for i in idx.tolist():
            c = classes[i]
            z = torch.randn(d)
            x = self._proto_means[c] + self._proto_chol[c] @ z
            feats.append(x)
            labels.append(c)
        return torch.stack(feats), torch.tensor(labels, dtype=torch.long)

    def before_task(self, task_id: int, train_loader: DataLoader):
        if task_id > 0 and self._efm is not None:
            n = self._efm.shape[0]
            self._efm_reg = (
                self.cfg.method.efc_lambda_efm * self._efm.to(self.device)
                + self.cfg.method.efc_eta * torch.eye(n, device=self.device)
            )

    def compute_loss(self, logits, targets, task_id):
        task_start, n_cls = self._task_start, self._task_ncls
        local_targets = targets - task_start
        logits_task = logits[:, task_start:task_start + n_cls]
        term1 = F.cross_entropy(logits_task, local_targets)

        if task_id == 0 or not self._proto_means:
            return term1

        proto_feats, proto_labels = self._sample_prototypes(targets.size(0))
        proto_logits = self.model.fc(proto_feats.to(self.device))
        combined_logits = torch.cat([logits, proto_logits], dim=0)
        combined_targets = torch.cat([targets, proto_labels.to(self.device)], dim=0)
        term2 = F.cross_entropy(combined_logits, combined_targets)

        with torch.no_grad():

            self._old_model.to(self.device)
            self._old_model(self._last_imgs)
        f_new = self.model.hooks['layer4']
        f_old = self._old_model.hooks['layer4'].detach()
        diff = f_new - f_old
        l_efm = (diff @ self._efm_reg * diff).sum(-1).mean()

        return term1 + term2 + l_efm

    def after_task(self, task_id: int, train_loader: DataLoader):
        classes_per_task = self._task_ncls
        task_start = self._task_start
        self.model.eval()
        self.model.detach_hooks = True

        feats_new, labels = self._extract_features(train_loader)
        labels_dev = labels.to(self.device)
        feats_new_dev = feats_new.to(self.device)

        if task_id > 0 and self._proto_means and self._old_model is not None:
            feats_old, _ = self._extract_features(train_loader, self._old_model)
            self._update_prototypes_sdc(feats_new_dev, feats_old.to(self.device))

        self._store_new_prototypes(feats_new, labels, task_start, classes_per_task)

        self._efm = self._compute_efm(train_loader)

        snap = copy.deepcopy(self.model)
        snap.eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        snap.to('cpu')
        self._old_model = snap

        print(f"  [EFC] Task {task_id}: {len(self._proto_means)} prototypes, "
              f"EFM trace={self._efm.trace().item():.2f}")

    def _build_optimizer(self):
        if self.task_id > 0 and getattr(self.cfg.method, 'efc_optimizer', 'shared') == 'paper':
            return torch.optim.Adam(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=1e-4, weight_decay=2e-4)
        return super()._build_optimizer()

    def train_task(self, task_id, train_loader, test_loader, classes_this_task):
        self.task_id = task_id
        self.seen_classes += classes_this_task
        self._task_start = self.seen_classes - classes_this_task
        self._task_ncls  = classes_this_task

        self.model.expand_classifier(self.seen_classes)
        self.model.to(self.device)
        self.model.detach_hooks = False

        self.before_task(task_id, train_loader)
        optimizer = self._build_optimizer()
        epochs = (self.cfg.train.init_epochs if task_id == 0
                  else self.cfg.train.epochs_per_task)
        scheduler = self._build_scheduler(optimizer, epochs)

        import time
        t0 = time.time()
        train_losses = []
        pbar = tqdm(range(epochs), desc=f"Task {task_id} [EFC]", leave=True)

        for epoch in pbar:
            self.model.train()
            epoch_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self._last_imgs = imgs
                logits = self.model(imgs)
                loss = self.compute_loss(logits, labels, task_id)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                epoch_loss += loss.item()

            if scheduler is not None:
                scheduler.step()
            epoch_loss /= len(train_loader)
            train_losses.append(epoch_loss)

            if (epoch + 1) % self.cfg.log_every == 0:
                self.model.detach_hooks = True
                acc = self._eval_accuracy(test_loader)
                self.model.detach_hooks = False
                pbar.set_postfix(loss=f"{epoch_loss:.3f}", acc=f"{acc:.2f}")

        self.model.detach_hooks = True
        self._last_classes_this_task = classes_this_task
        self.after_task(task_id, train_loader)

        if getattr(self.cfg, 'save_snapshots', False):
            import os
            method_tag = getattr(self.cfg.method, 'original_method_name',
                                 getattr(self.cfg.method, 'name', 'efc'))
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
            }, snap_path)
            self._snapshot_paths.append(snap_path)

        acc = self._eval_accuracy(test_loader)
        metrics = {
            'task_id': task_id,
            'train_loss': float(sum(train_losses) / len(train_losses)),
            'test_acc': acc,
            'time_s': time.time() - t0,
        }
        self._results.append(metrics)
        print(f"  -> Task {task_id} | acc: {acc:.2f}% | time: {metrics['time_s']:.1f}s")
        return metrics
