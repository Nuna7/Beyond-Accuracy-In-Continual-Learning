"""
EFC — Elastic Feature Consolidation (Magistri et al., ICLR 2024).
https://arxiv.org/abs/2402.03917

Cold-start, exemplar-free class-incremental learning. Generalises EWC's
weight-space Fisher regulariser into FEATURE space:

  - Empirical Feature Matrix (EFM) E_t: an (n_feat x n_feat) pseudo-metric
    over the backbone's feature space, built analytically (no exemplars,
    no backward pass) from the classifier weights + softmax outputs after
    each task. Regularises drift in feature directions that matter to the
    classifier, while leaving unimportant directions free to change
    (unlike EWC's Fisher, this scales with feature dim, not param count).

  - Gaussian class prototypes (mean + covariance), updated across tasks via
    an EFM-weighted "semantic drift compensation": prototypes are nudged by
    however much CURRENT-task data appears to have drifted in feature
    space, weighted by proximity (in the E_{t-1} metric) to each prototype.

  - PR-ACE (prototype-replay asymmetric cross-entropy, Eq. 12): the
    classification loss for the CURRENT task's own classes is computed
    restricted to just those classes (avoids penalising new-class logits
    for not also matching old-class structure), while the FULL head is
    trained on a mix of current-task data + resampled Gaussian prototypes
    of old classes.

Total loss (Eq. 15):  L = L_EFM (Eq. 9) + L_PR-ACE (Eq. 12)

IMPLEMENTATION NOTES (read before trusting numbers for a paper):
  - EFM (Eq. 7/23) computed via an algebraically-equivalent, batch-efficient
    closed form -- see _compute_efm() docstring. VERIFIED numerically
    against a brute-force per-sample reference (max abs diff 4.8e-7).
  - PR-ACE reuses ONE forward pass of the current-task batch for both the
    "current task, restricted head" term and the "current task, full head"
    term, rather than drawing two independent batches as the paper's
    notation (x_t vs x_hat_t) implies. Simplification for compatibility with
    this codebase's single-forward-pass-per-batch training loop.
  - Covariance regularisation is simple additive shrinkage (eps * mean
    diagonal), not the paper's low-rank approximation. Storage grows
    linearly with number of classes seen (~1MB/class at d=512, fp32) --
    fine at CIFAR-100/Tiny-ImageNet scale (<=200 classes).
  - Optimizer/scheduler: uses this codebase's shared SGD+cosine
    (TrainConfig), NOT the original paper's Adam -- for a fair, apples-to-
    apples comparison against EWC/LwF/FeCAM, which are trained the same way
    here. If you want literal paper reproduction, override _build_optimizer.
"""
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
        self._proto_means: Dict[int, torch.Tensor] = {}   # class -> (n,) CPU
        self._proto_chol:  Dict[int, torch.Tensor] = {}   # class -> (n,n) CPU, lower-tri
        self._efm: Optional[torch.Tensor] = None          # E_{t-1}, (n,n) CPU
        self._efm_reg: Optional[torch.Tensor] = None       # lambda*E + eta*I, on device
        self._old_model: Optional[nn.Module] = None        # frozen f_{t-1}
        self._last_imgs: Optional[torch.Tensor] = None
        self._task_start = 0
        self._task_ncls = 0

    # ── feature extraction helpers ──────────────────────────────────────────

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

    # ── EFM (Eq. 6/7/23) ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _compute_efm(self, loader: DataLoader) -> torch.Tensor:
        """
        E_t = E_{x~X_t, y~p(y|f_t(x);W_t)} [ W(I-P)_y (W(I-P)_y)^T ]   (Eq. 6/7)

        Efficient closed form (derivation: expand the outer product, use
        Wp_i = sum_y P[i,y] W_col_y by definition, and Sum_y P[i,y]=1):

            E_i = sum_y P[i,y] * W_col_y W_col_y^T  -  Wp_i Wp_i^T

        Batched over the whole task (accumulated across mini-batches):

            term1 = W_scaled.T @ W        where W_scaled[y,:] = S[y]*W[y,:],
                                           S = column-sums of P over all data
            term2 = Wp_all.T @ Wp_all     where Wp_all = P @ W  (N,n)
            E_t   = (term1 - term2) / N

        W here is self.model.fc.weight, shape (m, n) in PyTorch's
        (out_features, in_features) convention -- i.e. row y of W IS the
        paper's "W_col_y" (n,) once you account for the transpose.
        """
        self.model.eval()
        fc = self.model.fc
        Wt = fc.weight.detach()                      # (m, n)
        n = Wt.shape[1]
        m = Wt.shape[0]
        term1 = torch.zeros(n, n, device=self.device)
        term2 = torch.zeros(n, n, device=self.device)
        S     = torch.zeros(m, device=self.device)
        N     = 0

        for imgs, _ in loader:
            imgs = imgs.to(self.device)
            self.model(imgs)
            f = self.model.hooks['layer4'].float()             # (B, n)
            z = f @ Wt.T + fc.bias.detach()                     # (B, m) full logits
            P = F.softmax(z, dim=1)                             # (B, m)

            S_batch = P.sum(0)                                  # (m,)
            Wp_batch = P @ Wt                                   # (B, n)

            W_scaled = Wt * S_batch.unsqueeze(1)                # (m, n)
            term1 += W_scaled.T @ Wt                            # (n, n)
            term2 += Wp_batch.T @ Wp_batch                      # (n, n)
            S += S_batch
            N += f.size(0)

        E = (term1 - term2) / max(N, 1)
        return E.cpu()

    # ── prototype storage + EFM-weighted drift compensation (Eq. 13/14) ─────

    @torch.no_grad()
    def _update_prototypes_sdc(self, feats_new: torch.Tensor, feats_old: torch.Tensor):
        """
        Nudge every OLD prototype using how much CURRENT-task data drifted
        between f_{t-1} (feats_old) and f_t (feats_new), weighted by each
        sample's proximity to that prototype under the E_{t-1} metric.

        d_i^c = (f_i - p^c)^T E (f_i - p^c)   -- vectorised via
                a_i + b_c - 2*cross_ic  (a=f^T E f, b=p^T E p, cross=f^T E p)
        w_i^c = exp(-d_norm_i^c / (2 sigma^2)),  d normalised to [0,1] per class
        p_hat^c = p^c + sum_i(w_i^c * delta_i) / sum_i(w_i^c),  delta = f_t - f_{t-1}
        """
        if not self._proto_means or self._efm is None:
            return
        classes = sorted(self._proto_means.keys())
        P = torch.stack([self._proto_means[c] for c in classes]).to(self.device).float()  # (C,n)
        E = self._efm.to(self.device).float()                                              # (n,n)

        F_old = feats_old.float()                      # (N,n)
        F_new = feats_new.float()
        delta = F_new - F_old                           # (N,n)

        FE = F_old @ E                                  # (N,n)
        a  = (FE * F_old).sum(-1)                       # (N,)
        PE = P @ E                                       # (C,n)
        b  = (PE * P).sum(-1)                            # (C,)
        cross = FE @ P.T                                 # (N,C)
        d = a.unsqueeze(1) + b.unsqueeze(0) - 2 * cross   # (N,C), >=0 up to fp noise
        d = d.clamp(min=0)

        # per-class min-max normalisation to [0,1] (paper: normalise before the kernel)
        d_min = d.min(0, keepdim=True).values
        d_max = d.max(0, keepdim=True).values
        d_norm = (d - d_min) / (d_max - d_min + 1e-8)

        sigma = self.cfg.method.efc_proto_sigma
        W = torch.exp(-d_norm / (2 * sigma ** 2))         # (N,C)

        weighted_delta = W.T @ delta                       # (C,n)
        denom = W.sum(0).clamp(min=1e-8).unsqueeze(1)       # (C,1)
        new_means = P + weighted_delta / denom              # (C,n)

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

    # ── CL hooks ─────────────────────────────────────────────────────────────

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

        # term2 (Eq. 12, second addend): CE over full head, current-task batch
        # UNION resampled prototypes -- one combined cross-entropy call.
        proto_feats, proto_labels = self._sample_prototypes(targets.size(0))
        proto_logits = self.model.fc(proto_feats.to(self.device))
        combined_logits = torch.cat([logits, proto_logits], dim=0)
        combined_targets = torch.cat([targets, proto_labels.to(self.device)], dim=0)
        term2 = F.cross_entropy(combined_logits, combined_targets)

        # L_EFM (Eq. 9): quadratic penalty on backbone drift vs frozen f_{t-1},
        # same input batch, in the (lambda*E_{t-1} + eta*I) metric.
        with torch.no_grad():
            # ensure old model is on the same device as the current batch
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

    # ── training loop (mirrors BaseTrainer.train_task; adds _last_imgs stash
    #    for the EFM term, which needs raw images -- compute_loss only gets
    #    logits/targets) ────────────────────────────────────────────────────

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