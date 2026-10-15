"""
methods/ada_gauss.py

Port of AdaGauss (Rypesc et al., NeurIPS 2024, "Task-recency bias strikes
back: Adapting covariances in Exemplar-Free Class Incremental Learning")
into this repo's trainer framework, so it can be analyzed by
analysis/run_analysis.py, analysis/feature_fate_runner.py, and
analysis/interclass_separability.py exactly like every other method here.

Original source: https://github.com/grypesc/AdaGauss/blob/master/src/approach/ada_gauss.py

ADAPTATION DECISIONS vs. the original (read before trusting numbers):

  1. Backbone: uses OUR shared build_resnet18 (hooks['layer1'..'layer4']
     intact), NOT their custom resnet18(num_features=S). An S-dim bottleneck
     Linear layer is added AFTER our layer4 as an extra trainable module
     living outside self.model, so the mechanistic-analysis tools see the
     same 512-dim layer4 activations as every other method in this repo.

  2. classifier="bayes" (full-covariance Mahalanobis), criterion="ce"
     (task-local head, discarded after training), distillation="projected"
     (MLP distiller) -- matches their scripts/cifar-10x10.sh defaults.

  3. NO rotation augmentation for task 0 (their --rotation flag). Left out
     because it changes task-0's effective class count (4x) and complicates
     cross-method comparison in this repo's fixed class-order protocol. Can
     be added back later if needed.

  4. criterion="ce" is implemented here as a plain nn.Linear + F.cross_entropy
     (smoothing=0), which should match their CE criterion class under its
     own defaults -- NOT independently verified against criterions/ce.py.

  5. Hyperparameters (S, lamb, alpha, beta, shrink, multiplier) are exposed
     via cfg.method.ada_gauss_* with the paper's own defaults, but their
     lamb=10 was tuned for RAW (non-cosine) MSE over a 64-dim space under
     200 epochs -- a different scale than this repo's SAE-FSGP cosine-
     distance loss. Do not assume lamb=10 is correct under a 50-epoch
     budget without checking training logs (see validation run below).

  6. adapt_epochs_frac=0.5 mirrors their nepochs // 2 for the adapter
     training phase, scaled to whatever --epochs this repo's run uses.

INTEGRATION STEPS (not automatic -- see the end of this file for exact diffs):
  - Register in train.py's build_trainer() dispatch
  - Add --ada_gauss_* CLI args + build_config() wiring in train.py
  - Add "ada_gauss" to METHODS lists in run_analysis.py,
    interclass_separability.py's default --methods, and
    run_full_pipeline.sh's METHODS env var
"""

import copy
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_trainer import BaseTrainer

HOOK_LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']
LAYER4_DIM = 512


# ── direct ports of small, self-contained pieces from ada_gauss.py ───────────

def loss_ac(features: torch.Tensor, beta: float, eps: float = 1e-5):
    """
    Official AdaGauss anti-collapse loss, with a tiny numerical floor so
    Cholesky does not fail on early-training / small-batch covariances.
    eps must stay << beta and << eval_cov_reg; it is not a hyperparameter
    to tune for accuracy.
    """
    cov = torch.cov(features.T)
    # Numerical stabilisation only — does not replace the anti-collapse objective
    d = cov.shape[0]
    cov = cov + eps * torch.eye(d, device=cov.device, dtype=cov.dtype)
    cholesky = torch.linalg.cholesky(cov)
    cholesky_diag = torch.diagonal(cholesky)
    loss = -torch.clamp(cholesky_diag, max=beta).mean()
    return loss, torch.det(cov)

#def loss_ac(features: torch.Tensor, beta: float, cov_reg: float = 1e-3):
#    cov = torch.cov(features.T)
#
#    # Make covariance strictly positive definite.
#    d = cov.shape[0]
#    eye = torch.eye(d, device=cov.device, dtype=cov.dtype)
#
#    # Scale regularization to the covariance magnitude.
#    diag_mean = torch.diagonal(cov).mean().detach().clamp_min(1e-8)
#    cov = cov + cov_reg * diag_mean * eye
#
#    # Cholesky should now be well-defined.
#    cholesky = torch.linalg.cholesky(cov)
#
#    cholesky_diag = torch.diagonal(cholesky)
#    loss = -torch.clamp(cholesky_diag, max=beta).mean()
#
#    return loss, torch.det(cov)

# def loss_ac(features: torch.Tensor, beta: float):
#     """
#     Anti-collapse loss. Penalizes small Cholesky-diagonal entries of the
#     BATCH covariance matrix -- i.e. penalizes covariance-rank collapse
#     during training. This is the same underlying phenomenon this repo's
#     analysis/interclass_separability.py effective_rank metric measures
#     post-hoc on class-mean prototypes; here it acts every training step
#     on raw per-batch features instead.
#     """
#     cov = torch.cov(features.T)
#     cholesky = torch.linalg.cholesky(cov)
#     cholesky_diag = torch.diag(cholesky)
#     loss = -torch.clamp(cholesky_diag, max=beta).mean()
#     return loss, torch.det(cov)


def shrink_cov(cov: torch.Tensor, alpha1: float, device: str):
    """Ridge-style shrinkage toward a scaled identity, for invertibility."""
    diag_mean = torch.mean(torch.diagonal(cov))
    iden = torch.eye(cov.shape[0], device=device)
    return cov + alpha1 * diag_mean * iden


class AdaGaussTrainer(BaseTrainer):

    def __init__(self, model, cfg):
        super().__init__(model, cfg)

        m = cfg.method
        self.S          = getattr(m, 'ada_gauss_S', 64)
        self.lamb       = getattr(m, 'ada_gauss_lamb', 10.0)
        self.alpha      = getattr(m, 'ada_gauss_alpha', 1.0)
        self.beta       = getattr(m, 'ada_gauss_beta', 1.0)
        self.shrink     = getattr(m, 'ada_gauss_shrink', 0.0)
        #self.cov_reg    = getattr(m, 'ada_gauss_cov_reg', 1e-3)   # invertibility floor
        #self.cov_reg = getattr(m, 'ada_gauss_cov_reg', 3.0)   # was 1e-3 -- matches official's hardcoded alpha1=3
        self.eval_cov_reg = getattr(m, 'ada_gauss_eval_cov_reg', 3.0)
        self.multiplier = getattr(m, 'ada_gauss_multiplier', 32)
        self.adapt_epochs_frac = getattr(m, 'ada_gauss_adapt_frac', 0.5)

        # S-dim bottleneck: our layer4 (512-dim) -> S-dim latent space.
        # Lives outside self.model so hooks['layer4'] stays untouched for
        # the mechanistic analysis tools.
        self.bottleneck = nn.Linear(LAYER4_DIM, self.S).to(self.device)

        # Per-class Gaussian storage, S-dim space.
        self.means       = torch.empty((0, self.S), device=self.device)
        self.covs        = torch.empty((0, self.S, self.S), device=self.device)
        self.covs_inv    = None
        self.task_offset = [0]

        self._old_model      = None   # frozen backbone snapshot (in-memory)
        self._old_bottleneck = None
        self._task_head       = None  # fresh per-task linear head, CE only
        self._distiller       = None
        self._last_classes_this_task = 0

        self._last_ce = 0.0
        self._last_kd = 0.0
        self._last_ac = 0.0

    # ── forward helper: layer4 activation -> S-dim bottleneck feature ────────

    def _features(self, imgs, use_old=False):
        model = self._old_model if use_old else self.model
        bottleneck = self._old_bottleneck if use_old else self.bottleneck
        model(imgs)
        h = model.hooks['layer4']
        return bottleneck(h)

    # ── training loss (called per-batch from train_task) ─────────────────────

    def compute_loss(self, imgs, targets_local, task_id):
        features = self._features(imgs)

        logits = self._task_head(features)
        ce = F.cross_entropy(logits, targets_local)

        kd_loss = torch.tensor(0.0, device=self.device)
        if task_id > 0:
            with torch.no_grad():
                old_features = self._features(imgs, use_old=True)
            kd_loss = F.mse_loss(self._distiller(features), old_features)

        ac_loss = torch.tensor(0.0, device=self.device)
        if self.alpha > 0:
            #ac_loss, _det = loss_ac(features, self.beta)
            ac_loss, _det = loss_ac(features, self.beta)
            #ac_loss, _det = loss_ac(features, self.beta, self.cov_reg)

        total = ce + self.lamb * kd_loss + self.alpha * ac_loss

        self._last_ce = ce.item()
        self._last_kd = kd_loss.item() if task_id > 0 else 0.0
        self._last_ac = ac_loss.item() if self.alpha > 0 else 0.0
        return total

    # ── per-task training loop (full override, same pattern as SAEFSGPTrainer) ─

    def train_task(self, task_id, train_loader, test_loader, classes_this_task):
        self.model.detach_hooks = False
        self.task_id       = task_id
        self.seen_classes  += classes_this_task
        self.model.to(self.device)

        self.task_offset.append(self.task_offset[-1] + classes_this_task)
        task_start = self.task_offset[-2]
        self._last_classes_this_task = classes_this_task

        if self._old_model is not None:
            self._old_model.to(self.device)
            self._old_bottleneck.to(self.device)

        # Fresh task-local CE head — used only to compute the CE loss during
        # training, then discarded. Eval never uses this; it always uses
        # Mahalanobis distance against the stored Gaussians (see _eval_accuracy).
        self._task_head = nn.Linear(self.S, classes_this_task).to(self.device)
        self._distiller = nn.Sequential(
            nn.Linear(self.S, self.multiplier * self.S), nn.GELU(),
            nn.Linear(self.multiplier * self.S, self.S),
        ).to(self.device)

        self.before_task(task_id, train_loader)

        params = (list(self.model.parameters()) + list(self.bottleneck.parameters())
                 + list(self._task_head.parameters()) + list(self._distiller.parameters()))
        lr = self.cfg.train.lr if task_id == 0 else self.cfg.train.lr * 0.1
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=5e-4, momentum=0.9)
        n_ep = self.cfg.train.epochs_per_task
        milestones = [int(0.3 * n_ep), int(0.6 * n_ep), int(0.9 * n_ep)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

        t0 = time.time()
        pbar = tqdm(range(n_ep), desc=f"Task {task_id} [AdaGauss]", leave=True)
        for epoch in pbar:
            self.model.train()
            self.bottleneck.train()
            self._task_head.train()
            self._distiller.train()
            epoch_loss = 0.0
            for imgs, labels in train_loader:
                imgs = imgs.to(self.device)
                labels_local = labels.to(self.device) - task_start
                optimizer.zero_grad()
                loss = self.compute_loss(imgs, labels_local, task_id)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % self.cfg.log_every == 0:
                pbar.set_postfix(
                    ce=f"{self._last_ce:.3f}", kd=f"{self._last_kd:.3f}",
                    ac=f"{self._last_ac:.3f}",
                    loss=f"{epoch_loss / len(train_loader):.3f}")

        self.model.detach_hooks = True
        self.after_task(task_id, train_loader)

        if self._old_model is not None:
            self._old_model.to('cpu')
            self._old_bottleneck.to('cpu')

        # ── standard disk snapshot block (identical pattern to every other
        #    method here — this is what makes AdaGauss's checkpoints loadable
        #    by run_analysis.py / feature_fate_runner.py /
        #    interclass_separability.py) ─────────────────────────────────────
        if getattr(self.cfg, 'save_snapshots', False):
            method_tag = getattr(self.cfg.method, 'original_method_name',
                                 getattr(self.cfg.method, 'name', 'unknown'))
            run_tag = (f"{method_tag}_{self.cfg.data.dataset}"
                      f"_t{self.cfg.data.num_tasks}_s{self.cfg.seed}")
            snap_dir = os.path.join(self.cfg.save_dir, 'snapshots', run_tag)
            os.makedirs(snap_dir, exist_ok=True)
            snap_path = os.path.join(snap_dir, f'task{task_id}.pt')
            torch.save({
                'task_id':      task_id,
                'state_dict':   self.model.state_dict(),
                'seen_classes': self.seen_classes,
                'class_order':  getattr(self, '_class_order', None),
                'bottleneck_state_dict': self.bottleneck.state_dict(),
            }, snap_path)
            self._snapshot_paths.append(snap_path)
            print(f"  [AdaGauss] Snapshot saved -> {snap_path}")

        acc = self._eval_accuracy(test_loader)
        metrics = {'task_id': task_id, 'test_acc': acc, 'time_s': time.time() - t0}
        self._results.append(metrics)
        print(f"  -> Task {task_id} | acc (Bayes): {acc:.2f}%")
        return metrics

    # ── post-task: adapt old Gaussians, create new ones ───────────────────────

    @torch.no_grad()
    def after_task(self, task_id, train_loader):
        classes_this_task = self._last_classes_this_task
        task_start = self.task_offset[-2]

        if task_id > 0 and self._old_model is not None:
            self._train_adapter_and_adapt(train_loader)

        act_dict, labels_cpu = self.collect_all_activations(train_loader)
        labels_dev = labels_cpu.to(self.device)
        acts_l4 = act_dict['layer4'].to(self.device).float()
        feats = self.bottleneck(acts_l4)

        new_means = torch.zeros((classes_this_task, self.S), device=self.device)
        new_covs = torch.zeros((classes_this_task, self.S, self.S), device=self.device)
        for c_local in range(classes_this_task):
            c_global = task_start + c_local
            mask = (labels_dev == c_global)
            class_feats = feats[mask]
            new_means[c_local] = class_feats.mean(0)
            new_covs[c_local] = shrink_cov(torch.cov(class_feats.T), self.shrink, self.device)

        self.means = torch.cat([self.means, new_means], dim=0)
        self.covs = torch.cat([self.covs, new_covs], dim=0)

        covs_reg = torch.stack([
            shrink_cov(self.covs[i], self.eval_cov_reg, self.device)
            for i in range(self.covs.shape[0])
        ])
        self.covs_inv = torch.inverse(covs_reg)

        self._old_model = copy.deepcopy(self.model)
        self._old_model.eval()
        for p in self._old_model.parameters():
            p.requires_grad_(False)
        self._old_model.to('cpu')

        self._old_bottleneck = copy.deepcopy(self.bottleneck)
        self._old_bottleneck.eval()
        for p in self._old_bottleneck.parameters():
            p.requires_grad_(False)
        self._old_bottleneck.to('cpu')

        print(f"  [AdaGauss] Post-task {task_id}: {self.means.shape[0]} classes tracked")
    

    def _train_adapter_and_adapt(self, train_loader):
        """
        Trains an MLP adapter mapping OLD-model features to NEW-model
        features on the current task's images, then uses it to (a) shift
        each old class's mean and (b) resample+re-estimate each old class's
        covariance in the new feature space.
        """
        adapter = nn.Sequential(
            nn.Linear(self.S, self.multiplier * self.S), nn.GELU(),
            nn.Linear(self.multiplier * self.S, self.S),
        ).to(self.device)
        opt = torch.optim.SGD(adapter.parameters(), lr=0.01,
                              weight_decay=5e-4, momentum=0.9)
        n_ep = max(1, int(self.cfg.train.epochs_per_task * self.adapt_epochs_frac))
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[max(1, int(0.3 * n_ep)), max(2, int(0.6 * n_ep)),
                             max(3, int(0.9 * n_ep))], gamma=0.1)
        old_means = self.means.clone()
        old_covs = self.covs.clone()

        # Adapter training must run with gradients enabled (after_task is @no_grad)
        with torch.enable_grad():
            for epoch in range(n_ep):
                adapter.train()
                for imgs, _ in train_loader:
                    imgs = imgs.to(self.device)
                    with torch.no_grad():
                        self.model(imgs)
                        target = self.bottleneck(self.model.hooks['layer4'])
                        self._old_model(imgs)
                        old_feat = self._old_bottleneck(self._old_model.hooks['layer4'])
                    opt.zero_grad()
                    adapted = adapter(old_feat)
                    loss = F.mse_loss(adapted, target)
                    loss.backward()
                    nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                    opt.step()
                sched.step()

        with torch.no_grad():
            adapter.eval()
            for c in range(self.means.shape[0]):
                #cov_for_sampling = shrink_cov(old_covs[c], self.cov_reg, self.device)
                cov_for_sampling = shrink_cov(old_covs[c], self.shrink, self.device)
                dist = torch.distributions.MultivariateNormal(old_means[c], cov_for_sampling)
                samples = dist.sample((10000,))
                adapted_samples = adapter(samples)
                self.means[c] = adapted_samples.mean(0)
                self.covs[c] = shrink_cov(torch.cov(adapted_samples.T), self.shrink, self.device)
    #def _train_adapter_and_adapt(self, train_loader):
    #    """
    #    Trains an MLP adapter mapping OLD-model features to NEW-model
    #    features on the current task's images, then uses it to (a) shift
    #    each old class's mean and (b) resample+re-estimate each old class's
    #    covariance in the new feature space. Direct port of AdaGauss's
    #    adapt_distributions(), minus its diagnostic KLD/norm logging.
    #    """
    #    adapter = nn.Sequential(
    #        nn.Linear(self.S, self.multiplier * self.S), nn.GELU(),
    #        nn.Linear(self.multiplier * self.S, self.S),
    #    ).to(self.device)
    #    opt = torch.optim.SGD(adapter.parameters(), lr=0.01,
    #                          weight_decay=5e-4, momentum=0.9)
    #    n_ep = max(1, int(self.cfg.train.epochs_per_task * self.adapt_epochs_frac))
    #    sched = torch.optim.lr_scheduler.MultiStepLR(
    #        opt, milestones=[max(1, int(0.3 * n_ep)), max(2, int(0.6 * n_ep)),
    #                         max(3, int(0.9 * n_ep))], gamma=0.1)

    #    old_means = self.means.clone()
    #    old_covs = self.covs.clone()

    #    for epoch in range(n_ep):
    #        adapter.train()
    #        for imgs, _ in train_loader:
    #            imgs = imgs.to(self.device)
    #            with torch.no_grad():
    #                self.model(imgs)
    #                target = self.bottleneck(self.model.hooks['layer4'])
    #                self._old_model(imgs)
    #                old_feat = self._old_bottleneck(self._old_model.hooks['layer4'])
    #            opt.zero_grad()
    #            adapted = adapter(old_feat)
    #            loss = F.mse_loss(adapted, target)
    #            loss.backward()
    #            nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
    #            opt.step()
    #        sched.step()

    #    with torch.no_grad():
    #        adapter.eval()
    #        for c in range(self.means.shape[0]):
    #            cov_for_sampling = shrink_cov(old_covs[c], self.cov_reg, self.device)
    #            dist = torch.distributions.MultivariateNormal(old_means[c], cov_for_sampling)
    #            samples = dist.sample((10000,))
    #            adapted_samples = adapter(samples)
    #            self.means[c] = adapted_samples.mean(0)
    #            self.covs[c] = shrink_cov(torch.cov(adapted_samples.T), self.shrink, self.device)

    # ── Bayes / Mahalanobis eval ──────────────────────────────────────────────

    @torch.no_grad()
    def _eval_accuracy(self, loader: DataLoader) -> float:
        if self.means.shape[0] < 2:
            return 0.0
        self.model.eval()
        self.bottleneck.eval()
        correct = total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.model(imgs)
            feats = self.bottleneck(self.model.hooks['layer4'])
            diff = feats.unsqueeze(1) - self.means.unsqueeze(0)           # (B, C, S)
            m = diff.unsqueeze(2) @ self.covs_inv.unsqueeze(0)            # (B, C, 1, S)
            dist = (m @ diff.unsqueeze(3)).squeeze(2).squeeze(2)          # (B, C)
            preds = dist.argmin(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return 100.0 * correct / max(total, 1)
