import os
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer

class LwFTrainer(BaseTrainer):

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self._old_model = None
        self._n_old_classes = 0

    def before_task(self, task_id: int, train_loader: DataLoader):
        if task_id > 0:

            self._old_model = copy.deepcopy(self.model)
            self._old_model.eval()
            for p in self._old_model.parameters():
                p.requires_grad_(False)
            self._old_model.to(self.device)

    def compute_loss(self, logits, targets, task_id):
        ce = F.cross_entropy(logits, targets)

        if self._old_model is None or task_id == 0:
            return ce

        T = self.cfg.method.lwf_temperature
        α = self.cfg.method.lwf_alpha

        with torch.no_grad():
            old_logits = self._old_model(
                self._last_imgs)[:, :self._n_old_classes]

        new_logits_old = logits[:, :self._n_old_classes]

        kd = F.kl_div(
            F.log_softmax(new_logits_old / T, dim=1),
            F.softmax(old_logits        / T, dim=1),
            reduction='batchmean',
        ) * (T ** 2)

        return ce + α * kd

    def _train_step_pre(self, imgs):
        """Store imgs for use inside compute_loss."""
        self._last_imgs = imgs

    def after_task(self, task_id: int, train_loader: DataLoader):
        self._n_old_classes = self.seen_classes
        print(f"  [LwF] Snapshot saved after task {task_id} "
              f"({self._n_old_classes} old classes)")

        if getattr(self.cfg, 'save_snapshots', False):
            method_tag = getattr(self.cfg, 'method_tag', getattr(self.cfg.method, 'name', 'unknown'))
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
            }, snap_path)
            self._snapshot_paths.append(snap_path)
            print(f"  [BaseTrainer] Snapshot saved → {snap_path}")

    def train_task(self, task_id, train_loader, test_loader, classes_this_task):
        self.task_id = task_id
        self.seen_classes += classes_this_task
        self.model.expand_classifier(self.seen_classes)
        self.model.to(self.device)

        self.before_task(task_id, train_loader)
        optimizer = self._build_optimizer()

        scheduler = self._build_scheduler(optimizer, self.cfg.train.epochs_per_task)

        import time
        from tqdm import tqdm
        import torch.nn as nn

        t0 = time.time()
        pbar = tqdm(range(self.cfg.train.epochs_per_task),
                    desc=f"Task {task_id} [LwF]", leave=True)

        for epoch in pbar:
            self.model.train()
            epoch_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self._train_step_pre(imgs)
                logits = self.model(imgs)
                loss   = self.compute_loss(logits, labels, task_id)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                epoch_loss += loss.item()

            if scheduler:
                scheduler.step()

            if (epoch + 1) % self.cfg.log_every == 0:
                acc = self._eval_accuracy(test_loader)
                pbar.set_postfix(loss=f"{epoch_loss / len(train_loader):.3f}",
                                 acc=f"{acc:.2f}")

        self._last_classes_this_task = classes_this_task
        self.after_task(task_id, train_loader)
        acc = self._eval_accuracy(test_loader)
        metrics = {'task_id': task_id, 'test_acc': acc,
                   'time_s': time.time() - t0}
        self._results.append(metrics)
        print(f"  → Task {task_id} | acc: {acc:.2f}%")
        return metrics
