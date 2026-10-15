import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional

from models.resnet_hook import HookedResNet18
from utils.metrics import accuracy

class BaseTrainer:

    HOOK_LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']

    def __init__(self, model: HookedResNet18, cfg):
        self.model  = model
        self.cfg    = cfg
        self.device = torch.device(cfg.device
                                   if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.task_id                 = -1
        self.seen_classes            = 0
        self._last_classes_this_task = 0
        self._results: List[Dict]    = []
        self._snapshot_paths: List[str] = []

    def before_task(self, task_id: int, train_loader: DataLoader):
        pass

    def compute_loss(self, logits: torch.Tensor,
                     targets: torch.Tensor, task_id: int) -> torch.Tensor:
        return F.cross_entropy(logits, targets)

    def after_task(self, task_id: int, train_loader: DataLoader):
        pass

    def train_task(self, task_id: int,
                   train_loader: DataLoader,
                   test_loader:  DataLoader,
                   classes_this_task: int) -> Dict:

        self.task_id = task_id
        self.seen_classes += classes_this_task

        self.model.expand_classifier(self.seen_classes)
        self.model.to(self.device)

        self.before_task(task_id, train_loader)

        optimizer = self._build_optimizer()
        epochs = (
            self.cfg.train.init_epochs
            if task_id == 0
            else self.cfg.train.epochs_per_task
        )

        scheduler = self._build_scheduler(optimizer, epochs)

        t0 = time.time()

        train_losses = []

        pbar = tqdm(
            range(epochs),
            desc=f"Task {task_id} [{self.__class__.__name__}]",
            leave=True
        )

        for epoch in pbar:
            self.model.train()
            epoch_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits = self.model(imgs)
                loss   = self.compute_loss(logits, labels, task_id)

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
                acc = self._eval_accuracy(test_loader)
                pbar.set_postfix(loss=f"{epoch_loss:.3f}", acc=f"{acc:.2f}")

        self._last_classes_this_task = classes_this_task
        self.after_task(task_id, train_loader)

        if getattr(self.cfg, 'save_snapshots', False):
            import os, torch
            method_tag = getattr(self.cfg, 'method_tag', self.cfg.method.name)
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

        acc = self._eval_accuracy(test_loader)
        metrics = {
            'task_id':    task_id,
            'train_loss': float(sum(train_losses) / len(train_losses)),
            'test_acc':   acc,
            'time_s':     time.time() - t0,
        }
        self._results.append(metrics)
        print(f"  → Task {task_id} | acc: {acc:.2f}% | "
              f"time: {metrics['time_s']:.1f}s")
        return metrics

    @torch.no_grad()
    def _eval_accuracy(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        return 100.0 * correct / total

    @torch.no_grad()
    def collect_activations(
        self, loader: DataLoader, layer: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        all_acts, all_labels = [], []
        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            self.model(imgs)
            all_acts.append(self.model.hooks[layer].cpu())
            all_labels.append(labels)
        return torch.cat(all_acts), torch.cat(all_labels)

    @torch.no_grad()
    def collect_all_activations(
        self, loader: DataLoader
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        self.model.eval()
        buffers = {l: [] for l in self.HOOK_LAYERS}
        all_labels = []
        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            self.model(imgs)
            for l in self.HOOK_LAYERS:
                buffers[l].append(self.model.hooks[l].cpu())
            all_labels.append(labels)
        return (
            {l: torch.cat(v) for l, v in buffers.items()},
            torch.cat(all_labels),
        )

    def _build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.SGD(
            params,
            lr=self.cfg.train.lr,
            momentum=self.cfg.train.momentum,
            weight_decay=self.cfg.train.weight_decay,
        )

    def _build_scheduler(self, optimizer, epochs):
        sched = self.cfg.train.lr_scheduler

        if sched == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs
            )

        elif sched == "step":
            ms = self.cfg.train.lr_step_milestones
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                ms,
                gamma=0.1
            )

        return None

    def summary(self) -> Dict:
        accs = [r['test_acc'] for r in self._results]
        return {
            'per_task_acc': accs,
            'avg_acc':      sum(accs) / len(accs) if accs else 0.0,
        }
