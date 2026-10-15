import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer

class EWCTrainer(BaseTrainer):

    def __init__(self, model, cfg):
        super().__init__(model, cfg)

        self._fisher: dict[str, torch.Tensor] = {}
        self._optima: dict[str, torch.Tensor] = {}

    def before_task(self, task_id: int, train_loader: DataLoader):

        pass

    def compute_loss(self, logits, targets, task_id):
        ce = F.cross_entropy(logits, targets)
        if not self._fisher:
            return ce

        reg = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if n in self._fisher and p.requires_grad:
                stored = self._optima[n].to(p.device)
                fisher = self._fisher[n].to(p.device)

                if p.shape != stored.shape:
                    continue
                diff = p - stored
                reg  = reg + (fisher * diff.pow(2)).sum()

        return ce + (self.cfg.method.ewc_lambda / 2) * reg

    def after_task(self, task_id: int, train_loader: DataLoader):
        self.model.eval()
        new_fisher = {n: torch.zeros_like(p)
                      for n, p in self.model.named_parameters()
                      if p.requires_grad and 'fc' not in n}

        n_samples = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            self.model.zero_grad()
            logits = self.model(imgs)

            probs = F.softmax(logits.detach(), dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(1)
            loss = F.cross_entropy(logits, sampled)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None and n in new_fisher:
                    new_fisher[n] += p.grad.detach().pow(2) * imgs.size(0)

            n_samples += imgs.size(0)
            if n_samples >= 5000:
                break

        for n in new_fisher:
            new_fisher[n] /= n_samples

        gamma = self.cfg.method.ewc_gamma
        if self.cfg.method.ewc_online and self._fisher:
            for n in new_fisher:
                self._fisher[n] = gamma * self._fisher[n].to(self.device) + new_fisher[n]
        else:
            for n in new_fisher:
                if n in self._fisher:
                    self._fisher[n] = self._fisher[n].to(self.device) + new_fisher[n]
                else:
                    self._fisher[n] = new_fisher[n]

        for n, p in self.model.named_parameters():
            if p.requires_grad and 'fc' not in n:
                self._optima[n] = p.detach().clone().cpu()

        self.model.zero_grad()
        print(f"  [EWC] Fisher updated after task {task_id} "
              f"(n_samples={n_samples})")
