import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import Optional

from .compat import (
    AutoEncoderTopK,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
    get_lr_schedule,
)

class OfflineSAETrainer:

    def __init__(
        self,
        d_in:             int,
        expansion:        int   = 8,
        k:                int   = 32,
        auxk_alpha:       float = 1 / 32,
        lr:               Optional[float] = None,
        epochs:           int   = 25,
        batch_size:       int   = 512,
        warmup_frac:      float = 0.05,
        device:           str   = "cuda",
        verbose:          bool  = True,
        seed:             int   = 0,
    ):
        self.d_in       = d_in
        self.d_sae      = d_in * expansion
        self.k          = k
        self.auxk_alpha = auxk_alpha
        self.epochs     = epochs
        self.batch_size = batch_size
        self.warmup_frac = warmup_frac
        self.device     = device
        self.verbose    = verbose
        self.seed       = seed

        if lr is not None:
            self.lr = lr
        else:
            scale = self.d_sae / (2 ** 14)
            self.lr = 2e-4 / math.sqrt(max(scale, 1e-6))

    @staticmethod
    @torch.no_grad()
    def _geometric_median(points: torch.Tensor, max_iter: int = 100) -> torch.Tensor:
        guess = points.mean(0)
        for _ in range(max_iter):
            w = 1.0 / (points - guess).norm(dim=1).clamp(min=1e-8)
            w /= w.sum()
            new = (w[:, None] * points).sum(0)
            if (new - guess).norm() < 1e-5:
                break
            guess = new
        return guess

    def _auxk_loss(
        self,
        residual:     torch.Tensor,
        pre_relu:     torch.Tensor,
        dead_mask:    torch.Tensor,
        sae:          AutoEncoderTopK,
    ) -> torch.Tensor:
        n_dead = int(dead_mask.sum())
        if n_dead == 0:
            return residual.new_zeros(1).squeeze()

        k_aux = min(self.d_in // 2, n_dead)
        masked = torch.where(dead_mask[None], pre_relu, torch.full_like(pre_relu, -1e9))
        aux_vals, aux_idx = masked.topk(k_aux, dim=-1)
        aux_acts = torch.zeros_like(pre_relu).scatter_(-1, aux_idx, aux_vals)
        x_aux    = sae.decoder(aux_acts)

        l2 = (residual - x_aux).pow(2).sum(-1).mean()
        mu = residual.mean(0, keepdim=True).expand_as(residual)
        denom = (residual - mu).pow(2).sum(-1).mean().clamp(min=1e-8)
        return (l2 / denom).nan_to_num(0.0)

    def fit(self, activations: torch.Tensor) -> AutoEncoderTopK:
        torch.manual_seed(self.seed)
        device = self.device

        sae = AutoEncoderTopK(self.d_in, self.d_sae, self.k).to(device)

        acts_cpu = activations.float().cpu()
        eff_bs   = min(self.batch_size, len(acts_cpu))
        ds       = TensorDataset(acts_cpu)
        loader   = DataLoader(ds, batch_size=eff_bs,
                              shuffle=True, drop_last=False, num_workers=0)
        total_steps = self.epochs * len(loader)

        warmup = (max(0, min(int(total_steps * self.warmup_frac), total_steps - 1))
                  if total_steps > 1 else 0)

        opt = torch.optim.Adam(sae.parameters(), lr=self.lr, betas=(0.9, 0.999))
        if total_steps > 1:
            lr_fn = get_lr_schedule(total_steps, warmup_steps=warmup)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_fn)
        else:
            sched = None

        tokens_since_fired = torch.zeros(self.d_sae, dtype=torch.long, device=device)
        dead_threshold     = max(10_000, total_steps * eff_bs // 10)

        init_batch = acts_cpu[:512].to(device)
        sae.b_dec.data = self._geometric_median(init_batch)

        sae.train()
        step = 0
        pbar = tqdm(range(self.epochs),
                    desc=f"  SAE d={self.d_in}→{self.d_sae} k={self.k}",
                    leave=False, disable=not self.verbose)

        for epoch in pbar:
            epoch_loss = 0.0
            for (x,) in loader:
                x = x.to(device)

                f, top_vals, top_idx, pre_relu = sae.encode(
                    x, return_topk=True, use_threshold=False)
                x_hat   = sae.decode(f)
                residual = x - x_hat

                l2_loss = residual.pow(2).sum(-1).mean()
                dead    = tokens_since_fired >= dead_threshold
                aux     = self._auxk_loss(residual.detach(), pre_relu, dead, sae)
                loss    = l2_loss + self.auxk_alpha * aux

                opt.zero_grad()
                loss.backward()
                sae.decoder.weight.grad =                    remove_gradient_parallel_to_decoder_directions(
                        sae.decoder.weight, sae.decoder.weight.grad,
                        self.d_in, self.d_sae)
                nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                opt.step()
                if sched is not None: sched.step()
                sae.decoder.weight.data = set_decoder_norm_to_unit_norm(
                    sae.decoder.weight, self.d_in, self.d_sae)

                fired = torch.zeros(self.d_sae, dtype=torch.bool, device=device)
                fired[top_idx.flatten()] = True
                tokens_since_fired += x.size(0)
                tokens_since_fired[fired] = 0

                epoch_loss += loss.item()
                step += 1

            n_batches = max(len(loader), 1)
            pbar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}",
                             dead=int(tokens_since_fired.eq(0).logical_not().sum()))

        sae.eval()
        return sae

class SAEBank:
    def __init__(self):
        self._saes:       dict = {}
        self._prototypes: dict = {}

    def add_sae(self, task_id: int, layer: str, sae: AutoEncoderTopK):
        self._saes[(task_id, layer)] = sae

    def get_sae(self, task_id: int, layer: str) -> Optional[AutoEncoderTopK]:
        return self._saes.get((task_id, layer))

    def add_prototypes(self, task_id: int, layer: str,
                       dense: torch.Tensor,
                       sparse: torch.Tensor,
                       mask: torch.Tensor):
        self._prototypes[(task_id, layer)] = {
            'dense':  dense.cpu(),
            'sparse': sparse.cpu(),
            'mask':   mask.cpu(),
        }

    def get_prototypes(self, task_id: int, layer: str) -> Optional[dict]:
        return self._prototypes.get((task_id, layer))

    def task_ids(self):
        return sorted({k[0] for k in self._saes})

    def save(self, path: str):
        import os, torch
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({'saes': self._saes, 'prototypes': self._prototypes}, path)

    def load(self, path: str):
        data = torch.load(path, map_location='cpu')
        self._saes       = data['saes']
        self._prototypes = data['prototypes']
