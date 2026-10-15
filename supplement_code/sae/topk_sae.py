import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import Optional

class TopKSAE(nn.Module):

    def __init__(self, d_in: int, d_sae: int, k: int, aux_k: Optional[int] = None):
        super().__init__()
        self.d_in  = d_in
        self.d_sae = d_sae
        self.k     = k
        self.aux_k = aux_k if aux_k is not None else max(k, d_sae // 32)

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))

        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))

        self.b_pre = nn.Parameter(torch.zeros(d_in))

        nn.init.kaiming_uniform_(self.W_enc, nonlinearity='relu')

        self.W_dec.data = self.W_enc.data.T.clone()
        self._normalise_decoder()

        self.register_buffer('num_steps_since_fired',
                              torch.zeros(d_sae, dtype=torch.long))

    @torch.no_grad()
    def _normalise_decoder(self):
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data /= norms

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_c = x - self.b_pre
        pre = x_c @ self.W_enc
        acts = self._topk(pre, self.k)
        return acts, pre

    @staticmethod
    def _topk(scores: torch.Tensor, k: int) -> torch.Tensor:
        topk_vals, topk_idx = scores.topk(k, dim=-1)
        topk_vals = F.relu(topk_vals)
        out = torch.zeros_like(scores)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    def forward(self, x: torch.Tensor):
        acts, pre = self._encode(x)
        x_hat = acts @ self.W_dec + self.b_pre

        is_dead = (self.num_steps_since_fired > 500)
        n_dead  = is_dead.sum().item()

        if n_dead > 0 and self.training:

            residual = (x - x_hat).detach()

            dead_pre = pre.clone()
            dead_pre[:, ~is_dead] = -1e9
            k_dead = min(self.aux_k, n_dead)
            dead_acts = self._topk(dead_pre, k_dead)
            dead_rec  = dead_acts @ self.W_dec
            aux_loss  = (residual - dead_rec).pow(2).mean()
        else:
            aux_loss = x.new_zeros(1).squeeze()

        if self.training:
            with torch.no_grad():
                fired = (acts > 0).any(dim=0)
                self.num_steps_since_fired += 1
                self.num_steps_since_fired[fired] = 0

        return x_hat, acts, aux_loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        acts, _ = self._encode(x)
        return acts

    @torch.no_grad()
    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_pre

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def get_feature_directions(self, latent_indices: torch.Tensor) -> torch.Tensor:
        return self.W_dec[latent_indices]

class SAETrainer:
    def __init__(
        self,
        d_in:             int,
        expansion_factor: int   = 8,
        k:                int   = 32,
        aux_loss_coeff:   float = 1e-2,
        lr:               float = 1e-3,
        epochs:           int   = 20,
        batch_size:       int   = 256,
        device:           str   = "cuda",
        verbose:          bool  = False,
    ):
        self.d_in             = d_in
        self.d_sae            = d_in * expansion_factor
        self.k                = k
        self.aux_loss_coeff   = aux_loss_coeff
        self.lr               = lr
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.device           = device
        self.verbose          = verbose

    def fit(self, activations: torch.Tensor) -> TopKSAE:
        device = self.device
        sae = TopKSAE(self.d_in, self.d_sae, self.k).to(device)
        opt = torch.optim.Adam(sae.parameters(), lr=self.lr)

        ds     = TensorDataset(activations.float())
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                            drop_last=True)

        sae.train()
        pbar = tqdm(range(self.epochs), desc=f"SAE d={self.d_in}",
                    leave=False, disable=not self.verbose)

        for epoch in pbar:
            total_loss = 0.0
            for (x,) in loader:
                x = x.to(device)
                x_hat, acts, aux_loss = sae(x)

                rec_loss  = (x - x_hat).pow(2).mean()
                loss      = rec_loss + self.aux_loss_coeff * aux_loss

                opt.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                opt.step()

                sae._normalise_decoder()
                total_loss += loss.item()

            if self.verbose:
                pbar.set_postfix(loss=f"{total_loss / len(loader):.4f}")

        sae.eval()
        return sae

class SAEBank:
    def __init__(self):
        self._bank = {}

    def add(self, task_id: int, layer_name: str, sae: TopKSAE):
        self._bank[(task_id, layer_name)] = sae

    def get(self, task_id: int, layer_name: str) -> Optional[TopKSAE]:
        return self._bank.get((task_id, layer_name), None)

    def has(self, task_id: int, layer_name: str) -> bool:
        return (task_id, layer_name) in self._bank

    def tasks_trained(self):
        return sorted({k[0] for k in self._bank})
