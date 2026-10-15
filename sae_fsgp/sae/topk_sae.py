"""
TopK Sparse Autoencoder (SAE) for extracting sparse feature directions.

Architecture:
    encoder: Linear(d_in → d_sae), no bias, then TopK activation
    decoder: Linear(d_sae → d_in), columns kept unit-normalised

Loss:
    L = ||x - x_hat||²  +  aux_coeff * auxiliary_loss

Auxiliary loss prevents "dead" latents — latents that never activate across
a dataset. We use the approach from Anthropic's JumpReLU SAE paper: for dead
latents, reconstruct using the top-aux-k latents of the *pre-activation*
scores so they receive gradient.

Reference: Rajamanoharan et al. (2024) "Improving Dictionary Learning with
Gated Sparse Autoencoders"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import Optional


class TopKSAE(nn.Module):
    """
    TopK Sparse Autoencoder.

    Parameters
    ----------
    d_in       : input dimension (e.g. 512 for layer4 of ResNet-18)
    d_sae      : SAE latent dimension (d_in * expansion_factor)
    k          : number of active latents per sample
    aux_k      : number of latents used for the auxiliary dead-latent loss
    """

    def __init__(self, d_in: int, d_sae: int, k: int, aux_k: Optional[int] = None):
        super().__init__()
        self.d_in  = d_in
        self.d_sae = d_sae
        self.k     = k
        self.aux_k = aux_k if aux_k is not None else max(k, d_sae // 32)

        # Encoder: no bias (the centre-subtracted input acts as implicit bias)
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        # Decoder: rows are the dictionary atoms (will be kept normalised)
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        # Bias applied to input before encoding (learned data mean offset)
        self.b_pre = nn.Parameter(torch.zeros(d_in))

        nn.init.kaiming_uniform_(self.W_enc, nonlinearity='relu')
        # Initialise decoder as transposed encoder, then normalise
        self.W_dec.data = self.W_enc.data.T.clone()
        self._normalise_decoder()

        # Track dead latents (updated every forward pass during training)
        self.register_buffer('num_steps_since_fired',
                              torch.zeros(d_sae, dtype=torch.long))

    # ── internal ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _normalise_decoder(self):
        """Keep decoder columns unit-normalised."""
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data /= norms

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        acts   : (B, d_sae) sparse activations (TopK applied)
        pre    : (B, d_sae) pre-TopK scores (for auxiliary loss)
        """
        x_c = x - self.b_pre                        # centre
        pre = x_c @ self.W_enc                       # (B, d_sae)
        acts = self._topk(pre, self.k)               # sparse
        return acts, pre

    @staticmethod
    def _topk(scores: torch.Tensor, k: int) -> torch.Tensor:
        """Zero out all but the top-k entries per row (ReLU on top-k only)."""
        topk_vals, topk_idx = scores.topk(k, dim=-1)
        topk_vals = F.relu(topk_vals)
        out = torch.zeros_like(scores)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        x_hat    : (B, d_in)  reconstruction
        acts     : (B, d_sae) sparse activations
        aux_loss : scalar     auxiliary dead-latent loss
        """
        acts, pre = self._encode(x)
        x_hat = acts @ self.W_dec + self.b_pre       # decode

        # ── auxiliary loss (keep dead latents alive) ─────────────────────────
        # Find latents that have not fired in the last N steps
        is_dead = (self.num_steps_since_fired > 500)          # threshold
        n_dead  = is_dead.sum().item()

        if n_dead > 0 and self.training:
            # Residual after reconstruction with live latents
            residual = (x - x_hat).detach()
            # Reconstruct residual using top-aux_k of dead latent pre-scores
            dead_pre = pre.clone()
            dead_pre[:, ~is_dead] = -1e9              # mask live latents
            k_dead = min(self.aux_k, n_dead)
            dead_acts = self._topk(dead_pre, k_dead)
            dead_rec  = dead_acts @ self.W_dec
            aux_loss  = (residual - dead_rec).pow(2).mean()
        else:
            aux_loss = x.new_zeros(1).squeeze()

        # Update firing tracker
        if self.training:
            with torch.no_grad():
                fired = (acts > 0).any(dim=0)  # (d_sae,)
                self.num_steps_since_fired += 1
                self.num_steps_since_fired[fired] = 0

        return x_hat, acts, aux_loss

    # ── encode / decode only ─────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse codes only. No gradient."""
        acts, _ = self._encode(x)
        return acts

    @torch.no_grad()
    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Reconstruct from sparse codes. No gradient."""
        return acts @ self.W_dec + self.b_pre

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode-decode pass. No gradient."""
        return self.decode(self.encode(x))

    def get_feature_directions(self, latent_indices: torch.Tensor) -> torch.Tensor:
        """Return decoder columns for selected latent indices. Shape (|idx|, d_in)."""
        return self.W_dec[latent_indices]


# ── SAE trainer ───────────────────────────────────────────────────────────────

class SAETrainer:
    """
    Trains a TopK SAE on a tensor of activations.

    Usage:
        trainer = SAETrainer(d_in=512, expansion_factor=8, k=32, device='cuda')
        sae = trainer.fit(activations_tensor)   # (N, d_in) float32
    """

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
        """
        Train an SAE on the provided activation matrix.

        Parameters
        ----------
        activations : (N, d_in) tensor of pre-collected activations

        Returns
        -------
        Trained TopKSAE (on self.device, in eval mode)
        """
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
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                opt.step()
                # Keep decoder atoms normalised
                sae._normalise_decoder()
                total_loss += loss.item()

            if self.verbose:
                pbar.set_postfix(loss=f"{total_loss / len(loader):.4f}")

        sae.eval()
        return sae


# ── SAE bank ─────────────────────────────────────────────────────────────────

class SAEBank:
    """
    Stores one SAE per (task, layer) pair.

    sae_bank[(task_id, layer_name)] = trained TopKSAE
    """

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
