"""
Post-hoc SAE analysis pipeline.

For each method (finetune, ewc, lwf, fecam, sae_fsgp):
  1. Load the per-task backbone snapshots saved by base_trainer
  2. Collect activations on task-t training data using snapshot-t
  3. Train a small SAE on those activations
  4. Compute per-class discriminativity scores and prototype codes
  5. Save everything to <results_dir>/posthoc_sae/<method>/

This enables cross-method comparison: what features does EWC actually preserve?
Does its Fisher-weighted parameter space align with SAE-level feature importance?
"""

import os, sys, json, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.resnet_hook import build_resnet18
from sae.offline_trainer import OfflineSAETrainer, SAEBank

HOOK_LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']
HOOK_DIMS   = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}


# def load_snapshot(snap_path: str, dataset: str, device: str) -> torch.nn.Module:
#     ckpt      = torch.load(snap_path, map_location='cpu', weights_only=False)
#     n_classes = ckpt['seen_classes']
#     model     = build_resnet18(dataset, num_classes=n_classes)
#     model.load_state_dict(ckpt['state_dict'])
#     model.eval()
#     model.to(device)
#     return model

def load_snapshot(snap_path: str, dataset: str, device: str) -> torch.nn.Module:
    ckpt = torch.load(snap_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']

    # Prefer the fc size actually stored in the checkpoint.
    # DSAL freezes the backbone after task 0 and never expands fc, while
    # seen_classes keeps growing → trusting seen_classes alone mismatches.
    # AdaGauss / other methods can also save non-expanding or task-local heads.
    if 'fc.weight' in sd:
        n_classes = int(sd['fc.weight'].shape[0])
    else:
        n_classes = int(ckpt.get('seen_classes', 100))

    model = build_resnet18(dataset, num_classes=n_classes)
    incompatible = model.load_state_dict(sd, strict=False)
    # Optional diagnostics (ignore pure-fc missing/unexpected)
    interesting = [
        k for k in (incompatible.missing_keys + incompatible.unexpected_keys)
        if not k.startswith('fc.')
    ]
    if interesting:
        print(f"  [posthoc] non-fc load warnings: {interesting[:8]}")

    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def collect_activations(model, loader, device: str) -> tuple:
    buffers   = {l: [] for l in HOOK_LAYERS}
    all_labels = []
    model.eval()
    for imgs, labels in loader:
        model(imgs.to(device))
        for l in HOOK_LAYERS:
            buffers[l].append(model.hooks[l].cpu())
        all_labels.append(labels)
    return ({l: torch.cat(v) for l, v in buffers.items()},
            torch.cat(all_labels))


def train_posthoc_saes(
    snap_dir:    str,
    result_dir:  str,
    scenario,           # ClassIncrementalScenario
    dataset:     str,
    device:      str    = 'cuda',
    expansion:   int    = 8,
    epochs:      int    = 20,
    verbose:     bool   = True,
) -> str:
    """
    Train SAEs on each task's backbone snapshot and save analysis to result_dir.
    Returns path to the saved bank.
    """
    out_dir = os.path.join(result_dir, 'posthoc_sae')
    os.makedirs(out_dir, exist_ok=True)

    snap_paths = sorted(Path(snap_dir).glob('task*.pt'),
                        key=lambda p: int(p.stem.replace('task', '')))
    if not snap_paths:
        print(f"  [posthoc] No snapshots found in {snap_dir}")
        return out_dir

    bank_data   = {}     # (task_id, layer) → {codes, mask, protos}
    drift_data  = []

    for snap_path in snap_paths:
        task_id = int(snap_path.stem.replace('task', ''))
        if verbose:
            print(f"\n  [posthoc] Task {task_id}: loading snapshot {snap_path.name}")

        model = load_snapshot(str(snap_path), dataset, device)
        act_loader = scenario.get_activation_loader(task_id, max_samples=5000)
        act_dict, labels = collect_activations(model, act_loader, device)
        labels_dev = labels.to(device)

        classes_per_task = scenario.classes_per_task
        task_start       = task_id * classes_per_task

        for layer in HOOK_LAYERS:
            d_in = HOOK_DIMS[layer]
            acts = act_dict[layer].to(device)

            trainer = OfflineSAETrainer(
                d_in=d_in, expansion=expansion,
                epochs=epochs, device=device, verbose=False)
            sae = trainer.fit(acts)

            # Per-class prototype codes
            dense_protos, sparse_codes = [], []
            for c_local in range(classes_per_task):
                c_global = task_start + c_local
                mask_c   = (labels_dev == c_global)
                if mask_c.sum() == 0:
                    dense_protos.append(torch.zeros(d_in))
                    sparse_codes.append(torch.zeros(sae.dict_size))
                    continue
                mu_c = acts[mask_c].float().mean(0)
                z_c  = sae.encode(mu_c.unsqueeze(0)).squeeze(0).cpu()
                dense_protos.append(mu_c.cpu())
                sparse_codes.append(z_c)

            dense_stack  = torch.stack(dense_protos)
            sparse_stack = torch.stack(sparse_codes)

            # Discriminativity scores per latent
            discrim_scores = _compute_discriminativity(sparse_stack)

            bank_data[(task_id, layer)] = {
                'dense_protos':    dense_stack.numpy(),
                'sparse_codes':    sparse_stack.detach().numpy(),
                'discrim_scores':  discrim_scores.detach().numpy(),
                'sae_state':       {k: v.cpu() for k, v in sae.state_dict().items()},
            }
            if verbose:
                k_val = sae.k.item() if hasattr(sae.k, 'item') else sae.k
                print(f"    layer={layer:8s} | d={d_in}→{sae.dict_size} k={k_val}")

        model.to('cpu'); del model

    # Save
    bank_path = os.path.join(out_dir, 'sae_bank.pt')
    torch.save(bank_data, bank_path)
    if verbose:
        print(f"\n  [posthoc] SAE bank saved → {bank_path}")
    return out_dir


def _compute_discriminativity(sparse_codes: torch.Tensor) -> torch.Tensor:
    """
    For each latent j, how much does it distinguish individual classes?
    score(j) = max_c [ relu(mu_c[j] - mean_other_c[j]) ] / sigma_all(j)
    Shape: (d_sae,)
    """
    C, d_sae  = sparse_codes.shape
    sigma_all = sparse_codes.std(0).clamp(1e-6)
    scores    = torch.zeros(d_sae)
    for c in range(C):
        mu_c     = sparse_codes[c]
        if C > 1:
            others   = torch.cat([sparse_codes[:c], sparse_codes[c+1:]])
            mu_other = others.mean(0)
        else:
            mu_other = torch.zeros_like(mu_c)
        scores = torch.maximum(scores, (mu_c - mu_other).clamp(min=0) / sigma_all)
    return scores


def load_posthoc_bank(out_dir: str) -> dict:
    bank_path = os.path.join(out_dir, 'posthoc_sae', 'sae_bank.pt')
    return torch.load(bank_path, map_location='cpu')
