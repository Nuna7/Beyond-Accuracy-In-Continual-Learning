"""
Main experiment runner.

Usage:
    python train.py --method sae_fsgp --dataset cifar100 --num_tasks 10 --seed 0
    python train.py --method ewc      --dataset cifar100 --num_tasks 10 --seed 0
    python train.py --method fecam    --dataset cifar100 --num_tasks 10 --seed 0
    python train.py --method lwf      --dataset cifar100 --num_tasks 10 --seed 0
    python train.py --method finetune --dataset cifar100 --num_tasks 10 --seed 0
    python train.py --method dense_fsgp --dataset cifar100 --num_tasks 10  # ablation

The accuracy matrix R[i,j] is filled by evaluating on ALL seen tasks after
each new task, which is the standard EFCIL evaluation protocol.
"""
import os
import sys
import csv
import json
import argparse
import random
import numpy as np
import torch

# ── path setup (allow running from anywhere) ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExpConfig, DataConfig, MethodConfig
from data.cifar_cl import ClassIncrementalScenario
from models.resnet_hook import build_resnet18
from methods.base_trainer  import BaseTrainer
from methods.ewc           import EWCTrainer
from methods.lwf           import LwFTrainer
from methods.fecam         import FeCAMTrainer
from methods.sae_fsgp      import SAEFSGPTrainer
from methods.dsal import DSALTrainer
from methods.efc import EFCTrainer
from methods.ada_gauss import AdaGaussTrainer
from utils.metrics         import (accuracy, compute_metrics,
                                   count_stored_params, print_summary)


# ── universal NCM evaluation ─────────────────────────────────────────────────
import torch.nn.functional as _F

@torch.no_grad()
def _build_ncm_prototypes(trainer, task_loaders_cache, T, device):
    """
    Compute L2-normalised class prototypes from the FINAL backbone
    (after all tasks).  Works for every method — uses layer4 features.
    """
    model = trainer.model
    model.eval()
    model.detach_hooks = True   # safe default
    prototypes = {}             # class_id → (512,) normalised CPU tensor

    for t, (train_loader, _) in enumerate(task_loaders_cache):
        # collect final-backbone activations for task t's training data
        buffers, labels_buf = [], []
        for imgs, labels in train_loader:
            model(imgs.to(device))
            buffers.append(model.hooks['layer4'].cpu())
            labels_buf.append(labels)
        acts   = torch.cat(buffers)
        labels = torch.cat(labels_buf)
        for c in labels.unique():
            mu = acts[labels == c].float().mean(0)
            prototypes[int(c)] = _F.normalize(mu, dim=0)
    return prototypes

@torch.no_grad()
def _eval_ncm(model, loader, prototypes, device):
    model.eval()
    classes      = sorted(prototypes.keys())
    proto_matrix = torch.stack([prototypes[c] for c in classes]).to(device)
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        model(imgs)
        feats = _F.normalize(model.hooks['layer4'].float(), dim=1)
        preds = (feats @ proto_matrix.T).argmax(1)
        pred_labels = torch.tensor(
            [classes[p] for p in preds.cpu().tolist()], device=device)
        correct += (pred_labels == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / max(total, 1)


# ── seed ─────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── config builder ────────────────────────────────────────────────────────────

def build_config(args) -> ExpConfig:
    cfg = ExpConfig()

    cfg.data.dataset   = args.dataset
    cfg.data.num_tasks = args.num_tasks
    cfg.data.data_root = args.data_root

    cfg.method.name    = args.method
    cfg.seed           = args.seed
    cfg.device         = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.save_dir       = args.save_dir

    cfg.train.epochs_per_task = args.epochs
    cfg.train.batch_size      = args.batch_size
    cfg.train.num_workers     = args.num_workers

    # Dataset-specific num_classes
    cfg.data.num_classes = 100 if args.dataset == "cifar100" else 200

    # Optional per-run overrides for SAE-FSGP hyperparams
    if args.fsgp_lambda       is not None: cfg.method.fsgp_lambda       = args.fsgp_lambda
    if args.fsgp_dense_lambda is not None: cfg.method.fsgp_dense_lambda = args.fsgp_dense_lambda
    if args.fsgp_mask_mode    is not None: cfg.method.fsgp_mask_mode    = args.fsgp_mask_mode
    if args.fsgp_topk_per_class is not None: cfg.method.fsgp_topk_per_class = args.fsgp_topk_per_class
    if args.fsgp_norm_proj    is not None: cfg.method.fsgp_norm_proj    = args.fsgp_norm_proj
    cfg.save_snapshots = args.save_snapshots

    # Dense-FSGP ablation: same as SAE-FSGP but fsgp_use_sae=False
    original_method = args.method
    if args.method == "dense_fsgp":
        cfg.method.name        = "sae_fsgp"
        cfg.method.fsgp_use_sae = False
    cfg.method.original_method_name = original_method

    for key in ("ada_gauss_S", "ada_gauss_lamb", "ada_gauss_alpha",
               "ada_gauss_beta", "ada_gauss_shrink", "ada_gauss_multiplier",
               "ada_gauss_adapt_frac"):
        val = getattr(args, key)
        if val is not None:
            setattr(cfg.method, key, val)

    cfg.method.efc_optimizer = args.efc_optimizer
    return cfg


# ── trainer factory ───────────────────────────────────────────────────────────

def build_trainer(method: str, model, cfg):
    m = method.lower()
    if m in ("finetune", "ft"):
        return BaseTrainer(model, cfg)
    elif m == "ewc":
        return EWCTrainer(model, cfg)
    elif m == "lwf":
        return LwFTrainer(model, cfg)
    elif m == "fecam":
        return FeCAMTrainer(model, cfg)
    elif m in ("sae_fsgp", "dense_fsgp"):
        return SAEFSGPTrainer(model, cfg)
    elif m == "dsal":
        return DSALTrainer(model, cfg)
    elif m == "efc":
        return EFCTrainer(model, cfg)
    elif m == "ada_gauss":
        return AdaGaussTrainer(model, cfg)
    else:
        raise ValueError(f"Unknown method: {method}. "
                         "Choose from: finetune, ewc, lwf, fecam, sae_fsgp, dense_fsgp", "efc", "dsal")


# ── full CL loop ──────────────────────────────────────────────────────────────

def run_experiment(cfg: ExpConfig) -> dict:
    set_seed(cfg.seed)
    device = cfg.device

    print(f"\n{'='*65}")
    print(f"  Method  : {cfg.method.name}")
    print(f"  Dataset : {cfg.data.dataset}  |  Tasks: {cfg.data.num_tasks}")
    print(f"  Device  : {device}  |  Seed: {cfg.seed}")
    print(f"{'='*65}\n")

    # ── data ────────────────────────────────────────────────────────────────
    scenario = ClassIncrementalScenario(
        dataset     = cfg.data.dataset,
        num_tasks   = cfg.data.num_tasks,
        data_root   = cfg.data.data_root,
        seed        = cfg.data.class_order_seed,
        batch_size  = cfg.train.batch_size,
        num_workers = cfg.train.num_workers,
    )
    cfg.data.num_classes = scenario.num_classes

    # ── model ────────────────────────────────────────────────────────────────
    # Start with 1 output (will be expanded before each task)
    model = build_resnet18(cfg.data.dataset, num_classes=1)

    # ── trainer ──────────────────────────────────────────────────────────────
    trainer = build_trainer(cfg.method.name, model, cfg)

    # ── accuracy matrix  R[after_task, task_id] ──────────────────────────────
    T = cfg.data.num_tasks
    R = np.full((T, T), np.nan)

    task_loaders_cache = []   # cache (train, test) per task for re-evaluation

    for t in range(T):
        train_loader, test_loader = scenario.get_task_loaders(t)
        task_loaders_cache.append((train_loader, test_loader))

        # Train on task t
        trainer.train_task(
            task_id            = t,
            train_loader       = train_loader,
            test_loader        = test_loader,
            classes_this_task  = scenario.classes_per_task,
        )

        # Evaluate on ALL tasks seen so far (fills row t of R)
        print(f"  Evaluating on tasks 0..{t} …")
        for j in range(t + 1):
            _, test_j = task_loaders_cache[j]
            acc_j = trainer._eval_accuracy(test_j)
            R[t, j] = acc_j
            print(f"    task {j}: {acc_j:.2f}%")

    # ── universal NCM final evaluation ──────────────────────────────────────────
    # Compute class prototypes from FINAL backbone, evaluate all methods the
    # same way.  This gives fair numbers regardless of whether the method has
    # its own NCM override (EWC/LwF lack one, SAE-FSGP has one, etc.)
    print("\n  [NCM-eval] Building prototypes from final backbone …")
    try:
        ncm_protos = _build_ncm_prototypes(
            trainer, task_loaders_cache, T, torch.device(cfg.device))
        R_ncm = np.full((T, T), np.nan)
        for j in range(T):
            _, test_j = task_loaders_cache[j]
            R_ncm[T-1, j] = _eval_ncm(
                trainer.model, test_j, ncm_protos, torch.device(cfg.device))
        print(f"  [NCM-eval] Final per-task NCM acc: "
              f"{[f'{R_ncm[T-1,j]:.1f}' for j in range(T)]}")
        per_task_accs_ncm = [float(R_ncm[T-1, j]) for j in range(T)]
        avg_ncm = float(np.nanmean(per_task_accs_ncm))
        print(f"  [NCM-eval] Avg NCM acc = {avg_ncm:.2f}%")
    except Exception as e:
        print(f"  [NCM-eval] Failed: {e}")
        R_ncm = R.copy()
        per_task_accs_ncm = None
        avg_ncm = None

    # ── metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(R)
    storage = count_stored_params(trainer)

    per_task_accs = [float(R[T - 1, j]) for j in range(T)]
    print_summary(cfg.method.name, per_task_accs, metrics, storage)

    # ── interpretability analysis (SAE-FSGP only) ─────────────────────────────
    if cfg.method.name == "sae_fsgp" and cfg.method.fsgp_use_sae:
        from utils.interp import InterpAnalyser
        analyser = InterpAnalyser(trainer, scenario, device)
        analyser.run()
        # interp_dir = os.path.join(cfg.save_dir, "interp",
        #                           f"{cfg.data.dataset}_t{T}_seed{cfg.seed}")
        # analyser.save(interp_dir)

        method_tag = getattr(cfg.method, 'original_method_name', cfg.method.name)
        interp_dir = os.path.join(cfg.save_dir, "interp",
                            f"{method_tag}_{cfg.data.dataset}_t{T}_seed{cfg.seed}")
        analyser.save(interp_dir)
    # ── save results ──────────────────────────────────────────────────────────
    os.makedirs(cfg.save_dir, exist_ok=True)
    result = {
        #'method':            cfg.method.name,
        'method':            getattr(cfg.method, 'original_method_name', cfg.method.name),
        'dataset':           cfg.data.dataset,
        'num_tasks':         T,
        'seed':              cfg.seed,
        'per_task_acc':      per_task_accs,
        'per_task_acc_ncm':  per_task_accs_ncm,
        'avg_acc_ncm':       avg_ncm,
        'R':                 R.tolist(),
        'R_ncm':             R_ncm[T-1].tolist() if R_ncm is not None else None,
        **metrics,
        **{f'storage_{k}': v for k, v in storage.items()},
    }

    # CSV row
    csv_path = os.path.join(cfg.save_dir, "results.csv")
    try:
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(result.keys()), extrasaction='ignore')
            if write_header:
                writer.writeheader()
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in result.items()})
    except Exception as e:
        print(f"  [WARN] could not append to results.csv ({e}); your per-run JSON is unaffected.")

    # Full JSON
    method_for_file = getattr(cfg.method, 'original_method_name', cfg.method.name)
    tag = f"{method_for_file}_{cfg.data.dataset}_t{T}_s{cfg.seed}"
    #tag = f"{cfg.method.name}_{cfg.data.dataset}_t{T}_s{cfg.seed}"
    json_path = os.path.join(cfg.save_dir, f"{tag}.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  Results saved → {json_path}")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SAE-FSGP CL experiments")
    p.add_argument("--method",   default="sae_fsgp",
                   choices=["finetune","ewc","lwf","fecam","sae_fsgp","dense_fsgp", "dsal","efc", "ada_gauss"])
    p.add_argument("--dataset",  default="cifar100",
                   choices=["cifar100","tinyimagenet"])
    p.add_argument("--num_tasks", type=int, default=10)
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--epochs",    type=int, default=100)
    p.add_argument("--batch_size",type=int, default=128)
    p.add_argument("--data_root", default="./data_cache")
    p.add_argument("--save_dir",  default="./results")
    p.add_argument("--num_workers",type=int,default=4)
    # SAE-FSGP specific
    p.add_argument("--fsgp_lambda",   type=float, nargs=4,
                   default=None, help="4 per-layer lambdas e.g. 0.5 0.5 1.0 2.0")
    p.add_argument("--fsgp_dense_lambda", type=float, default=None)
    p.add_argument("--fsgp_mask_mode",    default=None, choices=["activity","discriminative"])
    p.add_argument("--fsgp_topk_per_class", type=int, default=None)
    p.add_argument("--fsgp_norm_proj",   action="store_true", default=None)
    p.add_argument("--no_fsgp_norm_proj",action="store_false", dest="fsgp_norm_proj")
    p.add_argument("--save_snapshots",   action="store_true", default=True)
    p.add_argument("--no_save_snapshots",action="store_false", dest="save_snapshots")
    # Ada specific
    p.add_argument("--ada_gauss_S",          type=int,   default=None)
    p.add_argument("--ada_gauss_lamb",       type=float, default=None)
    p.add_argument("--ada_gauss_alpha",      type=float, default=None)
    p.add_argument("--ada_gauss_beta",       type=float, default=None)
    p.add_argument("--ada_gauss_shrink",     type=float, default=None)
    p.add_argument("--ada_gauss_multiplier", type=int,   default=None)
    p.add_argument("--ada_gauss_adapt_frac", type=float, default=None)
    # EFC-specific
    p.add_argument('--efc_optimizer', type=str, default='shared', choices=['shared','paper'])

    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    cfg    = build_config(args)
    run_experiment(cfg)
