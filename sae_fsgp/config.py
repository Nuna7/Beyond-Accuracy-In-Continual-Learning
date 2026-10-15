"""
Central configuration for all experiments.
Edit this file to change datasets, splits, and hyperparameters.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    dataset: str = "cifar100"          # "cifar100" | "tinyimagenet"
    data_root: str = "./data_cache"
    num_tasks: int = 10                # T=5 | T=10 | T=20
    # classes_per_task derived from dataset/num_tasks at runtime
    class_order_seed: int = 42         # reproducible class ordering


@dataclass
class BackboneConfig:
    arch: str = "resnet18"
    # Hook layers: indices into [layer1, layer2, layer3, layer4]
    hook_layers: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    # Dims at each hook (after GAP): layer1=64, layer2=128, layer3=256, layer4=512
    hook_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    pretrained: bool = False           # train from scratch (standard in EFCIL)


@dataclass
class SAEConfig:
    expansion_factor: int = 8          # d_sae = expansion_factor * d_in
    topk: int = 32                     # number of active latents per sample
    lr: float = 1e-3
    epochs: int = 20                   # SAE training epochs per task
    batch_size: int = 256
    aux_loss_coeff: float = 1e-2       # auxiliary dead-latent prevention loss
    # topk scales with layer dim to keep sparsity ratio constant
    topk_by_layer: bool = True         # if True, topk = max(8, d_in // 16)


@dataclass
class TrainConfig:
    epochs_per_task: int = 50 
    init_epochs: int = 200 
    batch_size: int = 128
    lr: float = 0.1
    lr_scheduler: str = "cosine" 
    lr_step_milestones: List[int] = field(default_factory=lambda: [60, 80]) 
    weight_decay: float = 5e-4
    momentum: float = 0.9
    num_workers: int = 4


@dataclass
class MethodConfig:
    name: str = "sae_fsgp"
    # --- EWC ---
    ewc_lambda: float = 5000.0
    ewc_online: bool = True
    ewc_gamma: float = 1.0             # online EWC decay
    # --- LwF ---
    lwf_alpha: float = 1.0
    lwf_temperature: float = 2.0
    # --- SAE-FSGP (proposed) ---
    fsgp_lambda: List[float] = field(
        default_factory=lambda: [0.5, 0.5, 1.0, 2.0]  # per-layer weights
    )
    fsgp_use_sae: bool = True          # False → Dense-FSGP ablation
    fsgp_norm_proj: bool = True        # normalise projections → cosine loss, bounds [0,4]
    fsgp_dense_lambda: float = 0.0     # additional dense layer4 distillation weight
    fsgp_mask_mode: str = "activity"   # "activity" | "discriminative"
    fsgp_topk_per_class: int = 16      # latents per class for discriminative mask
    # --- FeCAM ---
    fecam_freeze_after: int = 0        # freeze backbone after task 0 (first task)
    fecam_cov_shrink: float = 1.0      # Ledoit-Wolf shrinkage
    fecam_tukey_lambda: float = 0.5    # Tukey normalization power
    # --- EFC (Elastic Feature Consolidation, ICLR 2024) ---
    efc_optimizer: str = "paper"
    efc_lambda_efm: float = 10.0       # lambda_EFM, Eq. 9
    efc_eta: float = 0.1               # damping eta, Eq. 9
    efc_proto_sigma: float = 0.2       # sigma for EFM-weighted drift kernel (paper's stated value)
    efc_cov_eps: float = 1e-3          # covariance shrinkage for Gaussian prototype sampling
    # --- DS-AL (Dual-Stream Analytic Learning, AAAI 2024) ---
    dsal_buffer_size: int = 4000       # d_buffer, random projection width -- not verified against paper
    dsal_gamma: float = 1.0            # ridge regularisation -- not verified against paper, tune this
    dsal_compensation_ratio: float = 0.6   # C, Eq. 19 -- paper's stated CIFAR-100 value
    dsal_activation: str = "tanh"      # sigma_C, compensation stream -- paper's stated best default
    dsal_activation_main: str = "relu" # sigma_M, main stream


@dataclass
class ExpConfig:
    data: DataConfig = field(default_factory=DataConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    device: str = "cuda"
    seed: int = 0
    save_dir: str = "./results"
    log_every: int = 10                # log every N epochs
    save_snapshots: bool = True        # save per-task backbone checkpoints for analysis


# ── Named experiment presets ──────────────────────────────────────────────────

def cifar100_t10() -> ExpConfig:
    """Standard CIFAR-100, 10 tasks × 10 classes (most common benchmark)."""
    cfg = ExpConfig()
    cfg.data.dataset = "cifar100"
    cfg.data.num_tasks = 10
    cfg.train.epochs_per_task = 50 
    cfg.train.init_epochs = 200 
    cfg.train.lr_scheduler = "cosine" 
    return cfg


def cifar100_t5() -> ExpConfig:
    """CIFAR-100, 5 tasks × 20 classes (matches FeCAM paper)."""
    cfg = cifar100_t10()
    cfg.data.num_tasks = 5
    cfg.train.epochs_per_task = 50 
    cfg.train.init_epochs = 200 
    cfg.train.lr_scheduler = "cosine"
    return cfg


def cifar100_t20() -> ExpConfig:
    """CIFAR-100, 20 tasks × 5 classes (hardest setting)."""
    cfg = cifar100_t10()
    cfg.data.num_tasks = 20
    cfg.train.epochs_per_task = 50 
    cfg.train.init_epochs = 200 
    cfg.train.lr_scheduler = "cosine"
    return cfg


def tinyimagenet_t10() -> ExpConfig:
    """Tiny-ImageNet, 10 tasks × 20 classes."""
    cfg = ExpConfig()
    cfg.data.dataset = "tinyimagenet"
    cfg.data.num_tasks = 10
    cfg.train.epochs_per_task = 50 
    cfg.train.init_epochs = 200 
    cfg.train.lr_scheduler = "cosine"
    return cfg


def tinyimagenet_t20() -> ExpConfig:
    """Tiny-ImageNet, 20 tasks × 10 classes."""
    cfg = tinyimagenet_t10()
    cfg.data.num_tasks = 20
    cfg.train.epochs_per_task = 50 
    cfg.train.init_epochs = 200 
    cfg.train.lr_scheduler = "cosine"
    return cfg
