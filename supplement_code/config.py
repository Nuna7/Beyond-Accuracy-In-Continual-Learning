from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DataConfig:
    dataset: str = "cifar100"
    data_root: str = "./path/to/data"
    num_tasks: int = 10

    class_order_seed: int = 42

@dataclass
class BackboneConfig:
    arch: str = "resnet18"

    hook_layers: List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    hook_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    pretrained: bool = False

@dataclass
class SAEConfig:
    expansion_factor: int = 8
    topk: int = 32
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 256
    aux_loss_coeff: float = 1e-2

    topk_by_layer: bool = True

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

    ewc_lambda: float = 5000.0
    ewc_online: bool = True
    ewc_gamma: float = 1.0

    lwf_alpha: float = 1.0
    lwf_temperature: float = 2.0

    fsgp_lambda: List[float] = field(
        default_factory=lambda: [0.5, 0.5, 1.0, 2.0]
    )
    fsgp_use_sae: bool = True
    fsgp_norm_proj: bool = True
    fsgp_dense_lambda: float = 0.0
    fsgp_mask_mode: str = "activity"
    fsgp_topk_per_class: int = 16

    fecam_freeze_after: int = 0
    fecam_cov_shrink: float = 1.0
    fecam_tukey_lambda: float = 0.5

    efc_optimizer: str = "paper"
    efc_lambda_efm: float = 10.0
    efc_eta: float = 0.1
    efc_proto_sigma: float = 0.2
    efc_cov_eps: float = 1e-3

    dsal_buffer_size: int = 4000
    dsal_gamma: float = 1.0
    dsal_compensation_ratio: float = 0.6
    dsal_activation: str = "tanh"
    dsal_activation_main: str = "relu"

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
    log_every: int = 10
    save_snapshots: bool = True

def cifar100_t10() -> ExpConfig:
    cfg = ExpConfig()
    cfg.data.dataset = "cifar100"
    cfg.data.num_tasks = 10
    cfg.train.epochs_per_task = 50
    cfg.train.init_epochs = 200
    cfg.train.lr_scheduler = "cosine"
    return cfg

def cifar100_t5() -> ExpConfig:
    cfg = cifar100_t10()
    cfg.data.num_tasks = 5
    cfg.train.epochs_per_task = 50
    cfg.train.init_epochs = 200
    cfg.train.lr_scheduler = "cosine"
    return cfg

def cifar100_t20() -> ExpConfig:
    cfg = cifar100_t10()
    cfg.data.num_tasks = 20
    cfg.train.epochs_per_task = 50
    cfg.train.init_epochs = 200
    cfg.train.lr_scheduler = "cosine"
    return cfg

def tinyimagenet_t10() -> ExpConfig:
    cfg = ExpConfig()
    cfg.data.dataset = "tinyimagenet"
    cfg.data.num_tasks = 10
    cfg.train.epochs_per_task = 50
    cfg.train.init_epochs = 200
    cfg.train.lr_scheduler = "cosine"
    return cfg

def tinyimagenet_t20() -> ExpConfig:
    cfg = tinyimagenet_t10()
    cfg.data.num_tasks = 20
    cfg.train.epochs_per_task = 50
    cfg.train.init_epochs = 200
    cfg.train.lr_scheduler = "cosine"
    return cfg
