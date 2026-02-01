from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Optional, Type

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import json
import os

def to_serializable(obj):
    if is_dataclass(obj):
        return {k: to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if hasattr(obj, "__name__"):  # classes (AdamW, BCEWithLogitsLoss 등)
        return obj.__name__
    return obj


def save_yaml(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def make_run_dir(base_dir="runs", trial=None):
    if trial is None:
        name = "manual"
    else:
        name = f"trial_{trial.number:04d}"
    run_dir = os.path.join(base_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

@dataclass
class Config:
    # --- paths ---
    train_path: str = "./train_data"
    val_path: str = "./val_data"
    test_path: str = "./test_data"
    save_path: str = "./model/best_model.pth"
    submission_path: str = "./submission.csv"

    # --- model ---
    model_name: str = "efficientnet_b4"
    image_size: int = 380
    pretrained: bool = True
    use_gradient_checkpointing: bool = True
    drop_rate: float = 0.4
    drop_path_rate: float = 0.2

    # --- training ---
    batch_size: int = 32
    epochs: int = 20
    seed: int = 42

    # --- device / loader ---
    device_str: str = "cuda"  # "cuda" or "cpu"
    num_workers: int = 12
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 4  # only used when num_workers > 0

    # --- optimizer / scheduler / criterion (classes + params) ---
    optimizer_cls: Type[optim.Optimizer] = optim.AdamW
    optimizer_params: dict[str, Any] = field(default_factory=lambda: {
        "lr": 1e-4,
        "weight_decay": 1e-2,
    })

    scheduler_cls: Type[optim.lr_scheduler._LRScheduler] = optim.lr_scheduler.CosineAnnealingLR
    scheduler_params: dict[str, Any] = field(default_factory=lambda: {
        "T_max": 20,      # will be synced with epochs in __post_init__
        "eta_min": 1e-6,
    })

    criterion_cls: Type[nn.Module] = nn.BCEWithLogitsLoss
    criterion_params: dict[str, Any] = field(default_factory=lambda: {
        # Keep as float here; convert to tensor on correct device when building criterion
        "pos_weight": None,  # e.g., 1.0 or None
    })

    # --- augmentation ---
    p_horizontal_flip: float = 0.5
    p_random_rotate90: float = 0.5
    p_transpose: float = 0.2
    use_default_transform: bool = True

    # --- early stopping ---
    patience: int = 10
    min_delta: float = 0.0
    mode: str = "max"
    verbose: bool = True

    def __post_init__(self):
        # Keep scheduler aligned with epochs
        if self.scheduler_cls == optim.lr_scheduler.CosineAnnealingLR:
            self.scheduler_params["T_max"] = self.epochs

        # Validate prefetch_factor usage
        if self.num_workers == 0:
            # prefetch_factor is ignored / may error depending on version
            self.prefetch_factor = 2  # harmless default, but you should not pass it to DataLoader

    @property
    def device(self) -> torch.device:
        if self.device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
