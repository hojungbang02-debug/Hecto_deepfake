from dataclasses import asdict
import optuna
from src.augmentation import get_train_transform
from train import train_one_epoch
from src.model import DeepFakeModel
from src.dataset import DeepFakeDataset
from src.EarlyStopping import EarlyStopping
from config import Config

import os
import random
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score



def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducibility (recommended for tuning consistency)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def validate_auc(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0

    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)  # (B,1)

        logits: torch.Tensor = model(images)  # (B,1) or (B,)
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)

        loss = criterion(logits, labels)
        running_loss += float(loss.item())

        probs = torch.sigmoid(logits)
        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    val_loss = running_loss / max(1, len(loader))

    y_true = torch.cat(all_labels, dim=0).numpy().ravel()
    y_prob = torch.cat(all_probs, dim=0).numpy().ravel()

    # ROC-AUC requires both classes in y_true
    val_auc = roc_auc_score(y_true, y_prob)

    return val_loss, float(val_auc)


def build_dataloader(ds, cfg: Config, shuffle: bool) -> DataLoader:
    dl_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
    )
    # Only set persistent_workers / prefetch_factor when num_workers > 0
    if cfg.num_workers > 0:
        dl_kwargs["persistent_workers"] = cfg.persistent_workers
        dl_kwargs["prefetch_factor"] = cfg.prefetch_factor

    return DataLoader(ds, **dl_kwargs)


def build_criterion(cfg: Config) -> nn.Module:
    params = dict(getattr(cfg, "criterion_params", {}) or {})
    # Handle pos_weight safely (float -> tensor on device)
    pw = params.get("pos_weight", None)
    if pw is None or pw == 1.0:
        # If you don't actually want imbalance handling, keep it None
        params.pop("pos_weight", None)
        return cfg.criterion_cls(**params)

    params["pos_weight"] = torch.tensor([float(pw)], device=cfg.device, dtype=torch.float32)
    return cfg.criterion_cls(**params)


def build_optimizer(cfg: Config, model: nn.Module) -> optim.Optimizer:
    params = dict(getattr(cfg, "optimizer_params", {}) or {})
    return cfg.optimizer_cls(model.parameters(), **params)


def build_scheduler(cfg: Config, optimizer: optim.Optimizer):
    params = dict(getattr(cfg, "scheduler_params", {}) or {})
    # Keep CosineAnnealingLR in sync with epochs if present
    if cfg.scheduler_cls == optim.lr_scheduler.CosineAnnealingLR:
        params["T_max"] = cfg.epochs
    return cfg.scheduler_cls(optimizer, **params)


def hold_out_train_and_validate(cfg: Config, trial: optuna.Trial=None) -> Tuple[float, float]:
    """
    Train on train_path, validate on val_path (hold-out).
    Returns: (best_val_loss_at_best_auc, best_val_auc)

    Designed to be called inside an Optuna objective.
    """
    seed_everything(cfg.seed)

    

    device = cfg.device
    transform = get_train_transform(cfg.image_size, cfg.p_horizontal_flip, cfg.p_random_rotate90, cfg.p_transpose) if cfg.use_default_transform else None
    early_stopping = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta, mode=cfg.mode, verbose=cfg.verbose)

    # --- Dataset / Loader ---
    train_dataset = DeepFakeDataset(
        root_dir=cfg.train_path,
        mode="train",
        image_size=cfg.image_size,
        transform=transform,
    )

    if not os.path.exists(cfg.val_path):
        raise RuntimeError(
            f"val_path '{cfg.val_path}' does not exist. Provide a real hold-out set."
        )

    val_dataset = DeepFakeDataset(
        root_dir=cfg.val_path,
        mode="val",
        image_size=cfg.image_size,
    )

    train_loader = build_dataloader(train_dataset, cfg, shuffle=True)   # train should be shuffled
    val_loader = build_dataloader(val_dataset, cfg, shuffle=False)

    # --- Model ---
    model = DeepFakeModel(model_name=cfg.model_name, pretrained=True, drop_rate=cfg.drop_rate, drop_path_rate=cfg.drop_path_rate)
    # if getattr(cfg, "use_gradient_checkpointing", False) and hasattr(model, "set_gradient_checkpointing"):
    #     model.set_gradient_checkpointing(True)
    model = model.to(device)

    # --- Loss / Optim / Sched / AMP ---
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=(device.type == "cuda")
    )
    # --- Train loop ---
    best_auc = -1.0
    best_loss = float("inf")


    for epoch in range(cfg.epochs):
        # Train one epoch (uses your existing function)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)

        # Validate with AUC
        val_loss, val_auc = validate_auc(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        # Track best by AUC
        if val_auc > best_auc:
            best_auc = val_auc
            best_loss = val_loss

            # Save best model
            os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
            torch.save(model.state_dict(), cfg.save_path)
        
        if trial is not None:
            trial.report(val_auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if early_stopping.step(val_auc):
            print(f"[STOP] Early stopping at epoch {epoch}")
            break

        print(
            f"[Epoch {epoch+1:02d}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_auc={val_auc:.6f} "
            f"best_auc={best_auc:.6f}"
        )

    return float(best_loss), float(best_auc)

if __name__ == "__main__":
    cfg = Config()
    cfg.batch_size = 50
    best_loss, best_auc = hold_out_train_and_validate(cfg)
    print(f"Best Loss: {best_loss:.4f} | Best AUC: {best_auc:.4f}")