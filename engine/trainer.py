"""Reproducible single-split transductive training engine."""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from engine.early_stopping import EarlyStopping


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters shared by every model (see the final report)."""

    lr: float = 0.01
    weight_decay: float = 5e-4
    dropout: float = 0.5
    patience: int = 50
    min_delta: float = 1e-4
    max_epochs: int = 1000
    seed: int = 42
    device: str = "cpu"
    checkpoint_path: str | None = None
    log_history: bool = True
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["checkpoint_path"] = None  # run-local path, not part of the config
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "TrainingConfig":
        known = {key: value for key, value in payload.items() if key in asdict(cls())}
        return cls(**known)


def set_global_seed(seed: int) -> None:
    """Seed every random number generator used by the pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_split_masks(data, split_idx: int):
    """Return the boolean train/val/test masks of one split."""
    for name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, name)
        if mask.ndim != 2 or mask.shape[1] != 10:
            raise ValueError(f"{name} must have shape [N, 10]")
        if split_idx < 0 or split_idx >= mask.shape[1]:
            raise ValueError(f"split_idx {split_idx} outside 0..9")
    train_mask = data.train_mask[:, split_idx]
    val_mask = data.val_mask[:, split_idx]
    test_mask = data.test_mask[:, split_idx]
    for name, mask in (
        ("train", train_mask),
        ("val", val_mask),
        ("test", test_mask),
    ):
        if not bool(mask.any()):
            raise ValueError(f"{name} mask of split {split_idx} is empty")
    return train_mask, val_mask, test_mask


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    criterion,
) -> tuple[float, torch.Tensor]:
    model.eval()
    logits = model(x, edge_index)
    loss = float(criterion(logits[mask], y[mask]).item())
    return loss, logits


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """Accuracy and macro-F1 from integer tensors."""
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    accuracy = float((y_true_np == y_pred_np).mean()) if y_true_np.size else 0.0
    macro_f1 = float(
        f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)
    )
    return {"accuracy": accuracy, "macro_f1": macro_f1}



def train_single_split(
    model: nn.Module,
    data,
    split_idx: int,
    config: TrainingConfig,
) -> dict:
    """Train one model on one split and evaluate it once on the test mask.

    Only training-mask labels contribute to the gradient. Validation loss drives
    early stopping; the best validation checkpoint is restored before the single
    test evaluation.
    """
    train_mask, val_mask, test_mask = resolve_split_masks(data, split_idx)
    device = torch.device(config.device)
    set_global_seed(config.seed)
    model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    config_dict = config.to_dict()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    checkpoint_path = config.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = str(
            Path("results") / "checkpoints" / f"checkpoint_split{split_idx}.pt"
        )
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        checkpoint_path=checkpoint_path,
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    training_start = time.perf_counter()
    epochs_trained = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        train_loss = float(loss.item())

        val_loss, _ = _evaluate(model, x, edge_index, y, val_mask, criterion)
        if config.log_history:
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
        epochs_trained = epoch
        if early_stopping.step(val_loss, model, epoch):
            break
    training_time = time.perf_counter() - training_start

    early_stopping.restore(model)
    model.to(device)
    evaluation_start = time.perf_counter()
    test_loss, test_logits = _evaluate(model, x, edge_index, y, test_mask, criterion)
    evaluation_time = time.perf_counter() - evaluation_start
    predictions = test_logits.argmax(dim=1)
    metrics = compute_metrics(y[test_mask], predictions[test_mask])

    result = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "best_val_loss": float(early_stopping.best_score),
        "best_epoch": int(early_stopping.best_epoch),
        "epochs_trained": int(epochs_trained),
        "trainable_parameters": int(trainable_parameters),
        "training_time_seconds": float(training_time),
        "evaluation_time_seconds": float(evaluation_time),
        "test_loss": float(test_loss),
        "num_train_nodes": int(train_mask.sum()),
        "num_val_nodes": int(val_mask.sum()),
        "num_test_nodes": int(test_mask.sum()),
        "status": "completed",
        "configuration": config_dict,
        "checkpoint_path": str(checkpoint_path),
    }
    if not math.isfinite(result["best_val_loss"]):
        result["status"] = "failed"
    if config.log_history:
        result["history"] = history
    return result


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> None:
    """Restore a checkpoint written by :class:`EarlyStopping`."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state"])
