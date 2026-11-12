"""Training engine: early stopping and the single-split training loop."""

from engine.early_stopping import EarlyStopping
from engine.trainer import (
    TrainingConfig,
    compute_metrics,
    load_checkpoint,
    resolve_split_masks,
    set_global_seed,
    train_single_split,
)

__all__ = [
    "EarlyStopping",
    "TrainingConfig",
    "compute_metrics",
    "load_checkpoint",
    "resolve_split_masks",
    "set_global_seed",
    "train_single_split",
]
