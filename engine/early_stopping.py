"""Validation-loss early stopping with best-checkpoint restoration."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from torch import nn


logger = logging.getLogger(__name__)


class EarlyStopping:
    """Track the best validation loss and keep the corresponding model weights.

    The monitored quantity is always the validation loss. Test metrics are never
    observed here, so they cannot influence model selection. The best weights are
    kept in memory (CPU copy) *and* written to ``checkpoint_path`` when a path is
    supplied, so a run can be inspected or resumed after the fact.
    """

    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 1e-4,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        if patience < 0:
            raise ValueError("patience must be non-negative")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        if self.checkpoint_path is not None:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_score = math.inf
        self.best_epoch = -1
        self.counter = 0
        self.stop = False
        self._best_state: dict[str, torch.Tensor] | None = None

    def _save(self, model: nn.Module, epoch: int) -> None:
        self._best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        if self.checkpoint_path is not None:
            torch.save(
                {
                    "model_state": self._best_state,
                    "epoch": epoch,
                    "val_loss": self.best_score,
                },
                self.checkpoint_path,
            )

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        """Update the tracker with a validation loss and return ``stop``.

        A non-finite score never becomes the best score but still consumes patience.
        """
        improved = math.isfinite(score) and score < self.best_score - self.min_delta
        if improved:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.counter = 0
            self._save(model, epoch)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

    def restore(self, model: nn.Module) -> bool:
        """Load the best weights into ``model``; return whether anything was restored."""
        if self.checkpoint_path is not None and self.checkpoint_path.is_file():
            payload = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(payload["model_state"])
            self.best_score = float(payload["val_loss"])
            self.best_epoch = int(payload["epoch"])
            return True
        if self._best_state is not None:
            model.load_state_dict(self._best_state)
            return True
        logger.warning("No checkpoint available to restore")
        return False
