"""Model factory and parameter accounting used by every experiment script."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from models.h2gcn import H2GCN
from models.mlp import MLP
from models.vanilla_gnn import GAT, GCN


MODEL_NAMES = ("MLP", "GCN", "GAT", "H2GCN")

DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "hidden_dim": 64,
    "dropout": 0.5,
    "gat_heads": 8,
    "h2gcn_hidden_dim": 64,
    "h2gcn_normalization": "row",
    "h2gcn_use_one_hop": True,
    "h2gcn_use_two_hop": True,
}

MODEL_DESCRIPTIONS = {
    "MLP": "structure-blind MLP baseline (node features only)",
    "GCN": "two-layer GCN with self-loops and symmetric normalisation",
    "GAT": "two-layer GAT, 8 heads x 8 channels, single output head",
    "H2GCN": "ego-neighbour separation with exact one-hop and two-hop diffusion",
}


def model_config_with_defaults(config: Mapping[str, Any] | None = None) -> dict:
    """Return a complete model configuration, filling in defaults."""
    resolved = dict(DEFAULT_MODEL_CONFIG)
    if config:
        resolved.update(config)
    return resolved


def build_model(
    name: str,
    num_features: int,
    num_classes: int,
    config: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Instantiate one of the supported architectures."""
    config = model_config_with_defaults(config)
    key = name.upper()
    dropout = float(config["dropout"])
    if key == "MLP":
        return MLP(
            num_features,
            num_classes,
            hidden_dim=int(config["hidden_dim"]),
            dropout=dropout,
        )
    if key == "GCN":
        return GCN(
            num_features,
            num_classes,
            hidden_dim=int(config["hidden_dim"]),
            dropout=dropout,
        )
    if key == "GAT":
        return GAT(
            num_features,
            num_classes,
            hidden_dim=int(config["hidden_dim"]),
            heads=int(config["gat_heads"]),
            dropout=dropout,
        )
    if key == "H2GCN":
        return H2GCN(
            num_features,
            num_classes,
            hidden_dim=int(config["h2gcn_hidden_dim"]),
            dropout=dropout,
            use_one_hop=bool(config["h2gcn_use_one_hop"]),
            use_two_hop=bool(config["h2gcn_use_two_hop"]),
            normalization=str(config["h2gcn_normalization"]),
        )
    raise ValueError(f"Unknown model: {name}. Expected one of {MODEL_NAMES}")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
