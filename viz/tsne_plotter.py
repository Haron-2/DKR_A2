"""t-SNE visualisation of split-0 node embeddings (qualitative evidence only)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


MODEL_ORDER = ("MLP", "GCN", "GAT", "H2GCN")


def load_embedding(path: str | Path) -> dict:
    """Load one exported embedding payload (tensors + metadata)."""
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def compute_tsne(
    embedding: torch.Tensor | np.ndarray,
    *,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Standardise features and embed them in two dimensions with t-SNE.

    Standardisation is applied because the embedding dimensions have very different
    scales across models. ``init="pca"`` gives a deterministic initialisation.
    """
    features = embedding.detach().cpu().numpy() if isinstance(embedding, torch.Tensor) else np.asarray(embedding)
    if features.shape[0] < 3:
        raise ValueError("t-SNE needs at least three samples")
    perplexity = float(min(perplexity, max(2.0, (features.shape[0] - 1) / 3.0)))
    scaled = StandardScaler().fit_transform(features)
    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    return model.fit_transform(scaled)


def plot_tsne(
    embeddings_dir: str | Path,
    output_path: str | Path,
    *,
    dataset: str = "chameleon",
    mode: str = "raw",
    split: int = 0,
    models: tuple[str, ...] = MODEL_ORDER,
    perplexity: float = 30.0,
    random_state: int = 42,
    write_coordinates: bool = True,
) -> Path | None:
    """Plot one t-SNE panel per model for a single dataset/condition/split."""
    embeddings_dir = Path(embeddings_dir)
    payloads = []
    for model in models:
        path = (
            embeddings_dir
            / f"embeddings_{model.lower()}_{dataset}_{mode}_split{split}.pt"
        )
        if path.is_file():
            payloads.append((model, load_embedding(path)))
    if not payloads:
        return None

    labels = payloads[0][1]["y"].numpy()
    classes = np.unique(labels)
    cmap = plt.get_cmap("tab10")
    class_colors = {label: cmap(index % 10) for index, label in enumerate(classes)}

    figure, axes = plt.subplots(
        1, len(payloads), figsize=(3.6 * len(payloads), 3.8), squeeze=False
    )
    rows = []
    for axis, (model, payload) in zip(axes[0], payloads):
        coordinates = compute_tsne(
            payload["embedding"], perplexity=perplexity, random_state=random_state
        )
        for label in classes:
            mask = labels == label
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=5,
                color=class_colors[label],
                label=f"class {int(label)}",
                alpha=0.75,
                linewidths=0,
            )
        axis.set_title(
            f"{model} (dim={payload['embedding_dim']}, acc={payload['accuracy']:.3f})",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        if write_coordinates:
            for node, (x_value, y_value) in enumerate(coordinates):
                rows.append(
                    {
                        "model": model,
                        "node": node,
                        "x": float(x_value),
                        "y": float(y_value),
                        "label": int(labels[node]),
                    }
                )
    handles, names = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, names, loc="lower center", ncol=len(classes), fontsize=8)
    figure.suptitle(
        f"t-SNE of split {split} embeddings — {dataset.capitalize()} {mode} "
        "(per-model t-SNE; axes are not comparable across panels)",
        fontsize=10,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    if write_coordinates and rows:
        csv_path = output_path.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", "node", "x", "y", "label"])
            writer.writeheader()
            writer.writerows(rows)
    return output_path
