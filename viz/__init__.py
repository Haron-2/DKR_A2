"""Visualisations for the A2 report."""

from viz.plots import (
    MODEL_COLORS,
    MODEL_ORDER,
    plot_ablation_comparison,
    plot_duplicate_leakage,
    plot_metric_comparison,
    plot_raw_vs_filtered,
    plot_statistical_comparisons,
    plot_structural_delta,
    plot_training_convergence,
)
from viz.tsne_plotter import compute_tsne, load_embedding, plot_tsne

__all__ = [
    "MODEL_COLORS",
    "MODEL_ORDER",
    "compute_tsne",
    "load_embedding",
    "plot_ablation_comparison",
    "plot_duplicate_leakage",
    "plot_metric_comparison",
    "plot_raw_vs_filtered",
    "plot_statistical_comparisons",
    "plot_structural_delta",
    "plot_training_convergence",
    "plot_tsne",
]
