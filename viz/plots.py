"""Figures for the A2 report.

Error bars always represent one sample standard deviation across the ten splits,
unless the function name states otherwise. Figures are written as PNG files with a
fixed colour/marker scheme shared by all figures.
"""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


MODEL_COLORS = {
    "MLP": "#4C72B0",
    "GCN": "#DD8452",
    "GAT": "#55A868",
    "H2GCN": "#C44E52",
}
MODEL_ORDER = ("MLP", "GCN", "GAT", "H2GCN")
MODE_COLORS = {"raw": "#333333", "filtered": "#7570B3"}
DATASET_ORDER = ("chameleon", "squirrel", "actor")


def _save(figure, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_metric_comparison(
    aggregates: list[dict],
    metric: str,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Grouped bars per dataset/condition with +/- 1 SD error bars."""
    rows = [row for row in aggregates if row["metric"] == metric and row["mean"] is not None]
    conditions = sorted({(row["dataset"], row["mode"]) for row in rows})
    if not conditions:
        return None
    lookup = {
        (row["dataset"], row["mode"], row["model"]): row for row in rows
    }
    figure, axes = plt.subplots(
        1, len(conditions), figsize=(4.2 * len(conditions), 4.2), squeeze=False
    )
    for axis, (dataset, mode) in zip(axes[0], conditions):
        models = [name for name in MODEL_ORDER if (dataset, mode, name) in lookup]
        means = [lookup[(dataset, mode, name)]["mean"] for name in models]
        errors = [
            lookup[(dataset, mode, name)]["std"] or 0.0 for name in models
        ]
        positions = np.arange(len(models))
        axis.bar(
            positions,
            means,
            yerr=errors,
            capsize=4,
            color=[MODEL_COLORS[name] for name in models],
            edgecolor="black",
            linewidth=0.6,
        )
        for position, mean in zip(positions, means):
            axis.text(
                position, mean + 0.01, f"{mean:.3f}", ha="center", va="bottom",
                fontsize=7,
            )
        lower = max(0.0, min(means) - 0.12) if means else 0.0
        axis.set_ylim(lower, 1.0)
        axis.set_xticks(positions)
        axis.set_xticklabels(models)
        axis.set_ylabel(f"{metric.replace('_', '-')} (mean ± SD, 10 splits)")
        axis.set_title(f"{dataset.capitalize()} — {mode}")
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle(title or f"{metric} by model and data condition")
    return _save(figure, output_path)


def plot_raw_vs_filtered(
    rows: list[dict],
    metric: str,
    output_path: str | Path,
) -> Path:
    """Dumbbell plot of raw versus filtered performance per model and dataset."""
    rows = [row for row in rows if row["metric"] == metric]
    datasets = [dataset for dataset in DATASET_ORDER if any(row["dataset"] == dataset for row in rows)]
    if not datasets:
        return None
    figure, axes = plt.subplots(
        1, len(datasets), figsize=(3.8 * len(datasets), 4.0), squeeze=False
    )
    for axis, dataset in zip(axes[0], datasets):
        subset = [row for row in rows if row["dataset"] == dataset]
        positions = np.arange(len(subset))
        for position, row in zip(positions, subset):
            axis.plot(
                [row["mean_raw"], row["mean_filtered"]],
                [position, position],
                color=MODEL_COLORS.get(row["model"], "gray"),
                linewidth=2.0,
                zorder=1,
            )
        axis.scatter(
            [row["mean_raw"] for row in subset],
            positions,
            marker="o",
            label="raw",
            color="#333333",
            zorder=2,
        )
        axis.scatter(
            [row["mean_filtered"] for row in subset],
            positions,
            marker="s",
            label="filtered",
            color="#7570B3",
            zorder=2,
        )
        axis.set_yticks(positions)
        axis.set_yticklabels([row["model"] for row in subset])
        axis.set_xlabel(f"{metric.replace('_', '-')}")
        axis.set_title(dataset.capitalize())
        axis.grid(axis="x", alpha=0.3)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle("Raw versus filtered performance (descriptive, not paired)")
    return _save(figure, output_path)



def plot_structural_delta(
    summary: list[dict],
    metric: str,
    output_path: str | Path,
) -> Path:
    """Structural Delta (model - MLP) per condition with SD error bars."""
    rows = [row for row in summary if row["metric"] == metric]
    conditions = sorted({(row["dataset"], row["mode"]) for row in rows})
    if not conditions:
        return None
    lookup = {(row["dataset"], row["mode"], row["model"]): row for row in rows}
    figure, axes = plt.subplots(
        1, len(conditions), figsize=(4.0 * len(conditions), 4.0), squeeze=False
    )
    for axis, (dataset, mode) in zip(axes[0], conditions):
        models = [
            name for name in MODEL_ORDER
            if name != "MLP" and (dataset, mode, name) in lookup
        ]
        positions = np.arange(len(models))
        means = [lookup[(dataset, mode, name)]["mean_delta"] for name in models]
        errors = [lookup[(dataset, mode, name)]["std_delta"] or 0.0 for name in models]
        axis.bar(
            positions,
            means,
            yerr=errors,
            capsize=4,
            color=[MODEL_COLORS[name] for name in models],
            edgecolor="black",
            linewidth=0.6,
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(positions)
        axis.set_xticklabels(models)
        axis.set_ylabel("Structural Delta (accuracy - MLP)")
        axis.set_title(f"{dataset.capitalize()} — {mode}")
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle("Structural Delta: paired per split, error bars = ±1 SD")
    return _save(figure, output_path)


def plot_training_convergence(
    history_dir: str | Path,
    output_path: str | Path,
    *,
    dataset: str = "chameleon",
    mode: str = "raw",
    split: int = 0,
    models: tuple[str, ...] = MODEL_ORDER,
) -> Path:
    """Validation-loss curves of the requested models for one condition/split."""
    history_dir = Path(history_dir)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    plotted = 0
    for model in models:
        path = history_dir / f"{model.lower()}_{dataset}_{mode}_s{split}.csv"
        if not path.is_file():
            continue
        epochs, train_loss, val_loss = [], [], []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                epochs.append(int(row["epoch"]))
                train_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
        axis.plot(
            epochs, val_loss, color=MODEL_COLORS.get(model, "gray"),
            label=f"{model} validation", linewidth=1.6,
        )
        axis.plot(
            epochs, train_loss, color=MODEL_COLORS.get(model, "gray"),
            linestyle=":", linewidth=0.9, alpha=0.7,
        )
        plotted += 1
    axis.set_xlabel("epoch")
    axis.set_ylabel("cross-entropy loss")
    axis.set_title(
        f"Training convergence — {dataset.capitalize()} {mode}, split {split}\n"
        "solid: validation (monitored), dotted: training"
    )
    if plotted == 0:
        plt.close(figure)
        return None
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    return _save(figure, output_path)



def plot_duplicate_leakage(
    audit_rows: list[dict],
    output_path: str | Path,
) -> Path:
    """Duplicate share and train-test leakage per dataset/condition."""
    if not audit_rows:
        return None
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels, duplicate_share = [], []
    for row in audit_rows:
        labels.append(f"{row['dataset']}\n{row['mode']}")
        duplicate_share.append(row["duplicates"]["duplicate_percentage"])
    positions = np.arange(len(labels))
    left.bar(
        positions, duplicate_share, color="#7570B3", edgecolor="black", linewidth=0.6
    )
    left.set_xticks(positions)
    left.set_xticklabels(labels, fontsize=8)
    left.set_ylabel("% of nodes in a duplicate group")
    left.set_title("Duplicate feature vectors")
    left.grid(axis="y", alpha=0.3)

    for index, row in enumerate(audit_rows):
        percentages = [item["leakage_percentage"] for item in row["leakage"]]
        offsets = np.linspace(-0.12, 0.12, len(percentages))
        right.scatter(
            np.full(len(percentages), index) + offsets,
            percentages,
            s=18,
            color=MODE_COLORS.get(row["mode"], "#333333"),
            alpha=0.8,
        )
        right.hlines(
            float(np.mean(percentages)), index - 0.25, index + 0.25,
            color="black", linewidth=1.4,
        )
    right.set_xticks(positions)
    right.set_xticklabels(labels, fontsize=8)
    right.set_ylabel("% of test nodes with a duplicated training node")
    right.set_title("Per-split train-test leakage (dots = splits, bar = mean)")
    right.grid(axis="y", alpha=0.3)
    return _save(figure, output_path)


def plot_statistical_comparisons(
    comparisons: list[dict],
    metric: str,
    output_path: str | Path,
) -> Path:
    """Forest plot of paired mean differences with 95% confidence intervals."""
    rows = [row for row in comparisons if row["metric"] == metric]
    conditions = sorted({(row["dataset"], row["mode"]) for row in rows})
    if not conditions:
        return None
    figure, axes = plt.subplots(
        1, len(conditions), figsize=(4.6 * len(conditions), 4.4), squeeze=False,
        sharey=True,
    )
    for axis, (dataset, mode) in zip(axes[0], conditions):
        subset = [row for row in rows if (row["dataset"], row["mode"]) == (dataset, mode)]
        positions = np.arange(len(subset))
        for position, row in zip(positions, subset):
            axis.errorbar(
                row["mean_difference"],
                position,
                xerr=[
                    [row["mean_difference"] - row["ci95_low"]],
                    [row["ci95_high"] - row["mean_difference"]],
                ],
                fmt="o",
                color=MODEL_COLORS.get(row["comparison"], "black"),
                capsize=4,
            )
            marker = "*" if row["significant_holm_0.05"] else "ns"
            axis.annotate(
                f"{marker} (p_holm={row['p_value_holm']:.3f})",
                (row["mean_difference"], position),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=7,
            )
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_yticks(positions)
        axis.set_yticklabels([row["comparison"] for row in subset], fontsize=8)
        axis.set_xlabel("mean paired difference vs MLP (95% CI)")
        axis.set_title(f"{dataset.capitalize()} — {mode}")
        axis.grid(axis="x", alpha=0.3)
    figure.suptitle(
        f"{metric}: paired per split; * = significant after Holm, ns = not significant"
    )
    return _save(figure, output_path)


ABLATION_COLORS = {
    "H2GCN": MODEL_COLORS["H2GCN"],
    "H2GCN[no_two_hop]": "#E17C7F",
    "H2GCN[no_one_hop]": "#8C6BB1",
}
ABLATION_LABELS = {
    "H2GCN": "full H2GCN",
    "H2GCN[no_two_hop]": "no 2-hop",
    "H2GCN[no_one_hop]": "no 1-hop",
}


def plot_ablation_comparison(
    full_rows: list[dict],
    variant_rows: list[dict],
    metric: str,
    output_path: str | Path,
) -> Path:
    """Compare the full H2GCN with the one-hop and two-hop ablations."""
    rows = [
        row for row in full_rows if row["metric"] == metric and row["mean"] is not None
    ]
    rows += [
        row for row in variant_rows if row["metric"] == metric and row["mean"] is not None
    ]
    conditions = sorted({(row["dataset"], row["mode"]) for row in rows})
    if not conditions:
        return None
    lookup = {(row["dataset"], row["mode"], row["model"]): row for row in rows}
    order = ("H2GCN", "H2GCN[no_two_hop]", "H2GCN[no_one_hop]")
    figure, axes = plt.subplots(
        1, len(conditions), figsize=(3.6 * len(conditions), 3.9), squeeze=False
    )
    for axis, (dataset, mode) in zip(axes[0], conditions):
        names = [name for name in order if (dataset, mode, name) in lookup]
        positions = np.arange(len(names))
        means = [lookup[(dataset, mode, name)]["mean"] for name in names]
        errors = [lookup[(dataset, mode, name)]["std"] or 0.0 for name in names]
        axis.bar(
            positions,
            means,
            yerr=errors,
            capsize=4,
            color=[ABLATION_COLORS[name] for name in names],
            edgecolor="black",
            linewidth=0.6,
        )
        axis.set_xticks(positions)
        axis.set_xticklabels([ABLATION_LABELS[name] for name in names], fontsize=8)
        axis.set_ylabel(f"{metric.replace('_', '-')} (mean ± SD)")
        axis.set_title(f"{dataset.capitalize()} — {mode}")
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle("H2GCN ablations: contribution of the one-hop and two-hop branches")
    return _save(figure, output_path)

