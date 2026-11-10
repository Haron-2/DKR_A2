"""Aggregation and paired statistical analysis of experiment results."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


METRICS = ("accuracy", "macro_f1")
MODEL_ORDER = ("MLP", "GCN", "GAT", "H2GCN")


def load_records(path: str | Path) -> list[dict]:
    """Load a results JSON document (a list of run records)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "runs" in payload:
        return payload["runs"]
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected results document in {path}")
    return payload


def _values(
    records: list[dict], metric: str, key: tuple[str, str, str]
) -> list[float]:
    return sorted(
        record[metric]
        for record in records
        if record["status"] == "completed"
        and (record["model"], record["dataset"], record["mode"]) == key
    )


def aggregate(records: list[dict], metric: str = "accuracy") -> list[dict]:
    """Mean and sample standard deviation per model/dataset/mode."""
    keys = sorted(
        {(record["model"], record["dataset"], record["mode"]) for record in records}
    )
    rows = []
    for model, dataset, mode in keys:
        values = _values(records, metric, (model, dataset, mode))
        attempted = len(
            [
                record
                for record in records
                if (record["model"], record["dataset"], record["mode"])
                == (model, dataset, mode)
            ]
        )
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "mode": mode,
                "metric": metric,
                "n_completed": len(values),
                "n_attempted": attempted,
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "min": float(np.min(values)) if values else None,
                "max": float(np.max(values)) if values else None,
            }
        )
    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["mode"],
            MODEL_ORDER.index(row["model"])
            if row["model"] in MODEL_ORDER
            else len(MODEL_ORDER),
        )
    )
    return rows


def structural_delta(
    records: list[dict],
    metric: str = "accuracy",
    baseline: str = "MLP",
) -> tuple[list[dict], list[dict]]:
    """Per-split ``metric(model) - metric(baseline)`` and its summary.

    Only runs of the same dataset, condition and split index are paired.
    """
    index = {
        (record["model"], record["dataset"], record["mode"], record["split"]): record
        for record in records
        if record["status"] == "completed"
    }
    per_split = []
    for (model, dataset, mode, split), record in sorted(index.items()):
        reference = index.get((baseline, dataset, mode, split))
        if reference is None or model == baseline:
            continue
        per_split.append(
            {
                "model": model,
                "dataset": dataset,
                "mode": mode,
                "split": split,
                "metric": metric,
                "baseline": baseline,
                "value": record[metric],
                "baseline_value": reference[metric],
                "delta": record[metric] - reference[metric],
            }
        )
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in per_split:
        grouped[(row["model"], row["dataset"], row["mode"])].append(row["delta"])
    summary = [
        {
            "model": model,
            "dataset": dataset,
            "mode": mode,
            "metric": metric,
            "baseline": baseline,
            "n_pairs": len(values),
            "mean_delta": float(np.mean(values)),
            "std_delta": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
        for (model, dataset, mode), values in grouped.items()
    ]
    summary.sort(
        key=lambda row: (
            row["dataset"],
            row["mode"],
            MODEL_ORDER.index(row["model"])
            if row["model"] in MODEL_ORDER
            else len(MODEL_ORDER),
        )
    )
    return per_split, summary


def paired_comparisons(
    records: list[dict],
    metric: str = "accuracy",
    baseline: str = "MLP",
    comparisons: tuple[str, ...] = ("GCN", "GAT", "H2GCN"),
) -> list[dict]:
    """Paired t-test, Wilcoxon test, effect size and 95% CI per dataset/mode.

    Pairs are matched strictly on ``(dataset, mode, split)``. Raw and filtered
    runs never enter the same comparison.
    """
    index = {
        (record["model"], record["dataset"], record["mode"], record["split"]): record
        for record in records
        if record["status"] == "completed"
    }
    conditions = sorted({(record["dataset"], record["mode"]) for record in records})
    rows = []
    for dataset, mode in conditions:
        for comparison in comparisons:
            pairs = []
            for split in range(10):
                left = index.get((baseline, dataset, mode, split))
                right = index.get((comparison, dataset, mode, split))
                if left is not None and right is not None:
                    pairs.append((left[metric], right[metric]))
            if len(pairs) < 3:
                continue
            baseline_values = np.array([pair[0] for pair in pairs], dtype=float)
            comparison_values = np.array([pair[1] for pair in pairs], dtype=float)
            differences = comparison_values - baseline_values
            n_pairs = len(differences)
            mean_diff = float(differences.mean())
            std_diff = float(differences.std(ddof=1))
            t_statistic, t_pvalue = stats.ttest_rel(
                comparison_values, baseline_values
            )
            if np.allclose(differences, 0.0):
                wilcoxon_statistic, wilcoxon_pvalue = float("nan"), 1.0
            else:
                wilcoxon_statistic, wilcoxon_pvalue = stats.wilcoxon(
                    differences, alternative="two-sided"
                )
            if std_diff > 0:
                half_width = float(
                    stats.t.ppf(0.975, n_pairs - 1) * std_diff / math.sqrt(n_pairs)
                )
                cohen_dz = mean_diff / std_diff
            else:
                half_width = 0.0
                cohen_dz = float("nan")
            rows.append(
                {
                    "baseline": baseline,
                    "comparison": comparison,
                    "dataset": dataset,
                    "mode": mode,
                    "metric": metric,
                    "n_pairs": n_pairs,
                    "mean_baseline": float(baseline_values.mean()),
                    "mean_comparison": float(comparison_values.mean()),
                    "mean_difference": mean_diff,
                    "std_difference": std_diff,
                    "t_statistic": float(t_statistic),
                    "p_value_ttest": float(t_pvalue),
                    "wilcoxon_statistic": float(wilcoxon_statistic),
                    "p_value_wilcoxon": float(wilcoxon_pvalue),
                    "cohen_dz": float(cohen_dz),
                    "ci95_low": mean_diff - half_width,
                    "ci95_high": mean_diff + half_width,
                }
            )
    adjusted = holm_correction([row["p_value_ttest"] for row in rows])
    for row, p_adjusted in zip(rows, adjusted):
        row["p_value_holm"] = float(p_adjusted)
        row["significant_holm_0.05"] = bool(p_adjusted < 0.05)
    return rows


def raw_vs_filtered_summary(
    records: list[dict], metric: str = "accuracy"
) -> list[dict]:
    """Descriptive raw-vs-filtered comparison (never a paired test).

    Chameleon and Squirrel filtered datasets contain different node populations
    and Actor raw/filtered use different masks, so pairing is invalid. The
    difference is therefore reported descriptively.
    """
    rows = []
    datasets = sorted({record["dataset"] for record in records})
    for dataset in datasets:
        for model in MODEL_ORDER:
            raw = [
                record[metric]
                for record in records
                if record["status"] == "completed"
                and record["model"] == model
                and record["dataset"] == dataset
                and record["mode"] == "raw"
            ]
            filtered = [
                record[metric]
                for record in records
                if record["status"] == "completed"
                and record["model"] == model
                and record["dataset"] == dataset
                and record["mode"] == "filtered"
            ]
            if not raw or not filtered:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "n_raw": len(raw),
                    "n_filtered": len(filtered),
                    "mean_raw": float(np.mean(raw)),
                    "mean_filtered": float(np.mean(filtered)),
                    "difference_filtered_minus_raw": float(
                        np.mean(filtered) - np.mean(raw)
                    ),
                    "test": "not performed (different node populations or split masks)",
                }
            )
    return rows


def holm_correction(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    count = len(pvalues)
    order = sorted(range(count), key=lambda index: pvalues[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        value = (count - rank) * pvalues[index]
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_between(
    left_records: list[dict],
    right_records: list[dict],
    metric: str = "accuracy",
    left_name: str = "left",
    right_name: str = "right",
) -> list[dict]:
    """Paired comparison of two record sets matched on dataset/mode/split.

    Used for the H2GCN ablations, where the full model comes from the primary
    results file and the variants from the ablation results file. Holm correction
    is applied within the returned family of tests.
    """
    left = [
        dict(record, model="__left__")
        for record in left_records
        if record.get("status") == "completed"
    ]
    right = [
        dict(record, model="__right__")
        for record in right_records
        if record.get("status") == "completed"
    ]
    rows = paired_comparisons(
        left + right, metric, baseline="__left__", comparisons=("__right__",)
    )
    for row in rows:
        row["baseline"] = left_name
        row["comparison"] = right_name
    return rows


def run_characteristics(records: list[dict]) -> list[dict]:
    """Mean epochs, training time and parameter counts per model/condition."""
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record.get("status") != "completed":
            continue
        grouped[(record["model"], record["dataset"], record["mode"])].append(record)
    rows = []
    for (model, dataset, mode), items in grouped.items():
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "mode": mode,
                "n_runs": len(items),
                "trainable_parameters": int(items[0]["trainable_parameters"]),
                "mean_epochs_trained": float(
                    np.mean([item["epochs_trained"] for item in items])
                ),
                "mean_best_epoch": float(
                    np.mean([item["best_epoch"] for item in items])
                ),
                "mean_training_time_seconds": float(
                    np.mean([item["training_time_seconds"] for item in items])
                ),
                "mean_evaluation_time_seconds": float(
                    np.mean([item["evaluation_time_seconds"] for item in items])
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["mode"],
            MODEL_ORDER.index(row["model"])
            if row["model"] in MODEL_ORDER
            else len(MODEL_ORDER),
        )
    )
    return rows

    return adjusted
