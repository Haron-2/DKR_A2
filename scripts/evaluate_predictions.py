"""Per-class diagnostics and confusion matrices from stored checkpoints.

Inference only: the script rebuilds each run's architecture, loads the best
validation checkpoint written during training and evaluates the test mask of the
corresponding split. No model is trained and no result file is modified.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from data.loader import PROJECT_ROOT, SUPPORTED_DATASETS, get_dataset
from engine.runner import atomic_write_json, load_results, write_csv
from engine.trainer import load_checkpoint, resolve_split_masks
from models.registry import build_model, model_config_with_defaults


DEFAULT_RESULTS = PROJECT_ROOT / "results" / "results.json"


def resolve_checkpoint(path: str | Path) -> Path:
    """Resolve a stored checkpoint path against the repository root."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


@torch.no_grad()
def evaluate_record(record: dict, data, split: int) -> dict:
    """Return the test predictions and per-class metrics of one checkpoint."""
    model = build_model(
        record["model"],
        data.num_features,
        data.num_classes,
        model_config_with_defaults(record["configuration"]["model"]),
    )
    load_checkpoint(model, resolve_checkpoint(record["checkpoint_path"]))
    model.eval()
    logits = model(data.x, data.edge_index)
    _, _, test_mask = resolve_split_masks(data, split)
    predictions = logits.argmax(dim=1)[test_mask].cpu().numpy()
    targets = data.y[test_mask].cpu().numpy()
    labels = np.arange(int(data.num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    predicted_counts = np.bincount(predictions, minlength=int(data.num_classes))
    predicted_share = predicted_counts / max(1, predictions.size)
    return {
        "predictions": predictions,
        "targets": targets,
        "labels": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "predicted_share": predicted_share,
        "accuracy": float((predictions == targets).mean()),
    }


def summarise(rows: list[dict]) -> list[dict]:
    """Aggregate per-class metrics over the ten splits of a condition."""
    grouped: dict[tuple[str, str, str, int], list[dict]] = {}
    for row in rows:
        key = (row["model"], row["dataset"], row["mode"], row["class"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (model, dataset, mode, label), items in sorted(grouped.items()):

        def stats(column: str) -> tuple[float, float]:
            values = np.array([item[column] for item in items], dtype=float)
            deviation = float(values.std(ddof=1)) if values.size > 1 else 0.0
            return float(values.mean()), deviation

        f1_mean, f1_std = stats("f1")
        recall_mean, recall_std = stats("recall")
        precision_mean, precision_std = stats("precision")
        share_mean, share_std = stats("predicted_share")
        support_mean, support_std = stats("support")
        summary.append(
            {
                "model": model,
                "dataset": dataset,
                "mode": mode,
                "class": label,
                "n_runs": len(items),
                "mean_f1": f1_mean,
                "std_f1": f1_std,
                "mean_recall": recall_mean,
                "std_recall": recall_std,
                "mean_precision": precision_mean,
                "std_precision": precision_std,
                "mean_support": support_mean,
                "std_support": support_std,
                "mean_predicted_share": share_mean,
                "std_predicted_share": share_std,
            }
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results" / "analysis"
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.set_num_threads(4)
    records = [
        record
        for record in load_results(args.results)
        if record.get("status") == "completed"
        and record.get("variant", "full") == "full"
        and resolve_checkpoint(record["checkpoint_path"]).is_file()
    ]
    records.sort(key=lambda record: record["run_id"])
    if args.limit is not None:
        records = records[: args.limit]
    print(f"Evaluating {len(records)} stored checkpoints (inference only)", flush=True)

    per_class_rows: list[dict] = []
    matrices: dict[str, dict] = {}
    data_cache: dict[tuple[str, str], object] = {}
    for index, record in enumerate(records, start=1):
        key = (record["dataset"], record["mode"])
        if key not in data_cache:
            data = get_dataset(
                record["dataset"], record["mode"], root=args.root, offline=args.offline
            )
            data.num_classes = int(data.metadata["summary"]["num_classes"])
            data.num_features = int(data.x.shape[1])
            data_cache[key] = data
        data = data_cache[key]
        metrics = evaluate_record(record, data, int(record["split"]))
        for position, label in enumerate(metrics["labels"]):
            per_class_rows.append(
                {
                    "run_id": record["run_id"],
                    "model": record["model"],
                    "dataset": record["dataset"],
                    "mode": record["mode"],
                    "split": int(record["split"]),
                    "class": int(label),
                    "precision": float(metrics["precision"][position]),
                    "recall": float(metrics["recall"][position]),
                    "f1": float(metrics["f1"][position]),
                    "support": int(metrics["support"][position]),
                    "predicted_share": float(metrics["predicted_share"][position]),
                }
            )
        matrices[record["run_id"]] = {
            "labels": [int(label) for label in metrics["labels"]],
            "matrix": confusion_matrix(
                metrics["targets"], metrics["predictions"], labels=metrics["labels"]
            ).tolist(),
            "accuracy": metrics["accuracy"],
        }
        if index % 40 == 0:
            print(f"  {index}/{len(records)} evaluated", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_class_metrics.csv", per_class_rows)
    atomic_write_json(
        args.output_dir / "confusion_matrices.json",
        {
            "description": (
                "Test-mask confusion matrices of the restored best checkpoints, "
                "computed without retraining. Rows are true classes, columns predicted."
            ),
            "runs": matrices,
        },
    )
    write_csv(args.output_dir / "per_class_summary.csv", summarise(per_class_rows))
    atomic_write_json(
        args.output_dir / "predictions_status.json",
        {"runs_evaluated": len(matrices), "source": str(args.results)},
    )
    print(f"Predictions analysis written to {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

