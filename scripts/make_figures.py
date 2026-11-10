"""Generate all report figures from the stored analysis artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from viz.plots import (
    plot_ablation_comparison,
    plot_duplicate_leakage,
    plot_metric_comparison,
    plot_raw_vs_filtered,
    plot_statistical_comparisons,
    plot_structural_delta,
    plot_training_convergence,
)
from viz.tsne_plotter import plot_tsne, load_embedding
from engine.runner import atomic_write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=PROJECT_ROOT / "results" / "analysis" / "summary.json"
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=PROJECT_ROOT / "results" / "analysis" / "duplicate_audit.json",
    )
    parser.add_argument("--logs", type=Path, default=PROJECT_ROOT / "results" / "logs")
    parser.add_argument(
        "--embeddings", type=Path, default=PROJECT_ROOT / "results" / "embeddings"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "figures"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.summary.is_file():
        parser.error(f"Missing analysis summary: {args.summary}")
    summary = _load(args.summary)
    output_dir = args.output_dir
    created = []

    for metric in ("accuracy", "macro_f1"):
        block = summary["metrics"][metric]
        created.append(
            plot_metric_comparison(
                block["aggregates"],
                metric,
                output_dir / f"{metric}_comparison.png",
                title=f"{metric.replace('_', '-').title()} by model and data condition",
            )
        )
        created.append(
            plot_structural_delta(
                block["structural_delta"], metric, output_dir / f"structural_delta_{metric}.png"
            )
        )
        created.append(
            plot_raw_vs_filtered(
                block["raw_vs_filtered"], metric, output_dir / f"raw_vs_filtered_{metric}.png"
            )
        )
        created.append(
            plot_statistical_comparisons(
                block["paired_comparisons"],
                metric,
                output_dir / f"statistical_comparisons_{metric}.png",
            )
        )

    for dataset, mode in (("chameleon", "raw"), ("chameleon", "filtered")):
        created.append(
            plot_training_convergence(
                args.logs,
                output_dir / f"training_convergence_{dataset}_{mode}.png",
                dataset=dataset,
                mode=mode,
                split=0,
            )
        )

    if args.audit.is_file():
        audit = _load(args.audit)
        created.append(
            plot_duplicate_leakage(
                audit["conditions"], output_dir / "duplicate_leakage.png"
            )
        )

    ablations = summary.get("ablations", {})
    if ablations.get("variants"):
        full_rows = [
            row
            for row in summary["metrics"]["accuracy"]["aggregates"]
            if row["model"] == "H2GCN" and row["dataset"] in ("chameleon", "squirrel")
        ]
        created.append(
            plot_ablation_comparison(
                full_rows,
                ablations["aggregates"]["accuracy"],
                "accuracy",
                output_dir / "ablation_comparison.png",
            )
        )

    for dataset in ("chameleon", "squirrel"):
        for mode in ("raw", "filtered"):
            path = plot_tsne(
                args.embeddings,
                output_dir / f"tsne_{dataset}_{mode}.png",
                dataset=dataset,
                mode=mode,
                split=0,
            )
            if path is not None:
                created.append(path)

    index = []
    if args.embeddings.is_dir():
        for path in sorted(args.embeddings.glob("embeddings_*.pt")):
            try:
                payload = load_embedding(path)
            except Exception:  # noqa: BLE001 - a corrupt export must not stop the run
                logging.warning("Could not read embedding file %s", path)
                continue
            index.append(
                {
                    "file": path.name,
                    "model": payload["model"],
                    "dataset": payload["dataset"],
                    "mode": payload["mode"],
                    "split": int(payload["split"]),
                    "num_nodes": int(payload["embedding"].shape[0]),
                    "embedding_dim": int(payload["embedding_dim"]),
                    "accuracy": payload["accuracy"],
                    "macro_f1": payload["macro_f1"],
                }
            )
        if index:
            atomic_write_json(args.summary.parent / "embeddings_index.json", index)
            print(f"embeddings index: {len(index)} files", flush=True)

    for path in created:
        if path is not None:
            print(f"figure: {path}", flush=True)
    created = [path for path in created if path is not None]
    if not created:
        logging.error("No figures were generated; check the input artifacts")
        return 1
    print(f"\n{len(created)} figures written to {output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
