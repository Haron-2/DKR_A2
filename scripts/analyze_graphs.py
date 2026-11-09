import argparse
import csv
import json
import logging
from pathlib import Path
import tempfile

import torch

from analysis.homophily import summarize_dataset
from data.loader import DEFAULT_ROOT, PROJECT_ROOT, SUPPORTED_DATASETS, get_dataset


def _write_summary(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output.parent, delete=False
        ) as target:
            temporary_path = Path(target.name)
            writer = csv.DictWriter(target, fieldnames=list(records[0]))
            writer.writeheader()
            for record in records:
                row = {**record, "class_counts": json.dumps(record["class_counts"])}
                writer.writerow(row)
        temporary_path.replace(output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _format_metric(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate descriptive graph statistics and homophily."
    )
    parser.add_argument("--dataset", choices=("all", "chameleon", "squirrel", "actor"), default="all")
    parser.add_argument("--mode", choices=("all", "raw", "filtered"), default="all")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "reports" / "dataset_summary.csv",
        help="CSV for selected conditions; undefined metrics are empty cells.",
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    conditions = [
        (name, mode) for name, mode in SUPPORTED_DATASETS
        if args.dataset in ("all", name) and args.mode in ("all", mode)
    ]
    if not conditions:
        parser.error("Actor has no official filtered condition. Use --dataset actor --mode raw.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.set_num_threads(4)
    records = []
    for name, mode in conditions:
        data = get_dataset(name, mode, root=args.root, offline=args.offline)
        record = {
            "dataset": name,
            "mode": mode,
            **summarize_dataset(data),
            "graph_policy": data.metadata["graph_policy"],
            "source_revision": data.metadata["source_revision"],
            "fingerprint": data.metadata["fingerprint"],
        }
        records.append(record)
        print(
            f"{name:10} {mode:8} "
            f"edge={_format_metric(record['edge_homophily'])} "
            f"node={_format_metric(record['node_homophily'])} "
            f"adjusted={_format_metric(record['adjusted_homophily'])}",
            flush=True,
        )
    _write_summary(records, args.output)
    print(f"Summary: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
