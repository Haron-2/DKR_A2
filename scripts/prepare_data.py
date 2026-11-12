import argparse
import csv
import json
import logging
from pathlib import Path
import tempfile

import torch
import torch_geometric

from data.loader import DEFAULT_ROOT, PROJECT_ROOT, SUPPORTED_DATASETS, get_dataset


def _write_reports(records: list[dict], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / "data_manifest.json"
    previous = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous = {(row["dataset"], row["mode"]): row for row in manifest["datasets"]}
    for record in records:
        previous[(record["dataset"], record["mode"])] = record
    rows = [previous[key] for key in sorted(previous)]
    manifest = {
        "schema_version": 1,
        "datasets": rows,
        "limitations": [
            "Actor filtered uses the Platonov et al. alternative splits only; no official duplicate-removed actor dataset exists.",
            "Raw and filtered splits are not matched experimental pairs except for Actor, where the node population is identical but the masks differ.",
            "node_id is local to a dataset version, not a raw-filtered mapping.",
        ],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=report_dir, delete=False
    ) as target:
        temporary_path = Path(target.name)
        json.dump(manifest, target, ensure_ascii=False, indent=2)
        target.write("\n")
    temporary_path.replace(manifest_path)

    fields = (
        "dataset", "mode", "num_nodes", "num_features", "num_classes",
        "num_edges_directed", "num_edges_undirected", "average_degree",
        "isolated_nodes", "fingerprint",
    )
    with (report_dir / "data_inventory.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "dataset": row["dataset"], "mode": row["mode"],
                "fingerprint": row["fingerprint"],
                **{key: row["summary"][key] for key in fields[2:-1]},
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare five graph dataset conditions.")
    parser.add_argument("--dataset", choices=("all", "chameleon", "squirrel", "actor"), default="all")
    parser.add_argument("--mode", choices=("all", "raw", "filtered"), default="all")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Reprocess existing raw files; do not redownload.")
    args = parser.parse_args()
    conditions = [
        (name, mode) for name, mode in SUPPORTED_DATASETS
        if args.dataset in ("all", name) and args.mode in ("all", mode)
    ]
    if not conditions:
        parser.error(
            f"No supported condition matches --dataset {args.dataset} "
            f"--mode {args.mode}."
        )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.set_num_threads(4)
    records = []
    for name, mode in conditions:
        data = get_dataset(name, mode, root=args.root, offline=args.offline, rebuild=args.rebuild)
        record = dict(data.metadata)
        record["versions"] = {**record["versions"], "torch_geometric": torch_geometric.__version__}
        records.append(record)
        # Preserve successful results if a subsequent source fails.
        _write_reports(records, args.report_dir)
        summary = record["summary"]
        print(
            f"{name:10} {mode:8} N={summary['num_nodes']:5} "
            f"F={summary['num_features']:4} E={summary['num_edges_undirected']:6} "
            f"classes={summary['num_classes']} splits=10",
            flush=True,
        )
    print(f"Reports: {args.report_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
