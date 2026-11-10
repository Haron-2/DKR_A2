"""Duplicate-node and train-test leakage audit for all six dataset conditions.

The audit never influences training. It writes a JSON document with per-condition
duplicate statistics and per-split leakage, plus a compact CSV of the split level
leakage values used by the figures and the report.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from analysis.duplicate_audit import DEFAULT_TOLERANCE, audit_dataset
from data.loader import DEFAULT_ROOT, PROJECT_ROOT, SUPPORTED_DATASETS, get_dataset
from engine.runner import atomic_write_json, write_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["all", "chameleon", "squirrel", "actor"], default="all"
    )
    parser.add_argument("--mode", choices=["all", "raw", "filtered"], default="all")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "analysis" / "duplicate_audit.json",
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.set_num_threads(4)
    conditions = [
        (name, mode)
        for name, mode in SUPPORTED_DATASETS
        if args.dataset in ("all", name) and args.mode in ("all", mode)
    ]
    records = []
    split_rows = []
    for name, mode in conditions:
        data = get_dataset(name, mode, root=args.root, offline=args.offline)
        audit = audit_dataset(data, tolerance=args.tolerance)
        record = {"dataset": name, "mode": mode, **audit}
        records.append(record)
        for row in audit["leakage"]:
            split_rows.append({"dataset": name, "mode": mode, **row})
        duplicates = audit["duplicates"]
        print(
            f"{name:10} {mode:8} nodes={duplicates['num_nodes']:5} "
            f"duplicate_groups={duplicates['duplicate_groups']:4} "
            f"duplicate_nodes={duplicates['duplicate_vertices']:5} "
            f"({duplicates['duplicate_percentage']:5.2f}%) "
            f"leakage={audit['leakage_summary']['mean_leakage_percentage']:5.2f}%",
            flush=True,
        )

    atomic_write_json(
        args.output,
        {
            "method": {
                "duplicate_definition": "||x_i - x_j||_2 < tolerance",
                "tolerance": args.tolerance,
                "grouping": "exact unique rows followed by a KD-tree radius query",
                "leakage_definition": (
                    "test node whose duplicate group contains at least one training node"
                ),
                "label_agreement": (
                    "fraction of duplicate nodes in groups whose labels are identical"
                ),
            },
            "conditions": records,
        },
    )
    write_csv(args.output.with_suffix(".csv"), split_rows)
    print(f"Audit: {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
