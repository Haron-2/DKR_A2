"""Aggregate the experiment results and run the statistical analysis.

Reads ``results/results.json`` (and ``results/ablation_results.json`` when present)
and writes every table used by the report into ``results/analysis/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from analysis.stats import (
    METRICS,
    aggregate,
    load_records,
    paired_between,
    paired_comparisons,
    raw_vs_filtered_summary,
    run_characteristics,
    structural_delta,
)
from engine.runner import atomic_write_json, write_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def analyze(records: list[dict], ablations: list[dict], output_dir: Path) -> dict:
    """Compute all tables and return the combined summary document."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"metrics": {}, "ablations": {}}
    for metric in METRICS:
        aggregates = aggregate(records, metric)
        delta_per_split, delta_summary = structural_delta(records, metric)
        comparisons = paired_comparisons(records, metric)
        raw_filtered = raw_vs_filtered_summary(records, metric)
        summary["metrics"][metric] = {
            "aggregates": aggregates,
            "structural_delta_per_split": delta_per_split,
            "structural_delta": delta_summary,
            "paired_comparisons": comparisons,
            "raw_vs_filtered": raw_filtered,
        }
        write_csv(output_dir / f"aggregate_{metric}.csv", aggregates)
        write_csv(output_dir / f"structural_delta_{metric}.csv", delta_summary)
        write_csv(output_dir / f"structural_delta_per_split_{metric}.csv", delta_per_split)
        write_csv(output_dir / f"paired_tests_{metric}.csv", comparisons)
        write_csv(output_dir / f"raw_vs_filtered_{metric}.csv", raw_filtered)

    characteristics = run_characteristics(records)
    summary["run_characteristics"] = characteristics
    write_csv(output_dir / "run_characteristics.csv", characteristics)

    if ablations:
        full_h2gcn = [
            record
            for record in records
            if record["model"] == "H2GCN" and record.get("variant", "full") == "full"
        ]
        variants = sorted({record["variant"] for record in ablations})
        summary["ablations"] = {
            "variants": variants,
            "aggregates": {},
            "paired": {},
        }
        for metric in METRICS:
            summary["ablations"]["aggregates"][metric] = aggregate(ablations, metric)
            write_csv(
                output_dir / f"ablation_aggregate_{metric}.csv",
                summary["ablations"]["aggregates"][metric],
            )
            rows = []
            for variant in variants:
                variant_records = [
                    record for record in ablations if record["variant"] == variant
                ]
                rows.extend(
                    paired_between(
                        full_h2gcn,
                        variant_records,
                        metric=metric,
                        left_name="H2GCN",
                        right_name=f"H2GCN[{variant}]",
                    )
                )
            summary["ablations"]["paired"][metric] = rows
            write_csv(output_dir / f"ablation_paired_{metric}.csv", rows)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results" / "results.json")
    parser.add_argument(
        "--ablations", type=Path, default=PROJECT_ROOT / "results" / "ablation_results.json"
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "analysis")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.results.is_file():
        parser.error(f"Missing primary results: {args.results}")
    records = load_records(args.results)
    ablations = load_records(args.ablations) if args.ablations.is_file() else []
    summary = analyze(records, ablations, args.output_dir)
    summary["sources"] = {
        "results": str(args.results),
        "ablations": str(args.ablations) if ablations else None,
        "num_primary_runs": len(records),
        "num_completed_primary_runs": sum(
            1 for record in records if record.get("status") == "completed"
        ),
        "num_ablation_runs": len(ablations),
        "num_completed_ablation_runs": sum(
            1 for record in ablations if record.get("status") == "completed"
        ),
    }
    atomic_write_json(args.output_dir / "summary.json", summary)

    for metric in METRICS:
        rows = summary["metrics"][metric]["aggregates"]
        print(f"\n{metric} (mean ± SD, n completed)")
        for row in rows:
            if row["mean"] is None:
                continue
            print(
                f"{row['dataset']:10} {row['mode']:8} {row['model']:6} "
                f"{row['mean']:.4f} ± {row['std'] or 0.0:.4f} "
                f"(n={row['n_completed']})",
                flush=True,
            )
    print(f"\nAnalysis written to {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
