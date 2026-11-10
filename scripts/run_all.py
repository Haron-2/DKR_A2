"""One-command orchestration of the complete A2 pipeline.

Runs, in order: data preparation, graph statistics, duplicate/leakage audit, the
240 primary runs, the H2GCN ablations, statistical analysis and all figures. Every
step is a separate process, so a failed or interrupted step can be resumed by
re-running the same command (completed runs are skipped by the runners).

Examples
--------
    python -m scripts.run_all --config configs/full.json
    python -m scripts.run_all --smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, arguments: list[str]) -> int:
    print(f"\n=== {name} ===\n$ {' '.join(arguments)}", flush=True)
    completed = subprocess.run(
        [sys.executable, *arguments], cwd=str(PROJECT_ROOT), check=False
    )
    if completed.returncode != 0:
        print(f"Step '{name}' exited with code {completed.returncode}", flush=True)
    return completed.returncode


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_artifacts(
    results_dir: Path,
    analysis_dir: Path,
    figures_dir: Path,
    *,
    min_figures: int = 7,
) -> bool:
    """Check that the expected artifacts exist and that the runs are complete."""
    problems = []
    for name in ("results.json", "results.csv", "run_status.json"):
        if not (results_dir / name).is_file():
            problems.append(f"missing {results_dir / name}")
    primary_status = read_json(results_dir / "run_status.json")
    if primary_status and not primary_status.get("complete", False):
        problems.append(
            "primary experiment incomplete: "
            f"{primary_status.get('completed_runs')}/{primary_status.get('expected_runs')}"
        )
    ablation_status = read_json(results_dir / "ablation_status.json")
    if ablation_status and not ablation_status.get("complete", False):
        problems.append(
            "ablations incomplete: "
            f"{ablation_status.get('completed_runs')}/{ablation_status.get('expected_runs')}"
        )
    if not analysis_dir.is_dir() or not (analysis_dir / "summary.json").is_file():
        problems.append("missing analysis summary")
    figures = sorted(figures_dir.glob("*.png")) if figures_dir.is_dir() else []
    if len(figures) < min_figures:
        problems.append(
            f"only {len(figures)} figures generated (expected >= {min_figures})"
        )

    if problems:
        print("\nArtifact validation failed:", flush=True)
        for problem in problems:
            print(f"  - {problem}", flush=True)
        return False
    print("\nArtifact validation passed:", flush=True)
    if primary_status:
        print(
            f"  primary runs: {primary_status['completed_runs']}/"
            f"{primary_status['expected_runs']} completed, "
            f"{primary_status['failed_runs']} failed",
            flush=True,
        )
    if ablation_status:
        print(
            f"  ablation runs: {ablation_status['completed_runs']}/"
            f"{ablation_status['expected_runs']} completed, "
            f"{ablation_status['failed_runs']} failed",
            flush=True,
        )
    print(f"  figures: {len(figures)}", flush=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "full.json"
    )
    parser.add_argument("--smoke", action="store_true", help="Run a tiny end-to-end check.")
    parser.add_argument("--skip-data", action="store_true", help="Skip dataset preparation.")
    parser.add_argument("--skip-primary", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Limit pending primary runs.")
    args = parser.parse_args(argv)

    results_dir = PROJECT_ROOT / ("results/smoke" if args.smoke else "results")
    analysis_dir = results_dir / "analysis"
    figures_dir = PROJECT_ROOT / "reports" / (
        "figures_smoke" if args.smoke else "figures"
    )

    scope: list[str] = []
    if args.smoke:
        scope = [
            "--datasets", "chameleon", "--modes", "raw", "--splits", "0",
            "--max-epochs", "15",
        ]

    if not args.skip_data:
        code = run_step("Prepare datasets", ["-m", "scripts.prepare_data"])
        if code != 0:
            return code

    code = run_step(
        "Graph statistics and homophily", ["-m", "scripts.analyze_graphs", "--offline"]
    )
    if code != 0:
        return code

    code = run_step(
        "Duplicate and leakage audit",
        [
            "-m", "scripts.audit_duplicates", "--offline",
            "--output", str(analysis_dir / "duplicate_audit.json"),
        ],
    )
    if code != 0:
        return code

    if not args.skip_primary:
        arguments = [
            "-m", "scripts.run_experiments", "--config", str(args.config),
            "--output-dir", str(results_dir), "--threads", str(args.threads),
        ] + scope
        if args.limit is not None:
            arguments += ["--limit", str(args.limit)]
        code = run_step("Primary experiment (240 runs)", arguments)
        if code != 0:
            return code

    if not args.skip_ablations:
        ablation_scope = (
            [
                "--datasets", "chameleon", "--modes", "raw", "--splits", "0",
                "--max-epochs", "15",
            ]
            if args.smoke
            else []
        )
        code = run_step(
            "H2GCN ablations",
            [
                "-m", "scripts.run_ablations", "--config", str(args.config),
                "--output-dir", str(results_dir), "--threads", str(args.threads),
            ] + ablation_scope,
        )
        if code != 0:
            return code

    if not args.skip_analysis:
        code = run_step(
            "Statistical analysis",
            [
                "-m", "scripts.analyze_results",
                "--results", str(results_dir / "results.json"),
                "--ablations", str(results_dir / "ablation_results.json"),
                "--output-dir", str(analysis_dir),
            ],
        )
        if code != 0:
            return code
        code = run_step(
            "Markdown tables",
            [
                "-m", "scripts.export_tables",
                "--analysis", str(analysis_dir),
                "--output-dir", str(PROJECT_ROOT / "reports" / ("tables_smoke" if args.smoke else "tables")),
            ],
        )
        if code != 0:
            return code
        code = run_step(
            "Figures",
            [
                "-m", "scripts.make_figures",
                "--summary", str(analysis_dir / "summary.json"),
                "--audit", str(analysis_dir / "duplicate_audit.json"),
                "--logs", str(results_dir / "logs"),
                "--embeddings", str(results_dir / "embeddings"),
                "--output-dir", str(figures_dir),
            ],
        )
        if code != 0:
            return code

    return (
        0
        if validate_artifacts(
            results_dir, analysis_dir, figures_dir, min_figures=3 if args.smoke else 7
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

