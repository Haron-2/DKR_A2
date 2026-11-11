"""H2GCN ablation runner.

Runs the two branch ablations (one-hop only, two-hop only) with exactly the same
optimiser, seeds, splits and model configuration as the full H2GCN of the primary
experiment. The full H2GCN results are read from ``results/results.json`` during
analysis, so no run is duplicated.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from data.loader import DEFAULT_ROOT, PROJECT_ROOT, SUPPORTED_DATASETS, get_dataset
from engine.runner import (
    atomic_write_json,
    execute_run,
    expected_run_ids,
    load_results,
    run_identifier,
    seed_for_split,
    validate_completeness,
    write_csv,
    write_history_csv,
)
from models.registry import build_model, count_parameters, model_config_with_defaults
from scripts.run_experiments import DEFAULT_CONFIG_PATH, default_training_params


DEFAULT_VARIANTS = (
    {"name": "no_two_hop", "h2gcn_use_one_hop": True, "h2gcn_use_two_hop": False},
    {"name": "no_one_hop", "h2gcn_use_one_hop": False, "h2gcn_use_two_hop": True},
)


def load_config(path: Path, args: argparse.Namespace) -> dict:
    """Read the JSON config and merge the ablation section with CLI overrides."""
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    config.setdefault("seed_base", 42)
    config.setdefault("data_root", str(DEFAULT_ROOT))
    config.setdefault("output_dir", str(PROJECT_ROOT / "results"))
    config.setdefault("training", {})
    config.setdefault("model_config", {})
    config.setdefault("splits", list(range(10)))
    config.setdefault("ablations", {})
    ablations = config["ablations"]
    ablations.setdefault("model", "H2GCN")
    ablations.setdefault("datasets", ["chameleon", "squirrel"])
    ablations.setdefault("modes", ["raw", "filtered"])
    ablations.setdefault("variants", [dict(variant) for variant in DEFAULT_VARIANTS])
    if args.datasets:
        ablations["datasets"] = args.datasets
    if args.modes:
        ablations["modes"] = args.modes
    if args.splits:
        config["splits"] = args.splits
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
    if args.root:
        config["data_root"] = str(args.root)
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    if args.seed_base is not None:
        config["seed_base"] = args.seed_base
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--datasets", nargs="+", choices=["chameleon", "squirrel", "actor"]
    )
    parser.add_argument("--modes", nargs="+", choices=["raw", "filtered"])
    parser.add_argument("--splits", nargs="+", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser


def _run_one(
    config,
    dataset,
    mode,
    model_name,
    variant_name,
    model_config,
    data,
    split,
    training_params,
    device,
    checkpoint_dir,
    log_dir,
) -> dict:
    """Execute one ablation run and persist its per-epoch history."""
    seed = seed_for_split(int(config["seed_base"]), split)
    try:
        record, history = execute_run(
            model_name=model_name,
            data=data,
            dataset=dataset,
            mode=mode,
            split=split,
            seed=seed,
            model_config=model_config,
            training_params=training_params,
            device=device,
            checkpoint_dir=checkpoint_dir,
            variant=variant_name,
        )
        write_history_csv(log_dir / f"{record['run_id']}.csv", history)
    except Exception as error:  # noqa: BLE001 - failures are recorded, not hidden
        logging.exception("Ablation run failed")
        record = {
            "run_id": run_identifier(model_name, dataset, mode, split, variant_name),
            "model": model_name,
            "dataset": dataset,
            "mode": mode,
            "split": split,
            "seed": seed,
            "variant": variant_name,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "configuration": {"training": training_params, "model": model_config},
        }
    return record


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config, args)
    torch.set_num_threads(max(1, args.threads))

    ablations = config["ablations"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "ablation_results.json"
    checkpoint_dir = output_dir / "checkpoints" / "ablations"
    log_dir = output_dir / "logs" / "ablations"
    base_model_config = model_config_with_defaults(config.get("model_config"))
    training_params = default_training_params(config)
    device = config["training"].get("device", "cpu")
    model_name = ablations["model"]
    variant_names = [variant["name"] for variant in ablations["variants"]]
    splits = config["splits"]

    expected = expected_run_ids(
        [model_name], ablations["datasets"], ablations["modes"], splits, variant_names
    )
    print(
        f"H2GCN ablations: {len(expected)} planned runs "
        f"({len(variant_names)} variants x {len(ablations['datasets'])} datasets x "
        f"{len(ablations['modes'])} conditions x {len(splits)} splits)",
        flush=True,
    )

    records = load_results(results_path)
    index = {record["run_id"]: record for record in records}
    executed = 0
    for dataset in ablations["datasets"]:
        for mode in ablations["modes"]:
            if (dataset, mode) not in SUPPORTED_DATASETS:
                print(f"Skipping unsupported condition {dataset}/{mode}", flush=True)
                continue
            data = get_dataset(
                dataset, mode, root=config["data_root"], offline=args.offline
            )
            data.num_classes = int(data.metadata["summary"]["num_classes"])
            data.num_features = int(data.x.shape[1])
            for variant in ablations["variants"]:
                variant_name = variant["name"]
                overrides = {
                    key: value for key, value in variant.items() if key != "name"
                }
                model_config = {**base_model_config, **overrides}
                model = build_model(
                    model_name, data.num_features, data.num_classes, model_config
                )
                print(
                    f"[{dataset}/{mode}] {model_name}/{variant_name}: "
                    f"{count_parameters(model)} trainable parameters, "
                    f"embedding_dim={getattr(model, 'embedding_dim', 'n/a')}",
                    flush=True,
                )
                for split in splits:
                    run_id = run_identifier(
                        model_name, dataset, mode, split, variant_name
                    )
                    existing = index.get(run_id)
                    if (
                        existing
                        and existing.get("status") == "completed"
                        and not args.force
                    ):
                        continue
                    if args.limit is not None and executed >= args.limit:
                        print(f"Reached --limit {args.limit}; stopping.", flush=True)
                        return _finalize(records, output_dir, expected)
                    record = _run_one(
                        config, dataset, mode, model_name, variant_name,
                        model_config, data, split, training_params, device,
                        checkpoint_dir, log_dir,
                    )
                    index[run_id] = record
                    records = [item for item in records if item["run_id"] != run_id]
                    records.append(record)
                    records.sort(key=lambda item: item["run_id"])
                    atomic_write_json(results_path, records)
                    executed += 1
                    accuracy = record.get("accuracy")
                    print(
                        f"{run_id:48} {record['status']:9} "
                        f"acc={None if accuracy is None else round(accuracy, 4)} "
                        f"epoch={record.get('best_epoch')}",
                        flush=True,
                    )
    return _finalize(records, output_dir, expected)


def _finalize(records, output_dir: Path, expected) -> int:
    """Write the CSV mirror and the ablation completeness report."""
    status = validate_completeness(records, expected)
    write_csv(output_dir / "ablation_results.csv", records)
    atomic_write_json(
        output_dir / "ablation_status.json",
        {
            "experiment": "ablation",
            "expected_runs": status["expected"],
            "completed_runs": status["completed"],
            "failed_runs": status["failed"],
            "missing_runs": status["missing"],
            "complete": status["complete"],
            "failed_run_ids": status["failed_run_ids"],
            "missing_run_ids": status["missing_run_ids"],
        },
    )
    print(
        f"\nCompleted {status['completed']}/{status['expected']} ablation runs "
        f"({status['failed']} failed, {status['missing']} missing).",
        flush=True,
    )
    print(f"Results: {(output_dir / 'ablation_results.json').resolve()}", flush=True)
    return 0 if status["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
