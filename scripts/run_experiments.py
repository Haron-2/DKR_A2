"""Primary experiment runner: 4 models x 3 datasets x 2 conditions x 10 splits.

Results are written incrementally after every run, so an interrupted session can be
resumed by simply re-running the same command: completed runs are skipped.
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
    DATASETS,
    MODES,
    PRIMARY_MODELS,
    SPLITS,
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
from models.registry import (
    MODEL_NAMES,
    build_model,
    count_parameters,
    model_config_with_defaults,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "full.json"


def load_config(path: Path, args: argparse.Namespace) -> dict:
    """Read the JSON configuration and apply command-line overrides."""
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    config.setdefault("seed_base", 42)
    config.setdefault("datasets", list(DATASETS))
    config.setdefault("modes", list(MODES))
    config.setdefault("models", list(PRIMARY_MODELS))
    config.setdefault("splits", list(SPLITS))
    config.setdefault("training", {})
    config.setdefault("model_config", {})
    config.setdefault("data_root", str(DEFAULT_ROOT))
    config.setdefault("output_dir", str(PROJECT_ROOT / "results"))
    config.setdefault("export_embeddings", True)
    config.setdefault("embedding_split", 0)

    if args.datasets:
        config["datasets"] = args.datasets
    if args.modes:
        config["modes"] = args.modes
    if args.models:
        config["models"] = [name.upper() for name in args.models]
    if args.splits:
        config["splits"] = args.splits
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
    if args.patience is not None:
        config["training"]["patience"] = args.patience
    if args.device:
        config["training"]["device"] = args.device
    if args.seed_base is not None:
        config["seed_base"] = args.seed_base
    if args.root:
        config["data_root"] = str(args.root)
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    if args.no_embeddings:
        config["export_embeddings"] = False
    return config


def default_training_params(config: dict) -> dict:
    """Shared optimizer / early-stopping configuration for every model."""
    params = {
        "lr": 0.01,
        "weight_decay": 5e-4,
        "dropout": 0.5,
        "patience": 50,
        "min_delta": 1e-4,
        "max_epochs": 1000,
        "log_history": True,
    }
    params.update(config.get("training", {}))
    params.pop("device", None)
    return params


def print_parameter_table(datasets, modes, models, model_config, data_root) -> None:
    """Print trainable parameter counts for every architecture/condition."""
    print("\nTrainable parameters", flush=True)
    print(
        f"{'dataset':10} {'mode':8} {'model':6} {'params':>12} {'embedding':>10}",
        flush=True,
    )
    for dataset in datasets:
        for mode in modes:
            if (dataset, mode) not in SUPPORTED_DATASETS:
                continue
            data = get_dataset(dataset, mode, root=data_root, offline=True)
            for model_name in models:
                model = build_model(
                    model_name, data.x.shape[1], data.num_classes, model_config
                )
                embedding_dim = getattr(model, "embedding_dim", None)
                if embedding_dim is None:
                    embedding_dim = getattr(model, "hidden_dim", None)
                print(
                    f"{dataset:10} {mode:8} {model_name:6} "
                    f"{count_parameters(model):>12} {str(embedding_dim):>10}",
                    flush=True,
                )
    print(flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS))
    parser.add_argument("--modes", nargs="+", choices=list(MODES))
    parser.add_argument(
        "--models", nargs="+", choices=[name.lower() for name in MODEL_NAMES]
    )
    parser.add_argument("--splits", nargs="+", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Run at most N pending runs.")
    parser.add_argument("--force", action="store_true", help="Re-run completed runs.")
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--print-parameters", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config, args)
    torch.set_num_threads(max(1, args.threads))

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    embeddings_dir = output_dir / "embeddings"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    model_config = model_config_with_defaults(config.get("model_config"))
    training_params = default_training_params(config)
    device = config["training"].get("device", "cpu")
    seed_base = int(config["seed_base"])
    embedding_split = int(config["embedding_split"])

    if args.print_parameters:
        print_parameter_table(
            config["datasets"], config["modes"], config["models"],
            model_config, config["data_root"],
        )
        return 0

    records = load_results(results_path)
    index = {record["run_id"]: record for record in records}
    expected = expected_run_ids(
        config["models"], config["datasets"], config["modes"], config["splits"]
    )
    print(
        f"Primary experiment: {len(expected)} planned runs "
        f"({len(config['models'])} models x {len(config['datasets'])} datasets x "
        f"{len(config['modes'])} conditions x {len(config['splits'])} splits)",
        flush=True,
    )

    requested = [
        (dataset, mode)
        for dataset in config["datasets"]
        for mode in config["modes"]
    ]
    conditions = [
        condition for condition in requested if condition in SUPPORTED_DATASETS
    ]
    unsupported = [
        condition for condition in requested if condition not in SUPPORTED_DATASETS
    ]
    if unsupported:
        print(f"Skipping unsupported conditions: {unsupported}", flush=True)

    executed = 0
    for dataset, mode in conditions:
        data = get_dataset(dataset, mode, root=config["data_root"], offline=args.offline)
        data.num_classes = int(data.metadata["summary"]["num_classes"])
        data.num_features = int(data.x.shape[1])
        for model_name in config["models"]:
            model = build_model(
                model_name, data.num_features, data.num_classes, model_config
            )
            print(
                f"[{dataset}/{mode}] {model_name}: "
                f"{count_parameters(model)} trainable parameters",
                flush=True,
            )
            for split in config["splits"]:
                run_id = run_identifier(model_name, dataset, mode, split)
                existing = index.get(run_id)
                if (
                    existing
                    and existing.get("status") == "completed"
                    and not args.force
                ):
                    continue
                if args.limit is not None and executed >= args.limit:
                    print(f"Reached --limit {args.limit}; stopping.", flush=True)
                    return _finalize(records, config, results_path, output_dir, expected)
                seed = seed_for_split(seed_base, split)
                embedding_path = None
                if config.get("export_embeddings") and split == embedding_split:
                    embedding_path = (
                        embeddings_dir
                        / f"embeddings_{model_name.lower()}_{dataset}_{mode}"
                        f"_split{split}.pt"
                    )
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
                        embedding_path=embedding_path,
                    )
                    write_history_csv(log_dir / f"{run_id}.csv", history)
                except Exception as error:  # noqa: BLE001 - recorded, not hidden
                    logging.exception("Run %s failed", run_id)
                    record = {
                        "run_id": run_id,
                        "model": model_name,
                        "dataset": dataset,
                        "mode": mode,
                        "split": split,
                        "seed": seed,
                        "variant": "full",
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "configuration": {
                            "training": training_params,
                            "model": model_config,
                        },
                    }
                index[run_id] = record
                records = [item for item in records if item["run_id"] != run_id]
                records.append(record)
                records.sort(key=lambda item: item["run_id"])
                atomic_write_json(results_path, records)
                executed += 1
                accuracy = record.get("accuracy")
                macro_f1 = record.get("macro_f1")
                print(
                    f"{run_id:38} {record['status']:9} "
                    f"acc={None if accuracy is None else round(accuracy, 4)} "
                    f"f1={None if macro_f1 is None else round(macro_f1, 4)} "
                    f"epoch={record.get('best_epoch')} "
                    f"time={round(record.get('training_time_seconds') or 0.0, 1)}s",
                    flush=True,
                )

    return _finalize(records, config, results_path, output_dir, expected)


def _finalize(records, config, results_path, output_dir, expected) -> int:
    """Write the CSV mirror and the completeness report."""
    status = validate_completeness(records, expected)
    write_csv(output_dir / "results.csv", records)
    atomic_write_json(
        output_dir / "run_status.json",
        {
            "experiment": "primary",
            "expected_runs": status["expected"],
            "completed_runs": status["completed"],
            "failed_runs": status["failed"],
            "missing_runs": status["missing"],
            "complete": status["complete"],
            "failed_run_ids": status["failed_run_ids"],
            "missing_run_ids": status["missing_run_ids"],
            "config": {
                "datasets": config["datasets"],
                "modes": config["modes"],
                "models": config["models"],
                "splits": config["splits"],
                "training": config["training"],
                "model_config": config["model_config"],
            },
        },
    )
    print(
        f"\nCompleted {status['completed']}/{status['expected']} runs "
        f"({status['failed']} failed, {status['missing']} missing).",
        flush=True,
    )
    if status["missing_run_ids"]:
        print(f"Missing (first 10): {status['missing_run_ids'][:10]}", flush=True)
    if status["failed_run_ids"]:
        print(f"Failed: {status['failed_run_ids']}", flush=True)
    print(f"Results: {results_path.resolve()}", flush=True)
    return 0 if status["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())

