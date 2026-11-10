"""Run orchestration helpers shared by the primary and ablation experiments."""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Iterable

import torch

from engine.trainer import TrainingConfig, set_global_seed, train_single_split
from models.registry import build_model, model_config_with_defaults


logger = logging.getLogger(__name__)

PRIMARY_MODELS = ("MLP", "GCN", "GAT", "H2GCN")
DATASETS = ("chameleon", "squirrel", "actor")
MODES = ("raw", "filtered")
SPLITS = tuple(range(10))
RESULT_FIELDS = (
    "run_id", "model", "dataset", "mode", "split", "seed", "variant", "status",
    "accuracy", "macro_f1", "best_val_loss", "best_epoch", "epochs_trained",
    "trainable_parameters", "training_time_seconds", "evaluation_time_seconds",
    "test_loss", "num_train_nodes", "num_val_nodes", "num_test_nodes",
    "checkpoint_path", "timestamp", "configuration",
)


def run_identifier(
    model: str, dataset: str, mode: str, split: int, variant: str | None = None
) -> str:
    """Unique, readable run identifier."""
    base = f"{model.lower()}_{dataset}_{mode}_s{split}"
    if variant in (None, "", "full"):
        return base
    return f"{model.lower()}_{variant}_{dataset}_{mode}_s{split}"


def seed_for_split(seed_base: int, split: int, variant_offset: int = 0) -> int:
    """Seed shared by all models that are compared on the same split."""
    return int(seed_base) + int(split) + int(variant_offset)


def atomic_write_json(path: str | Path, payload) -> None:
    """Write JSON via a temporary file so interrupted runs cannot corrupt it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as target:
            temporary_path = Path(target.name)
            json.dump(payload, target, ensure_ascii=False, indent=2)
            target.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_csv(path: str | Path, rows: Iterable[dict]) -> None:
    """Write a CSV with the union of all keys, preserving first-seen order."""
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def load_results(path: str | Path) -> list[dict]:
    """Load existing run records, tolerating a missing or empty file."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "runs" in payload:
        return list(payload["runs"])
    return list(payload)


def completed_run_ids(records: Iterable[dict]) -> set[str]:
    return {
        record["run_id"] for record in records if record.get("status") == "completed"
    }


def expected_run_ids(
    models: Iterable[str],
    datasets: Iterable[str],
    modes: Iterable[str],
    splits: Iterable[int],
    variants: Iterable[str | None] = (None,),
) -> list[str]:
    """Expected run identifiers in deterministic order."""
    identifiers = []
    for dataset in datasets:
        for mode in modes:
            for model in models:
                for variant in variants:
                    for split in splits:
                        identifiers.append(run_identifier(model, dataset, mode, split, variant))
    return identifiers


def validate_completeness(records: list[dict], expected: Iterable[str]) -> dict:
    """Compare expected against recorded runs and classify every identifier."""
    expected = list(expected)
    by_id = {}
    for record in records:
        by_id[record["run_id"]] = record
    completed = [run_id for run_id in expected if by_id.get(run_id, {}).get("status") == "completed"]
    failed = [run_id for run_id in expected if by_id.get(run_id, {}).get("status") == "failed"]
    missing = [run_id for run_id in expected if run_id not in by_id]
    return {
        "expected": len(expected),
        "completed": len(completed),
        "failed": len(failed),
        "missing": len(missing),
        "complete": len(completed) == len(expected),
        "failed_run_ids": failed,
        "missing_run_ids": missing,
        "unexpected_run_ids": sorted(set(by_id) - set(expected)),
    }


def execute_run(
    *,
    model_name: str,
    data,
    dataset: str,
    mode: str,
    split: int,
    seed: int,
    model_config: dict,
    training_params: dict,
    device: str,
    checkpoint_dir: str | Path,
    variant: str | None = None,
    embedding_path: str | Path | None = None,
) -> tuple[dict, dict | None]:
    """Train one (model, condition, split) run and optionally export embeddings.

    Returns the run record and the per-epoch loss history. Every run receives its
    own checkpoint path and its own model instance.
    """
    resolved_model_config = model_config_with_defaults(model_config)
    run_id = run_identifier(model_name, dataset, mode, split, variant)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_id}.pt"

    set_global_seed(seed)
    model = build_model(
        model_name, data.num_features, data.num_classes, resolved_model_config
    )
    config = TrainingConfig(
        seed=seed,
        device=device,
        checkpoint_path=str(checkpoint_path),
        **training_params,
    )
    started = time.perf_counter()
    result = train_single_split(model, data, split, config)
    wall_time = time.perf_counter() - started
    history = result.pop("history", None)

    record = {
        "run_id": run_id,
        "model": model_name,
        "dataset": dataset,
        "mode": mode,
        "split": int(split),
        "seed": int(seed),
        "variant": variant or "full",
        "status": result["status"],
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "best_val_loss": result["best_val_loss"],
        "best_epoch": result["best_epoch"],
        "epochs_trained": result["epochs_trained"],
        "trainable_parameters": result["trainable_parameters"],
        "training_time_seconds": result["training_time_seconds"],
        "evaluation_time_seconds": result["evaluation_time_seconds"],
        "test_loss": result["test_loss"],
        "num_train_nodes": result["num_train_nodes"],
        "num_val_nodes": result["num_val_nodes"],
        "num_test_nodes": result["num_test_nodes"],
        "checkpoint_path": result["checkpoint_path"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_time_seconds": wall_time,
        "configuration": {
            "training": result["configuration"],
            "model": resolved_model_config,
        },
    }

    if embedding_path is not None and record["status"] == "completed":
        export_embeddings(
            model, data, Path(embedding_path), record, device=device
        )
    return record, history


@torch.no_grad()
def export_embeddings(
    model,
    data,
    path: Path,
    record: dict,
    *,
    device: str = "cpu",
) -> dict:
    """Save the pre-classifier representation of the restored best model."""
    model.eval()
    model.to(device)
    embeddings = model.get_embeddings(
        data.x.to(device), data.edge_index.to(device)
    )
    payload = {
        "embedding": embeddings.detach().cpu().float(),
        "y": data.y.detach().cpu().long(),
        "node_index": torch.arange(data.x.shape[0], dtype=torch.int64),
        "model": record["model"],
        "dataset": record["dataset"],
        "mode": record["mode"],
        "split": int(record["split"]),
        "variant": record["variant"],
        "embedding_dim": int(embeddings.shape[1]),
        "accuracy": record["accuracy"],
        "macro_f1": record["macro_f1"],
        "configuration": record["configuration"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "embedding_dim": int(embeddings.shape[1]),
        "num_nodes": int(embeddings.shape[0]),
        "model": record["model"],
        "dataset": record["dataset"],
        "mode": record["mode"],
        "split": int(record["split"]),
        "accuracy": record["accuracy"],
    }


def write_history_csv(path: str | Path, history: dict | None) -> None:
    """Persist per-epoch training and validation losses for one run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for index, (train_loss, val_loss) in enumerate(
            zip(history["train_loss"], history["val_loss"]), start=1
        ):
            writer.writerow([index, train_loss, val_loss])
