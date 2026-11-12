import csv
import json
from pathlib import Path

import pytest
import torch

from analysis.stats import aggregate, paired_comparisons
from engine.runner import (
    atomic_write_json,
    completed_run_ids,
    execute_run,
    expected_run_ids,
    load_results,
    validate_completeness,
)
from scripts import analyze_results
from scripts.run_experiments import main as run_experiments_main
from viz.plots import plot_metric_comparison
from viz.tsne_plotter import load_embedding, plot_tsne


def synthetic_records(offset=0.0):
    records = []
    for split in range(10):
        for model, base in (
            ("MLP", 0.40), ("GCN", 0.45), ("GAT", 0.42), ("H2GCN", 0.50)
        ):
            records.append(
                {
                    "run_id": f"{model.lower()}_toy_raw_s{split}",
                    "model": model,
                    "dataset": "toy",
                    "mode": "raw",
                    "split": split,
                    "seed": split,
                    "variant": "full",
                    "status": "completed",
                    "accuracy": base + offset + 0.01 * (split % 3),
                    "macro_f1": base + offset + 0.01 * (split % 3),
                    "epochs_trained": 20,
                    "best_epoch": 10,
                    "trainable_parameters": 100,
                    "training_time_seconds": 0.5,
                    "evaluation_time_seconds": 0.05,
                }
            )
    return records


def test_execute_run_persists_embeddings(tmp_path, synthetic_data):
    embedding_path = tmp_path / "embeddings" / "embeddings_mlp_toy_raw_split0.pt"
    record, history = execute_run(
        model_name="MLP",
        data=synthetic_data,
        dataset="toy",
        mode="raw",
        split=0,
        seed=3,
        model_config={"hidden_dim": 8},
        training_params={"max_epochs": 2, "patience": 5},
        device="cpu",
        checkpoint_dir=tmp_path / "checkpoints",
        embedding_path=embedding_path,
    )
    assert record["status"] == "completed"
    assert history is not None and len(history["train_loss"]) == 2
    assert embedding_path.is_file()
    payload = load_embedding(embedding_path)
    assert payload["embedding"].shape == (synthetic_data.x.shape[0], 8)
    assert payload["embedding_dim"] == 8
    assert payload["model"] == "MLP"
    assert payload["split"] == 0
    assert payload["y"].shape == (synthetic_data.x.shape[0],)


def test_result_serialization_and_completeness(tmp_path):
    records = synthetic_records()
    path = tmp_path / "results.json"
    atomic_write_json(path, records)
    loaded = load_results(path)
    assert len(loaded) == len(records)
    assert completed_run_ids(loaded) == {record["run_id"] for record in records}

    expected = expected_run_ids(
        ("MLP", "GCN", "GAT", "H2GCN"), ("toy",), ("raw",), range(10)
    )
    status = validate_completeness(loaded, expected)
    assert status["complete"] is True
    assert status["completed"] == 40

    incomplete = [record for record in records if record["split"] != 9]
    status = validate_completeness(incomplete, expected)
    assert status["complete"] is False
    assert status["missing"] == 4


def test_analysis_produces_all_tables(tmp_path):
    output = tmp_path / "analysis"
    summary = analyze_results.analyze(synthetic_records(), [], output)
    for metric in ("accuracy", "macro_f1"):
        assert summary["metrics"][metric]["aggregates"]
        assert (output / f"aggregate_{metric}.csv").is_file()
        assert (output / f"paired_tests_{metric}.csv").is_file()
    assert summary["metrics"]["accuracy"]["paired_comparisons"]
    aggregates = aggregate(synthetic_records(), "accuracy")
    assert all(row["n_completed"] == 10 for row in aggregates)
    comparisons = paired_comparisons(synthetic_records())
    assert len(comparisons) == 3
    assert all(row["n_pairs"] == 10 for row in comparisons)


def test_figure_generation(tmp_path):
    output = tmp_path / "figures" / "accuracy.png"
    path = plot_metric_comparison(
        aggregate(synthetic_records(), "accuracy"), "accuracy", output
    )
    assert path is not None and Path(path).is_file()
    assert Path(path).stat().st_size > 0
