import copy

import pytest
import torch

import engine.trainer as trainer_module
from engine.early_stopping import EarlyStopping
from engine.runner import execute_run
from engine.trainer import (
    TrainingConfig,
    compute_metrics,
    resolve_split_masks,
    train_single_split,
)
from models.mlp import MLP


def make_model(data):
    torch.manual_seed(0)
    return MLP(data.num_features, data.num_classes, hidden_dim=8)


def test_single_epoch_training_and_result_schema(synthetic_data, tmp_path):
    model = make_model(synthetic_data)
    before = copy.deepcopy(model.state_dict())
    config = TrainingConfig(
        max_epochs=3, patience=50, seed=1, checkpoint_path=str(tmp_path / "run.pt")
    )
    result = train_single_split(model, synthetic_data, 0, config)
    assert result["status"] == "completed"
    assert result["epochs_trained"] == 3
    assert result["best_epoch"] >= 1
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert result["trainable_parameters"] > 0
    assert len(result["history"]["train_loss"]) == 3
    assert len(result["history"]["val_loss"]) == 3
    assert (tmp_path / "run.pt").is_file()
    assert any(
        not torch.allclose(before[key], value)
        for key, value in model.state_dict().items()
    )


def test_only_training_labels_affect_gradients(synthetic_data):
    train_mask, val_mask, test_mask = resolve_split_masks(synthetic_data, 0)
    modified = synthetic_data.clone()
    modified.y = synthetic_data.y.clone()
    outside = val_mask | test_mask
    modified.y[outside] = (modified.y[outside] + 1) % synthetic_data.num_classes

    config_a = TrainingConfig(max_epochs=1, seed=7, checkpoint_path=None)
    model_a = make_model(synthetic_data)
    train_single_split(model_a, synthetic_data, 0, config_a)

    config_b = TrainingConfig(max_epochs=1, seed=7, checkpoint_path=None)
    model_b = make_model(modified)
    train_single_split(model_b, modified, 0, config_b)

    for key, value in model_a.state_dict().items():
        assert torch.allclose(value, model_b.state_dict()[key]), (
            "labels outside the training mask must not influence the update"
        )


def test_training_labels_do_change_gradients(synthetic_data):
    modified = synthetic_data.clone()
    modified.y = synthetic_data.y.clone()
    train_mask, _, _ = resolve_split_masks(synthetic_data, 0)
    first = int(torch.nonzero(train_mask).flatten()[0])
    modified.y[first] = (modified.y[first] + 1) % synthetic_data.num_classes

    config = TrainingConfig(max_epochs=1, seed=7, checkpoint_path=None)
    model_a = make_model(synthetic_data)
    train_single_split(model_a, synthetic_data, 0, config)
    model_b = make_model(modified)
    train_single_split(model_b, modified, 0, config)

    assert any(
        not torch.allclose(value, model_b.state_dict()[key])
        for key, value in model_a.state_dict().items()
    )


def test_early_stopping_tracks_best_and_restores(tmp_path):
    model = MLP(3, 2, hidden_dim=4)
    checkpoint = tmp_path / "best.pt"
    stopper = EarlyStopping(patience=2, min_delta=1e-4, checkpoint_path=checkpoint)
    stopper.step(1.0, model, 1)
    stopper.step(0.9, model, 2)
    best_state = copy.deepcopy(model.state_dict())
    assert stopper.best_epoch == 2
    assert not stopper.step(0.9005, model, 3), "improvement below min_delta"
    assert stopper.stop is False
    assert stopper.step(0.95, model, 4) is True
    assert stopper.best_epoch == 2
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    assert stopper.restore(model) is True
    for key, value in best_state.items():
        assert torch.allclose(value, model.state_dict()[key])
    assert checkpoint.is_file()


def test_non_finite_validation_loss_never_becomes_best(tmp_path):
    model = MLP(3, 2, hidden_dim=4)
    stopper = EarlyStopping(patience=1, checkpoint_path=tmp_path / "best.pt")
    stopper.step(0.5, model, 1)
    stopper.step(float("nan"), model, 2)
    assert stopper.best_score == 0.5
    assert stopper.best_epoch == 1
    assert stopper.stop is True


def test_test_metrics_are_computed_once(synthetic_data, monkeypatch):
    calls = {"count": 0}
    original = trainer_module.compute_metrics

    def counting_metrics(y_true, y_pred):
        calls["count"] += 1
        return original(y_true, y_pred)

    monkeypatch.setattr(trainer_module, "compute_metrics", counting_metrics)
    config = TrainingConfig(max_epochs=4, seed=0, checkpoint_path=None)
    model = make_model(synthetic_data)
    train_single_split(model, synthetic_data, 0, config)
    assert calls["count"] == 1


def test_accuracy_and_macro_f1():
    y_true = torch.tensor([0, 0, 1, 2, 2])
    y_pred = torch.tensor([0, 1, 1, 2, 0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(0.6)


def test_checkpoint_paths_are_unique_per_run(tmp_path, synthetic_data):
    records = []
    for index, model_name in enumerate(("MLP", "GCN")):
        record, _ = execute_run(
            model_name=model_name,
            data=synthetic_data,
            dataset="toy",
            mode="raw",
            split=0,
            seed=index,
            model_config={"hidden_dim": 8},
            training_params={"max_epochs": 1, "patience": 5},
            device="cpu",
            checkpoint_dir=tmp_path / "checkpoints",
        )
        records.append(record)
    assert records[0]["checkpoint_path"] != records[1]["checkpoint_path"]
    assert all((tmp_path / "checkpoints").glob("*.pt"))
