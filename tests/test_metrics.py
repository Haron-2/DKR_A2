import math

import pytest
import torch
from torch_geometric.data import Data

from analysis.duplicate_audit import (
    audit_dataset,
    duplicate_statistics,
    find_duplicate_groups,
    leakage_statistics,
)
from analysis.homophily import (
    adjusted_homophily,
    edge_homophily,
    homophily_metrics,
    node_homophily,
)
from analysis.stats import holm_correction, paired_comparisons, structural_delta


def undirected(pairs):
    edges = []
    for left, right in pairs:
        edges.append((left, right))
        edges.append((right, left))
    return torch.tensor(edges, dtype=torch.int64).t().contiguous()


def test_homophily_matches_hand_computation():
    edge_index = undirected([(0, 3), (1, 3), (2, 3), (0, 1)])
    y = torch.tensor([0, 0, 0, 1], dtype=torch.int64)
    metrics = homophily_metrics(edge_index, y)
    assert math.isclose(metrics["edge_homophily"], 0.25)
    assert math.isclose(metrics["node_homophily"], 0.25)
    assert math.isclose(
        metrics["adjusted_homophily"], (0.25 - 0.53125) / (1 - 0.53125)
    )
    assert math.isclose(
        metrics["adjusted_homophily_node_proportions"],
        (0.25 - 0.625) / (1 - 0.625),
    )
    assert math.isclose(edge_homophily(edge_index, y), 0.25)
    assert math.isclose(node_homophily(edge_index, y), 0.25)
    assert math.isclose(adjusted_homophily(edge_index, y), (0.25 - 0.53125) / 0.46875)


def test_homophily_rejects_self_loops_and_duplicates():
    y = torch.tensor([0, 1, 0], dtype=torch.int64)
    with pytest.raises(ValueError):
        homophily_metrics(torch.tensor([[0, 0], [0, 1]]), y)
    with pytest.raises(ValueError):
        homophily_metrics(torch.tensor([[0, 1, 1], [1, 0, 0]]), y)


def test_duplicate_groups_and_statistics():
    x = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [2.0, 2.0], [2.0, 2.0]]
    )
    y = torch.tensor([0, 0, 1, 1, 2, 0], dtype=torch.int64)
    groups = find_duplicate_groups(x)
    assert sorted(sorted(group) for group in groups) == [[0, 1], [2, 3], [4, 5]]
    statistics = duplicate_statistics(x, y)
    assert statistics["duplicate_groups"] == 3
    assert statistics["duplicate_pairs"] == 3
    assert statistics["duplicate_vertices"] == 6
    assert math.isclose(statistics["duplicate_percentage"], 100.0)
    assert statistics["num_unique_feature_vectors"] == 3
    assert statistics["groups_with_identical_labels"] == 2
    assert statistics["groups_with_conflicting_labels"] == 1
    assert math.isclose(statistics["label_agreement_percentage"], 100.0 * 4 / 6)


def test_duplicate_tolerance_is_respected():
    x = torch.tensor([[0.0, 0.0], [1e-7, 0.0], [1e-3, 0.0]])
    groups = find_duplicate_groups(x, tolerance=1e-6)
    # Distinct integer-valued groups are not affected by the grid shortcut, and
    # 1e-3 is far outside the tolerance.
    assert sorted(sorted(group) for group in groups) == [[0, 1]]


def test_leakage_statistics_on_known_split():
    x = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [2.0, 2.0]]
    )
    y = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    train = torch.zeros((5, 1), dtype=torch.bool)
    test = torch.zeros((5, 1), dtype=torch.bool)
    train[0, 0] = True
    test[[1, 2], 0] = True
    records = leakage_statistics(x, train, test, y)
    assert len(records) == 1
    record = records[0]
    assert record["leaking_test_nodes"] == 1
    assert math.isclose(record["leakage_percentage"], 50.0)
    assert record["crossing_groups"] == 1
    assert record["crossing_groups_with_identical_labels"] == 1


def test_audit_dataset_reports_the_dataset_name():
    data = Data(
        x=torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        y=torch.tensor([0, 0, 1], dtype=torch.int64),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
    )
    data.train_mask = torch.tensor([[True], [False], [True]])
    data.val_mask = torch.tensor([[False], [False], [False]])
    data.test_mask = torch.tensor([[False], [True], [False]])
    data.metadata = {"dataset": "toy"}
    audit = audit_dataset(data)
    assert audit["dataset"] == "toy"
    assert audit_dataset(data, name="override")["dataset"] == "override"


def test_holm_correction_matches_manual_values():
    adjusted = holm_correction([0.01, 0.04, 0.03])
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)
    assert holm_correction([]) == []


def make_records(values_by_model, dataset="toy", mode="raw"):
    records = []
    for model, values in values_by_model.items():
        for split, value in enumerate(values):
            records.append(
                {
                    "run_id": f"{model}_{dataset}_{mode}_s{split}",
                    "model": model,
                    "dataset": dataset,
                    "mode": mode,
                    "split": split,
                    "seed": split,
                    "variant": "full",
                    "status": "completed",
                    "accuracy": value,
                    "macro_f1": value,
                    "epochs_trained": 10,
                    "best_epoch": 5,
                    "trainable_parameters": 100,
                    "training_time_seconds": 1.0,
                    "evaluation_time_seconds": 0.1,
                }
            )
    return records


def test_structural_delta_and_paired_comparison():
    records = make_records(
        {
            "MLP": [0.50, 0.52, 0.48, 0.51, 0.49],
            "GCN": [0.60, 0.63, 0.57, 0.62, 0.58],
        }
    )
    per_split, summary = structural_delta(records)
    assert len(per_split) == 5
    assert summary[0]["model"] == "GCN"
    assert summary[0]["mean_delta"] == pytest.approx(0.1)
    comparisons = paired_comparisons(records, comparisons=("GCN",))
    assert len(comparisons) == 1
    row = comparisons[0]
    assert row["n_pairs"] == 5
    assert row["mean_difference"] == pytest.approx(0.1)
    assert row["p_value_ttest"] < 0.01
    assert row["p_value_holm"] == pytest.approx(row["p_value_ttest"])
    assert row["ci95_low"] < row["mean_difference"] < row["ci95_high"]
    assert row["significant_holm_0.05"] is True
