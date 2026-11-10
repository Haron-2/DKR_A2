"""Duplicate-node and train-test leakage audit.

Duplicate nodes are feature vectors whose Euclidean distance is below
``tolerance`` (default ``1e-6``). Grouping is hash/nearest-neighbour based: float
rows are first deduplicated exactly, then near-identical unique rows are merged
with a KD-tree radius query, so no ``N x N`` distance matrix is ever built.

The audit is deliberately independent from the training pipeline: no label,
group or leakage information is ever passed to a model or to model selection.
"""

from __future__ import annotations

import numpy as np
import torch


DEFAULT_TOLERANCE = 1e-6


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _exact_groups(features: np.ndarray) -> tuple[list[list[int]], int]:
    """Group exactly identical feature rows with a sort-based hash."""
    unique_rows, inverse = np.unique(features, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    buckets: dict[int, list[int]] = {}
    for node, unique_id in enumerate(inverse.tolist()):
        buckets.setdefault(unique_id, []).append(node)
    return [members for members in buckets.values() if len(members) > 1], len(unique_rows)


def _tolerance_groups(
    features: np.ndarray,
    tolerance: float,
    *,
    chunk_size: int = 256,
) -> tuple[list[list[int]], int]:
    """Merge near-identical unique rows using blocked BLAS distance computation.

    Only used when the features are not (near) integral, where an exact float
    comparison would not be equivalent to the tolerance criterion.
    """
    unique_rows, inverse = np.unique(features, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    unique_rows = unique_rows.astype(np.float64)
    union_find = _UnionFind(len(unique_rows))
    squared_norms = np.einsum("ij,ij->i", unique_rows, unique_rows)
    threshold = float(tolerance) ** 2
    for start in range(0, len(unique_rows), chunk_size):
        stop = min(start + chunk_size, len(unique_rows))
        block = unique_rows[start:stop]
        distances = (
            squared_norms[start:stop, None]
            + squared_norms[None, :]
            - 2.0 * block @ unique_rows.T
        )
        rows, columns = np.nonzero(distances < threshold)
        for local_row, column in zip(rows.tolist(), columns.tolist()):
            if column > start + local_row:
                union_find.union(start + local_row, column)
    buckets: dict[int, list[int]] = {}
    for node, unique_id in enumerate(inverse.tolist()):
        buckets.setdefault(union_find.find(unique_id), []).append(node)
    return [members for members in buckets.values() if len(members) > 1], len(unique_rows)


def find_duplicate_groups(
    x: torch.Tensor | np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[list[int]]:
    """Return duplicate groups of node indices (groups of size >= 2 only).

    Two nodes belong to the same group when ``||x_i - x_j||_2 < tolerance``.
    """
    groups, _ = _duplicate_groups_with_method(x, tolerance)
    return groups


def _duplicate_groups_with_method(
    x: torch.Tensor | np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[list[list[int]], str]:
    features = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    if features.ndim != 2:
        raise ValueError("x must have shape [N, F]")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    integral = bool(
        np.all(np.abs(features - np.rint(features)) < tolerance / 2.0)
    )
    if integral:
        # Distinct integer vectors differ by at least 1 in L2, far above the
        # tolerance, so exact grouping is equivalent to the tolerance criterion.
        groups, unique_count = _exact_groups(features)
        return groups, "exact_integer_grid"
    groups, unique_count = _tolerance_groups(features, tolerance)
    return groups, "blocked_euclidean_tolerance"


def duplicate_statistics(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    """Summarise duplicates: pairs, groups, affected nodes and label agreement."""
    features = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    groups, method = _duplicate_groups_with_method(features, tolerance)
    num_nodes = int(features.shape[0])
    duplicate_vertices = sum(len(group) for group in groups)
    duplicate_pairs = sum(len(group) * (len(group) - 1) // 2 for group in groups)
    statistics = {
        "tolerance": float(tolerance),
        "grouping_method": method,
        "num_nodes": num_nodes,
        "num_unique_feature_vectors": int(np.unique(features, axis=0).shape[0]),
        "duplicate_groups": len(groups),
        "duplicate_pairs": int(duplicate_pairs),
        "duplicate_vertices": int(duplicate_vertices),
        "duplicate_percentage": (
            100.0 * duplicate_vertices / num_nodes if num_nodes else 0.0
        ),
        "largest_group": int(max((len(group) for group in groups), default=0)),
    }
    if y is not None:
        labels = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        consistent = [group for group in groups if np.unique(labels[group]).size == 1]
        consistent_nodes = sum(len(group) for group in consistent)
        statistics.update(
            {
                "groups_with_identical_labels": len(consistent),
                "groups_with_conflicting_labels": len(groups) - len(consistent),
                "label_agreement_percentage": (
                    100.0 * consistent_nodes / duplicate_vertices
                    if duplicate_vertices
                    else 0.0
                ),
            }
        )
    return statistics


def leakage_statistics(
    x: torch.Tensor | np.ndarray,
    train_mask: torch.Tensor | np.ndarray,
    test_mask: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[dict]:
    """Quantify train-test leakage from equivalent feature vectors per split.

    ``train_mask`` and ``test_mask`` are ``[N, S]`` boolean arrays. For every split
    the leaking test nodes are those whose duplicate group also contains a training
    node; label agreement is computed over the groups that cross the boundary.
    """
    features = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    labels = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    train = (
        train_mask.detach().cpu().numpy()
        if isinstance(train_mask, torch.Tensor)
        else np.asarray(train_mask)
    )
    test = (
        test_mask.detach().cpu().numpy()
        if isinstance(test_mask, torch.Tensor)
        else np.asarray(test_mask)
    )
    if train.shape != test.shape or train.ndim != 2:
        raise ValueError("train_mask and test_mask must share shape [N, S]")
    groups = find_duplicate_groups(features, tolerance=tolerance)
    node_group = {node: index for index, group in enumerate(groups) for node in group}

    records = []
    for split in range(train.shape[1]):
        train_nodes = set(np.flatnonzero(train[:, split]).tolist())
        test_nodes = np.flatnonzero(test[:, split])
        test_node_set = set(test_nodes.tolist())
        leaking_nodes = [
            int(node)
            for node in test_nodes
            if node in node_group
            and any(
                other in train_nodes for other in groups[node_group[int(node)]]
            )
        ]
        crossing_groups = [
            group
            for group in groups
            if set(group) & train_nodes and set(group) & test_node_set
        ]
        agreement = [
            group
            for group in crossing_groups
            if np.unique(labels[group]).size == 1
        ]
        records.append(
            {
                "split": split,
                "num_train_nodes": len(train_nodes),
                "num_test_nodes": int(test_nodes.size),
                "leaking_test_nodes": len(leaking_nodes),
                "leakage_percentage": (
                    100.0 * len(leaking_nodes) / int(test_nodes.size)
                    if test_nodes.size
                    else 0.0
                ),
                "crossing_groups": len(crossing_groups),
                "crossing_groups_with_identical_labels": len(agreement),
                "crossing_label_agreement_percentage": (
                    100.0 * len(agreement) / len(crossing_groups)
                    if crossing_groups
                    else 0.0
                ),
            }
        )
    return records


def audit_dataset(data, tolerance: float = DEFAULT_TOLERANCE) -> dict:
    """Run the duplicate and leakage audit for all ten splits of one dataset."""
    duplicates = duplicate_statistics(data.x, data.y, tolerance=tolerance)
    leakage = leakage_statistics(
        data.x, data.train_mask, data.test_mask, data.y, tolerance=tolerance
    )
    percentages = [record["leakage_percentage"] for record in leakage]
    return {
        "dataset": getattr(data, "name", None),
        "duplicates": duplicates,
        "leakage": leakage,
        "leakage_summary": {
            "mean_leakage_percentage": float(np.mean(percentages)),
            "max_leakage_percentage": float(np.max(percentages)),
            "min_leakage_percentage": float(np.min(percentages)),
            "leaking_test_nodes_total": int(
                sum(record["leaking_test_nodes"] for record in leakage)
            ),
        },
    }
