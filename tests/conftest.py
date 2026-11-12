import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_synthetic_data(
    num_nodes: int = 30,
    num_features: int = 6,
    num_classes: int = 3,
    num_splits: int = 10,
    seed: int = 0,
) -> Data:
    """Small transductive graph with ten mutually exclusive mask columns."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, num_features, generator=generator)
    y = torch.randint(0, num_classes, (num_nodes,), generator=generator)
    ring = torch.stack([torch.arange(num_nodes), (torch.arange(num_nodes) + 1) % num_nodes])
    extra = torch.randint(0, num_nodes, (max(1, num_nodes // 3), 2), generator=generator).t()
    edge_index = torch.cat([ring, extra], dim=1)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    train_masks = torch.zeros(num_nodes, num_splits, dtype=torch.bool)
    val_masks = torch.zeros_like(train_masks)
    test_masks = torch.zeros_like(train_masks)
    train_end = max(1, int(0.6 * num_nodes))
    val_end = train_end + max(1, int(0.2 * num_nodes))
    for split in range(num_splits):
        permutation = torch.randperm(num_nodes, generator=generator)
        train_masks[permutation[:train_end], split] = True
        val_masks[permutation[train_end:val_end], split] = True
        test_masks[permutation[val_end:], split] = True
    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_masks,
        val_mask=val_masks,
        test_mask=test_masks,
    )
    data.num_features = num_features
    data.num_classes = num_classes
    return data


@pytest.fixture()
def synthetic_data() -> Data:
    return make_synthetic_data()
