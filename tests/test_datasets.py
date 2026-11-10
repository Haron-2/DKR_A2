import pytest
import torch

from data.loader import SUPPORTED_DATASETS, get_dataset
from data.validation import validate_dataset


CONDITIONS = [condition for condition in SUPPORTED_DATASETS]


@pytest.fixture(scope="module", params=CONDITIONS, ids=["_".join(c) for c in CONDITIONS])
def condition(request):
    name, mode = request.param
    try:
        data = get_dataset(name, mode, offline=True)
    except FileNotFoundError:
        pytest.skip(f"{name}/{mode} is not prepared locally")
    return name, mode, data


def test_dataset_metadata(condition):
    name, mode, data = condition
    assert data.num_features == data.x.shape[1]
    assert data.num_classes == int(data.y.max()) + 1
    assert data.metadata["dataset"] == name
    assert data.metadata["mode"] == mode
    assert data.metadata["graph_policy"] == "undirected_coalesced_without_self_loops"
    assert data.metadata["split_ids"] == list(range(10))
    assert data.metadata["fingerprint"]


def test_dataset_tensors_and_masks(condition):
    name, mode, data = condition
    summary = validate_dataset(data)
    assert summary["num_nodes"] == data.x.shape[0]
    assert data.edge_index.dtype == torch.int64
    assert data.x.dtype == torch.float32
    assert torch.isfinite(data.x).all()
    for field in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, field)
        assert mask.dtype == torch.bool
        assert mask.shape == (data.x.shape[0], 10)
        assert mask.any()
    membership = (
        data.train_mask.to(torch.int8)
        + data.val_mask.to(torch.int8)
        + data.test_mask.to(torch.int8)
    )
    assert torch.all(membership == 1), "train/val/test partitions must be disjoint"
    assert int(data.num_classes) == summary["num_classes"]


def test_filter_filtered_conditions_are_documented(condition):
    name, mode, data = condition
    if mode == "filtered":
        assert "Platonov" in data.metadata["split_source"]
        if name == "actor":
            assert data.metadata["duplicate_removal"].startswith("none")
        else:
            assert data.metadata["duplicate_removal"].startswith("upstream")
    else:
        assert data.metadata["duplicate_removal"].startswith("none")


def test_actor_conditions_share_the_same_graph():
    try:
        raw = get_dataset("actor", "raw", offline=True)
        filtered = get_dataset("actor", "filtered", offline=True)
    except FileNotFoundError:
        pytest.skip("Actor conditions are not prepared locally")
    assert torch.equal(raw.x, filtered.x)
    assert torch.equal(raw.y, filtered.y)
    assert raw.edge_index.shape == filtered.edge_index.shape
    assert not torch.equal(raw.train_mask, filtered.train_mask)
    assert filtered.metadata["split_seed"] is not None
    assert "custom" in filtered.metadata["split_source"]
