import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected


def validate_dataset(data: Data) -> dict:
    """Validate the model graph and all ten complete data splits."""
    if data.x is None or data.x.ndim != 2 or data.x.dtype != torch.float32:
        raise ValueError("x must be float32 with shape [N, F]")
    num_nodes, num_features = data.x.shape
    if num_nodes == 0 or num_features == 0 or not torch.isfinite(data.x).all():
        raise ValueError("x must be nonempty and contain only finite values")
    if data.y is None or data.y.shape != (num_nodes,) or data.y.dtype != torch.int64:
        raise ValueError("y must be int64 with shape [N]")
    classes = torch.unique(data.y)
    if not torch.equal(classes, torch.arange(len(classes))):
        raise ValueError("y must use consecutive class labels starting at zero")

    edge_index = data.edge_index
    if edge_index is None or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.dtype != torch.int64:
        raise ValueError("edge_index must be int64")
    if edge_index.numel():
        if edge_index.min() < 0 or edge_index.max() >= num_nodes:
            raise ValueError("edge_index contains a node outside 0..N-1")
        if (edge_index[0] == edge_index[1]).any():
            raise ValueError("Model graph must not contain self-loops")
        edge_ids = edge_index[0] * num_nodes + edge_index[1]
        if torch.unique(edge_ids).numel() != edge_index.shape[1]:
            raise ValueError("Model graph contains repeated edges")
    if not is_undirected(edge_index, num_nodes=num_nodes):
        raise ValueError("Model graph must contain both directions of every edge")

    masks = []
    for name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, name, None)
        if mask is None or mask.dtype != torch.bool or mask.shape != (num_nodes, 10):
            raise ValueError(f"{name} must be bool with shape [{num_nodes}, 10]")
        masks.append(mask)
    membership = sum(mask.to(torch.int8) for mask in masks)
    if not torch.all(membership == 1):
        raise ValueError("Each node must belong to exactly one partition per split")

    splits = []
    for split_id in range(10):
        split = {"split_id": split_id}
        for name, mask in zip(("train", "val", "test"), masks):
            labels = data.y[mask[:, split_id]]
            if labels.numel() == 0:
                raise ValueError(f"{name} is empty in split {split_id}")
            split[f"{name}_nodes"] = labels.numel()
            split[f"{name}_class_counts"] = torch.bincount(
                labels, minlength=len(classes)
            ).tolist()
        splits.append(split)

    degree = torch.bincount(edge_index[0], minlength=num_nodes)
    return {
        "num_nodes": num_nodes,
        "num_features": num_features,
        "num_classes": len(classes),
        "num_edges_directed": edge_index.shape[1],
        "num_edges_undirected": edge_index.shape[1] // 2,
        "average_degree": edge_index.shape[1] / num_nodes,
        "isolated_nodes": int((degree == 0).sum()),
        "class_counts": torch.bincount(data.y, minlength=len(classes)).tolist(),
        "splits": splits,
    }
