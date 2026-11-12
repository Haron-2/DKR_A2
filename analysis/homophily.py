import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected

from data.validation import validate_dataset


def _validate_graph(edge_index: torch.Tensor, y: torch.Tensor) -> None:
    if y.ndim != 1 or y.dtype != torch.int64:
        raise ValueError("y must be int64 with shape [N]")
    if y.numel() and y.min() < 0:
        raise ValueError("y must contain known, nonnegative class labels")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.dtype != torch.int64 or edge_index.device != y.device:
        raise ValueError("edge_index must be int64 on the same device as y")
    if edge_index.numel():
        if edge_index.min() < 0 or edge_index.max() >= y.numel():
            raise ValueError("edge_index contains a node outside 0..N-1")
        if (edge_index[0] == edge_index[1]).any():
            raise ValueError("Homophily expects a graph without self-loops")
        edge_ids = edge_index[0] * y.numel() + edge_index[1]
        if torch.unique(edge_ids).numel() != edge_index.shape[1]:
            raise ValueError("Homophily expects a graph without repeated edges")
    if not is_undirected(edge_index, num_nodes=y.numel()):
        raise ValueError("Homophily expects both orientations of every edge")


@torch.no_grad()
def homophily_metrics(edge_index: torch.Tensor, y: torch.Tensor) -> dict:
    """Calculate homophily on a simple undirected graph without modifying inputs.

    Each edge must be stored in both directions. Node homophily averages over
    all vertices, giving isolated vertices a zero contribution, as in PyG.
    Adjusted homophily uses degree-weighted class proportions, not class sizes:
    q_k = sum(degree[v] for y[v] == k) / sum(degree).
    See https://www.dgl.ai/dgl_docs/en/2.0.x/generated/dgl.adjusted_homophily.html.
    Undefined quantities are returned as None. No edges makes edge and adjusted
    homophily undefined; one class among edge endpoints makes adjusted homophily
    undefined. An empty graph also has undefined node homophily.

    ``adjusted_homophily_node_proportions`` is additionally reported using plain
    node proportions p_k = n_k / N (the formula stated in the assignment). Both
    quantities are documented because they answer slightly different questions:
    the degree-weighted version is the expected homophily of a random edge walk,
    the node-proportion version the expected homophily of two random vertices.
    """
    _validate_graph(edge_index, y)
    num_nodes = y.numel()
    num_edges = edge_index.shape[1]
    degree = torch.bincount(edge_index[0], minlength=num_nodes)
    result = {
        "edge_homophily": None,
        "node_homophily": 0.0 if num_nodes else None,
        "adjusted_homophily": None,
        "adjusted_homophily_node_proportions": None,
        "expected_edge_homophily": None,
        "node_isolate_policy": "zero_contribution",
    }
    if num_edges == 0:
        return result

    same_label = y[edge_index[0]] == y[edge_index[1]]
    same_label_degree = torch.bincount(
        edge_index[0, same_label], minlength=num_nodes
    )
    node_ratios = same_label_degree.to(torch.float64) / degree.clamp_min(1)
    h_edge = float(same_label.to(torch.float64).mean())
    _, class_index = torch.unique(y, return_inverse=True)
    class_degree = torch.bincount(class_index, weights=degree.to(torch.float64))
    class_proportions = class_degree / degree.sum()
    expected = float(class_proportions.square().sum())
    denominator = 1.0 - expected

    class_counts = torch.bincount(class_index).to(torch.float64)
    node_proportions = class_counts / class_counts.sum()
    expected_node = float(node_proportions.square().sum())
    denominator_node = 1.0 - expected_node

    result["edge_homophily"] = h_edge
    result["node_homophily"] = float(node_ratios.mean())
    result["expected_edge_homophily"] = expected
    if denominator > 0:
        result["adjusted_homophily"] = (h_edge - expected) / denominator
    if denominator_node > 0:
        result["adjusted_homophily_node_proportions"] = (
            h_edge - expected_node
        ) / denominator_node
    return result


def edge_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float | None:
    """Return the fraction of edges joining vertices with the same label."""
    return homophily_metrics(edge_index, y)["edge_homophily"]


def node_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float | None:
    """Return mean neighborhood homophily, with zero contribution from isolates."""
    return homophily_metrics(edge_index, y)["node_homophily"]


def adjusted_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float | None:
    """Return edge homophily adjusted for degree-weighted class proportions."""
    return homophily_metrics(edge_index, y)["adjusted_homophily"]


def summarize_dataset(data: Data) -> dict:
    """Summarize the model graph using all labels for descriptive reporting only.

    These full-label statistics must not guide model selection using test labels.
    num_edges_undirected counts each pair once; average_degree equals 2|E|/N.
    """
    summary = validate_dataset(data)
    fields = (
        "num_nodes", "num_features", "num_classes", "num_edges_directed",
        "num_edges_undirected", "average_degree", "isolated_nodes", "class_counts",
    )
    return {
        **{field: summary[field] for field in fields},
        **homophily_metrics(data.edge_index, data.y),
    }
