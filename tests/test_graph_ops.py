import pytest
import torch

from models.h2gcn import (
    H2GCN,
    build_adjacency,
    build_adjacencies,
    build_two_hop_adjacency,
    normalize_adjacency,
)


def make_undirected(pairs: list[tuple[int, int]]) -> torch.Tensor:
    edges = []
    for left, right in pairs:
        edges.append((left, right))
        edges.append((right, left))
    return torch.tensor(edges, dtype=torch.int64).t().contiguous()


def support(matrix: torch.Tensor) -> set[tuple[int, int]]:
    matrix = matrix.coalesce()
    return nonzero_entries(matrix)


def nonzero_entries(matrix: torch.Tensor) -> set[tuple[int, int]]:
    matrix = matrix.coalesce()
    values = matrix.values()
    indices = matrix.indices()
    entries = set()
    for position in range(indices.shape[1]):
        if float(values[position]) != 0.0:
            entries.add((int(indices[0, position]), int(indices[1, position])))
    return entries


@pytest.mark.parametrize(
    "edge_index",
    [
        make_undirected([(0, 1), (1, 2), (2, 3)]),
        torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.int64),
        torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.int64),
    ],
)
def test_one_hop_removes_self_loops_and_duplicates(edge_index):
    adjacency = build_adjacency(edge_index, num_nodes=4)
    assert set(torch.unique(adjacency.values()).tolist()) == {1.0}
    assert (0, 0) not in nonzero_entries(adjacency)
    entries = nonzero_entries(adjacency)
    assert (0, 0) not in entries
    assert len(entries) == len(set(entries))
    assert (0, 1) in entries


def test_exact_two_hop_on_path_graph():
    edge_index = make_undirected([(0, 1), (1, 2), (2, 3)])
    one_hop = build_adjacency(edge_index, num_nodes=4)
    two_hop = build_two_hop_adjacency(one_hop, num_nodes=4)
    assert nonzero_entries(two_hop) == {(0, 2), (2, 0), (1, 3), (3, 1)}


def test_two_hop_excludes_first_order_and_self_on_triangle():
    edge_index = make_undirected([(0, 1), (1, 2), (0, 2)])
    one_hop = build_adjacency(edge_index, num_nodes=3)
    two_hop = build_two_hop_adjacency(one_hop, num_nodes=3)
    assert two_hop._nnz() == 0
    dense = two_hop.to_dense()
    assert torch.count_nonzero(dense) == 0


def test_star_center_has_no_two_hop_neighbours():
    edge_index = make_undirected([(0, 1), (0, 2), (0, 3)])
    one_hop = build_adjacency(edge_index, num_nodes=4)
    two_hop = build_two_hop_adjacency(one_hop, num_nodes=4)
    entries = nonzero_entries(two_hop)
    assert {(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)} == entries
    assert not any(row == 0 for row, _ in entries)


def test_two_hop_isolated_nodes_and_empty_neighbourhoods():
    edge_index = torch.tensor([[0], [1]], dtype=torch.int64)
    one_hop = build_adjacency(edge_index, num_nodes=5)
    two_hop = build_two_hop_adjacency(one_hop, num_nodes=5)
    assert two_hop.shape == (5, 5)
    assert two_hop._nnz() == 0
    a1, a2 = build_adjacencies(edge_index, num_nodes=5)
    assert a1 is not None and a2 is not None
    assert a2._nnz() == 0


def test_row_normalisation_sums_to_one():
    edge_index = make_undirected([(0, 1), (1, 2), (2, 3)])
    one_hop = build_adjacency(edge_index, num_nodes=4)
    a1, a2 = build_adjacencies(edge_index, num_nodes=4)
    row_sums_one_hop = a1.to_dense().sum(dim=1)
    assert torch.allclose(row_sums_one_hop, torch.ones(4))
    row_sums_two_hop = a2.to_dense().sum(dim=1)
    expected = torch.tensor([1.0, 1.0, 1.0, 1.0])
    assert torch.allclose(row_sums_two_hop, expected)


def test_symmetric_normalisation_is_symmetric():
    edge_index = make_undirected([(0, 1), (1, 2), (2, 3)])
    one_hop = build_adjacency(edge_index, num_nodes=4)
    symmetric = normalize_adjacency(one_hop, num_nodes=4, mode="sym")
    dense = symmetric.to_dense()
    assert torch.allclose(dense, dense.t())
    assert torch.allclose(dense.diagonal(), torch.zeros(4))


def test_duplicate_edges_do_not_create_two_hop_entries():
    clean = make_undirected([(0, 1), (1, 2)])
    doubled = torch.cat([clean, clean], dim=1)
    one_hop_clean = build_adjacency(clean, num_nodes=3)
    one_hop_doubled = build_adjacency(doubled, num_nodes=3)
    assert torch.allclose(one_hop_clean.to_dense(), one_hop_doubled.to_dense())
    two_hop_clean = build_two_hop_adjacency(one_hop_clean, num_nodes=3)
    two_hop_doubled = build_two_hop_adjacency(one_hop_doubled, num_nodes=3)
    assert torch.allclose(two_hop_clean.to_dense(), two_hop_doubled.to_dense())


def test_h2gcn_forward_with_isolated_node():
    edge_index = make_undirected([(0, 1)])
    x = torch.randn(4, 5)
    model = H2GCN(5, 3, hidden_dim=8)
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
    assert logits.shape == (4, 3)
    assert torch.isfinite(logits).all()


def test_two_hop_matches_bruteforce_on_random_graph():
    generator = torch.Generator().manual_seed(7)
    num_nodes = 25
    random_edges = torch.randint(0, num_nodes, (60, 2), generator=generator)
    one_hop = build_adjacency(random_edges.t(), num_nodes=num_nodes)
    dense = one_hop.to_dense()
    product = (dense @ dense > 0).float()
    expected = product * (1.0 - dense) * (
        1.0 - torch.eye(num_nodes, dtype=torch.float32)
    )
    computed = build_two_hop_adjacency(one_hop, num_nodes=num_nodes).to_dense()
    assert torch.equal(computed > 0, expected > 0)
