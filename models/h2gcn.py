"""H2GCN core-principles implementation (Zhu et al., 2020, arXiv:2006.11468).

Implemented principles:

* ego-neighbour separation (the projected input features ``H0`` are kept in every
  concatenation and are never overwritten by aggregated messages);
* aggregation over the exact one-hop and two-hop neighbourhoods;
* concatenation of intermediate representations from both propagation rounds.

Normalisation
-------------
Both adjacency matrices are *row normalised* (each row sums to one), i.e. the
propagation performs mean aggregation over the exact one-hop / two-hop neighbour set.
Self connections and one-hop pairs are removed from ``A2`` before normalisation, and
both matrices are strictly binary before normalisation. ``normalization="sym"`` is
available for symmetric ``D^{-1/2} A D^{-1/2}`` normalisation; the default ``row``
matches the neighbourhood-averaging operator of the reference implementation.

This is an implementation of the core H2GCN principles, not a bit-exact
reproduction of every detail of the authors' code:

* no trainable transformations are applied inside the diffusion iterations;
* the only dropout is applied once to the final concatenated representation before
  the linear classifier;
* the embedding width is ``9 x hidden_dim`` for the full model (3 blocks of
  ``hidden_dim`` in round one and 6 in round two) and is intentionally not resized
  to match the other architectures.
"""

from collections import OrderedDict

import torch
from torch import nn


_ADJACENCY_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_ADJACENCY_CACHE_LIMIT = 2


def _cached_adjacencies(key: tuple, builder) -> tuple:
    """LRU cache so repeated splits of one dataset reuse the 2-hop operator."""
    if key in _ADJACENCY_CACHE:
        _ADJACENCY_CACHE.move_to_end(key)
        return _ADJACENCY_CACHE[key]
    value = builder()
    _ADJACENCY_CACHE[key] = value
    while len(_ADJACENCY_CACHE) > _ADJACENCY_CACHE_LIMIT:
        _ADJACENCY_CACHE.popitem(last=False)
    return value


def clear_adjacency_cache() -> None:
    """Drop cached operators (used by tests and between dataset conditions)."""
    _ADJACENCY_CACHE.clear()


DENSE_DENSITY_THRESHOLD = 0.05
DENSE_MIN_NODES = 512


def apply_operator(operator: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply a sparse or dense propagation operator to a feature matrix."""
    if operator.layout == torch.sparse_coo:
        return torch.sparse.mm(operator, x)
    return operator @ x


def _to_operator(operator: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Store very dense operators densely (pure performance optimisation).

    The exact two-hop set of a high-degree graph can cover a large share of all
    possible pairs, where dense BLAS multiplication is much faster than the sparse
    kernel. The arithmetic is unchanged.
    """
    if num_nodes >= DENSE_MIN_NODES:
        capacity = float(num_nodes) * float(num_nodes)
        if operator._nnz() > DENSE_DENSITY_THRESHOLD * capacity:
            return operator.to_dense()
    return operator


def build_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the coalesced binary adjacency without self-loops.

    Repeated edges are merged, self-loops are dropped and both directions of every
    edge are kept if present in the input. ``num_nodes`` may exceed the number of
    referenced nodes (isolated vertices are allowed).
    """
    indices = edge_index.to(torch.int64)
    keep = indices[0] != indices[1]
    indices = indices[:, keep]
    values = torch.ones(indices.shape[1], dtype=dtype)
    adjacency = torch.sparse_coo_tensor(
        indices, values, (num_nodes, num_nodes), dtype=dtype
    ).coalesce()
    binary = torch.ones_like(adjacency.values())
    return torch.sparse_coo_tensor(
        adjacency.indices(), binary, (num_nodes, num_nodes), dtype=dtype
    ).coalesce()


def normalize_adjacency(
    adjacency: torch.Tensor,
    num_nodes: int,
    *,
    mode: str = "row",
) -> torch.Tensor:
    """Normalise a sparse binary adjacency matrix.

    ``mode="row"`` applies ``D^{-1} A`` (mean aggregation) and ``mode="sym"`` applies
    ``D^{-1/2} A D^{-1/2}``. Self-loops are never added.
    """
    adjacency = adjacency.coalesce()
    indices = adjacency.indices()
    values = adjacency.values().to(torch.float32)
    degree = torch.bincount(indices[0], minlength=num_nodes).to(torch.float32)
    if mode == "row":
        weights = values / degree[indices[0]].clamp_min(1.0)
    elif mode == "sym":
        weights = values / (
            degree[indices[0]].clamp_min(1.0).sqrt()
            * degree[indices[1]].clamp_min(1.0).sqrt()
        )
    else:
        raise ValueError(f"Unsupported normalization: {mode}")
    return torch.sparse_coo_tensor(
        indices, weights, (num_nodes, num_nodes), dtype=torch.float32
    ).coalesce()


def build_two_hop_adjacency(
    one_hop: torch.Tensor,
    num_nodes: int,
    *,
    block_rows: int = 256,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the exact two-hop adjacency, excluding self and one-hop pairs.

    ``A2 = binarize(A @ A)`` with every entry removed for which the same pair is a
    self connection or an existing one-hop edge, i.e.

    ``A2 = (A^2 > 0) & (A == 0) & (I == 0)``.

    The sparse product is evaluated in row blocks so that high-degree graphs do not
    materialise the complete product at once. The returned matrix is binary and is
    *not* normalised; use :func:`normalize_adjacency` afterwards.
    """
    if one_hop.layout != torch.sparse_coo:
        raise ValueError("one_hop must be a sparse COO tensor")
    one_hop = one_hop.coalesce()
    indices = one_hop.indices()
    values = one_hop.values()
    one_hop_keys = torch.sort(indices[0] * num_nodes + indices[1]).values

    row_chunks: list[torch.Tensor] = []
    col_chunks: list[torch.Tensor] = []
    for start in range(0, num_nodes, block_rows):
        stop = min(start + block_rows, num_nodes)
        in_block = (indices[0] >= start) & (indices[0] < stop)
        if not bool(in_block.any()):
            continue
        block = torch.sparse_coo_tensor(
            torch.stack([indices[0, in_block] - start, indices[1, in_block]]),
            values[in_block],
            (stop - start, num_nodes),
            dtype=dtype,
        ).coalesce()
        product = torch.sparse.mm(block, one_hop).coalesce()
        if product._nnz() == 0:
            continue
        rows = product.indices()[0] + start
        cols = product.indices()[1]
        keep = rows != cols
        if bool(keep.any()):
            keys = rows * num_nodes + cols
            keep &= ~torch.isin(keys, one_hop_keys)
        if bool(keep.any()):
            row_chunks.append(rows[keep])
            col_chunks.append(cols[keep])
    if not row_chunks:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.int64),
            torch.zeros(0, dtype=dtype),
            (num_nodes, num_nodes),
            dtype=dtype,
        ).coalesce()

    rows = torch.cat(row_chunks)
    cols = torch.cat(col_chunks)
    # Blocks partition the rows and each coalesced product has unique (row, col)
    # pairs, so no duplicate entries can remain after concatenation.
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        torch.ones(rows.numel(), dtype=dtype),
        (num_nodes, num_nodes),
        dtype=dtype,
    ).coalesce()


def build_adjacencies(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    use_one_hop: bool = True,
    use_two_hop: bool = True,
    normalization: str = "row",
    block_rows: int = 256,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Build the normalised one-hop and two-hop operators used by H2GCN."""
    one_hop = build_adjacency(edge_index, num_nodes)
    a1 = None
    if use_one_hop:
        a1 = _to_operator(
            normalize_adjacency(one_hop, num_nodes, mode=normalization), num_nodes
        )
    a2 = None
    if use_two_hop:
        a2 = _to_operator(
            normalize_adjacency(
                build_two_hop_adjacency(one_hop, num_nodes, block_rows=block_rows),
                num_nodes,
                mode=normalization,
            ),
            num_nodes,
        )
    return a1, a2


class H2GCN(nn.Module):
    """H2GCN with configurable one-hop / two-hop branches (ablations)."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        *,
        use_one_hop: bool = True,
        use_two_hop: bool = True,
        normalization: str = "row",
        block_rows: int = 256,
    ) -> None:
        super().__init__()
        if not (use_one_hop or use_two_hop):
            raise ValueError("H2GCN needs at least one propagation branch")
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.use_one_hop = bool(use_one_hop)
        self.use_two_hop = bool(use_two_hop)
        self.normalization = str(normalization)
        self.block_rows = int(block_rows)
        self.projection = nn.Linear(self.num_features, self.hidden_dim)
        branch_count = int(self.use_one_hop) + int(self.use_two_hop)
        first_round = self.hidden_dim * (1 + branch_count)
        second_round = first_round * branch_count
        self.embedding_dim = first_round + second_round
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)

    def _adjacencies(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        key = (
            edge_index.data_ptr(),
            tuple(edge_index.shape),
            int(num_nodes),
            str(edge_index.device),
            self.use_one_hop,
            self.use_two_hop,
            self.normalization,
            self.block_rows,
        )

        def builder() -> tuple[torch.Tensor | None, torch.Tensor | None]:
            a1, a2 = build_adjacencies(
                edge_index,
                num_nodes,
                use_one_hop=self.use_one_hop,
                use_two_hop=self.use_two_hop,
                normalization=self.normalization,
                block_rows=self.block_rows,
            )
            device = edge_index.device
            if a1 is not None:
                a1 = a1.to(device)
            if a2 is not None:
                a2 = a2.to(device)
            return a1, a2

        return _cached_adjacencies(key, builder)

    def _propagate(
        self, operator: torch.Tensor | None, x: torch.Tensor
    ) -> torch.Tensor:
        if operator is None:
            raise RuntimeError("Propagation branch is disabled")
        return apply_operator(operator, x)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return the concatenation ``[H0, R1(1), R2(1), R1(2), R2(2)]``."""
        num_nodes = x.shape[0]
        a1, a2 = self._adjacencies(edge_index, num_nodes)
        h0 = torch.relu(self.projection(x))
        blocks = [h0]
        first_round = [h0]
        if self.use_one_hop:
            r1 = self._propagate(a1, h0)
            blocks.append(r1)
            first_round.append(r1)
        if self.use_two_hop:
            r2 = self._propagate(a2, h0)
            blocks.append(r2)
            first_round.append(r2)
        h1 = torch.cat(first_round, dim=1)
        if self.use_one_hop:
            blocks.append(self._propagate(a1, h1))
        if self.use_two_hop:
            blocks.append(self._propagate(a2, h1))
        return torch.dropout(
            torch.cat(blocks, dim=1), p=self.dropout_p, train=self.training
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.get_embeddings(x, edge_index))

