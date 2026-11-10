import pytest
import torch
from torch import nn

from models.h2gcn import (
    H2GCN,
    apply_operator,
    build_adjacency,
    build_adjacencies,
    build_two_hop_adjacency,
)
from models.mlp import MLP
from models.registry import MODEL_NAMES, build_model, count_parameters
from models.vanilla_gnn import GAT, GCN


MODEL_CLASSES = {"MLP": MLP, "GCN": GCN, "GAT": GAT, "H2GCN": H2GCN}
NUM_CLASSES = 3


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_common_interface(name, synthetic_data):
    model = build_model(name, synthetic_data.num_features, NUM_CLASSES)
    assert isinstance(model, nn.Module)
    model.train()
    logits = model(synthetic_data.x, synthetic_data.edge_index)
    assert logits.shape == (synthetic_data.x.shape[0], NUM_CLASSES)
    assert torch.isfinite(logits).all()
    embeddings = model.get_embeddings(synthetic_data.x, synthetic_data.edge_index)
    assert embeddings.ndim == 2
    assert embeddings.shape[0] == synthetic_data.x.shape[0]
    assert torch.isfinite(embeddings).all()
    assert count_parameters(model) > 0


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_backward_and_eval_mode(name, synthetic_data):
    model = build_model(name, synthetic_data.num_features, NUM_CLASSES)
    model.train()
    logits = model(synthetic_data.x, synthetic_data.edge_index)
    loss = nn.functional.cross_entropy(
        logits[synthetic_data.train_mask[:, 0]], synthetic_data.y[synthetic_data.train_mask[:, 0]]
    )
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients and any(bool(gradient.abs().sum() > 0) for gradient in gradients)
    model.eval()
    with torch.no_grad():
        eval_logits = model(synthetic_data.x, synthetic_data.edge_index)
    assert torch.isfinite(eval_logits).all()


def test_mlp_ignores_edge_index(synthetic_data):
    torch.manual_seed(0)
    model = MLP(synthetic_data.num_features, NUM_CLASSES, hidden_dim=16, dropout=0.5)
    model.eval()
    with torch.no_grad():
        left = model(synthetic_data.x, synthetic_data.edge_index)
        shuffled = synthetic_data.edge_index.flip(0)
        right = model(synthetic_data.x, shuffled)
        empty = model(synthetic_data.x, torch.empty((2, 0), dtype=torch.long))
    assert torch.allclose(left, right)
    assert torch.allclose(left, empty)


def test_gat_output_and_embedding_dimensions(synthetic_data):
    model = GAT(synthetic_data.num_features, NUM_CLASSES, hidden_dim=64, heads=8)
    model.eval()
    with torch.no_grad():
        logits = model(synthetic_data.x, synthetic_data.edge_index)
        embeddings = model.get_embeddings(synthetic_data.x, synthetic_data.edge_index)
    assert logits.shape == (synthetic_data.x.shape[0], NUM_CLASSES)
    assert embeddings.shape == (synthetic_data.x.shape[0], 64)
    assert model.conv2.heads == 1


def test_h2gcn_embedding_dimension_and_ablations(synthetic_data):
    full = H2GCN(synthetic_data.num_features, NUM_CLASSES, hidden_dim=16)
    assert full.embedding_dim == 9 * 16
    no_two_hop = H2GCN(
        synthetic_data.num_features, NUM_CLASSES, hidden_dim=16, use_two_hop=False
    )
    assert no_two_hop.embedding_dim == 4 * 16
    no_one_hop = H2GCN(
        synthetic_data.num_features, NUM_CLASSES, hidden_dim=16, use_one_hop=False
    )
    assert no_one_hop.embedding_dim == 4 * 16
    for model in (full, no_two_hop, no_one_hop):
        model.eval()
        with torch.no_grad():
            embeddings = model.get_embeddings(
                synthetic_data.x, synthetic_data.edge_index
            )
            logits = model(synthetic_data.x, synthetic_data.edge_index)
        assert embeddings.shape == (synthetic_data.x.shape[0], model.embedding_dim)
        assert logits.shape == (synthetic_data.x.shape[0], NUM_CLASSES)
        assert torch.isfinite(logits).all()


def test_h2gcn_rejects_missing_branches(synthetic_data):
    with pytest.raises(ValueError):
        H2GCN(
            synthetic_data.num_features,
            NUM_CLASSES,
            use_one_hop=False,
            use_two_hop=False,
        )


def test_h2gcn_intermediate_block_order(synthetic_data):
    """The representation must follow [H0, R1(1), R2(1), R1(2), R2(2)]."""
    model = H2GCN(synthetic_data.num_features, NUM_CLASSES, hidden_dim=4, dropout=0.0)
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(synthetic_data.x, synthetic_data.edge_index)
        h0 = torch.relu(model.projection(synthetic_data.x))
        a1, a2 = build_adjacencies(
            synthetic_data.edge_index, synthetic_data.x.shape[0]
        )
        r1 = apply_operator(a1, h0)
        r2 = apply_operator(a2, h0)
        h1 = torch.cat([h0, r1, r2], dim=1)
        r1_second = apply_operator(a1, h1)
        r2_second = apply_operator(a2, h1)
    hidden = 4
    second = 3 * hidden
    assert embeddings.shape[1] == 9 * hidden
    assert torch.allclose(embeddings[:, :hidden], h0)
    assert torch.allclose(embeddings[:, hidden:2 * hidden], r1)
    assert torch.allclose(embeddings[:, 2 * hidden:second], r2)
    assert torch.allclose(embeddings[:, second:second + 3 * hidden], r1_second)
    assert torch.allclose(embeddings[:, second + 3 * hidden:], r2_second)
