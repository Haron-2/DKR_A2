"""Model architectures for the A2 heterophily comparison.

All models implement the common interface ``forward(x, edge_index) -> logits`` and
``get_embeddings(x, edge_index) -> embeddings`` where the embeddings are the
representation immediately before the classifier.
"""

from models.h2gcn import H2GCN
from models.mlp import MLP
from models.registry import MODEL_NAMES, build_model, count_parameters
from models.vanilla_gnn import GAT, GCN

__all__ = [
    "GAT",
    "GCN",
    "H2GCN",
    "MLP",
    "MODEL_NAMES",
    "build_model",
    "count_parameters",
]
