import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv


class GCN(nn.Module):
    """Two-layer graph convolution with standard self-loops and GCN normalisation.

    ``GCNConv`` adds self-loops and applies the symmetric normalisation
    ``D^{-1/2} (A + I) D^{-1/2}`` internally.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.conv1 = GCNConv(self.num_features, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.num_classes)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return the 64-dimensional hidden state that feeds the output layer."""
        hidden = F.relu(self.conv1(x, edge_index))
        return F.dropout(hidden, p=self.dropout_p, training=self.training)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.get_embeddings(x, edge_index), edge_index)


class GAT(nn.Module):
    """Two-layer graph attention network.

    The hidden layer uses ``heads`` attention heads with
    ``hidden_dim // heads`` channels each, concatenated to ``hidden_dim``. The
    output layer uses a single averaging head, so the logits have exactly
    ``num_classes`` columns per node (not ``num_classes * heads``).
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        heads: int = 8,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads for GAT")
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.dropout_p = float(dropout)
        self.conv1 = GATConv(
            self.num_features,
            self.hidden_dim // self.heads,
            heads=self.heads,
            concat=True,
            dropout=self.dropout_p,
            add_self_loops=True,
        )
        self.conv2 = GATConv(
            self.hidden_dim,
            self.num_classes,
            heads=1,
            concat=False,
            dropout=self.dropout_p,
            add_self_loops=True,
        )

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return the concatenated multi-head hidden representation."""
        hidden = F.elu(self.conv1(x, edge_index))
        return F.dropout(hidden, p=self.dropout_p, training=self.training)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.get_embeddings(x, edge_index), edge_index)
