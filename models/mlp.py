import torch
from torch import nn


class MLP(nn.Module):
    """Structure-blind baseline that only uses node features.

    ``edge_index`` is accepted for interface compatibility and is ignored, so the
    model predictions are invariant to the graph. Layer normalisation is used
    instead of batch normalisation: it has no batch-statistic dependency at
    inference time, which keeps full-graph transductive inference well defined.
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
        self.encoder = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
        )
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

    def get_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the hidden representation produced before the classifier."""
        return self.encoder(x)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.classifier(self.get_embeddings(x, edge_index))
