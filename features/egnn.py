"""EGNN: Equivariant Graph Neural Network with frozen coordinate updates.

Key constraint: During property prediction, pos_update is locked to zero
to preserve physical structural integrity of the 3D conformation.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class EGNNLayer(nn.Module):
    """Single EGNN layer with optional position update freezing."""

    def __init__(self, hidden_dim: int = 128, freeze_coords: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.freeze_coords = freeze_coords

        # Message network: [h_i, h_j, ||x_i - x_j||^2] → message
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Coordinate update network
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Node update network: [h_i, aggregated_message] → new_h
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Node features (B*N, hidden_dim)
            x: Node coordinates (B*N, 3)
            edge_index: (2, E) edge indices
        Returns:
            updated h, updated x (frozen = same as input)
        """
        src, dst = edge_index[0], edge_index[1]

        # Pairwise distances
        dist = torch.sum((x[src] - x[dst]) ** 2, dim=-1, keepdim=True)  # (E, 1)

        # Edge messages
        edge_input = torch.cat([h[src], h[dst], dist], dim=-1)
        messages = self.edge_mlp(edge_input)  # (E, hidden_dim)

        # Aggregate messages to nodes
        aggregated = torch.zeros_like(h)
        aggregated = aggregated.index_add(0, dst, messages)

        # Coordinate update (FROZEN for property prediction)
        if not self.freeze_coords:
            coord_weights = self.coord_mlp(messages)  # (E, 1)
            delta_x = (x[src] - x[dst]) * coord_weights  # (E, 3)
            x_update = torch.zeros_like(x)
            x_update = x_update.index_add(0, dst, delta_x)
            x = x + x_update

        # Node update with residual
        h_new = self.node_mlp(torch.cat([h, aggregated], dim=-1))
        h = h + h_new

        return h, x


class FrozenEGNN(nn.Module):
    """Multi-layer EGNN with frozen coordinates for molecular property prediction.

    Args:
        num_layers: Number of EGNN layers (default 4).
        hidden_dim: Hidden dimension (default 128).
        freeze_coords: Lock position updates to zero.
    """

    def __init__(self, num_layers: int = 4, hidden_dim: int = 128, freeze_coords: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.freeze_coords = freeze_coords

        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, freeze_coords=freeze_coords)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: Node features (N, hidden_dim) for single molecule.
            x: Coordinates (N, 3).
            edge_index: (2, E) edges.

        Returns:
            h_geo: Global geometric feature vector (hidden_dim,).
        """
        for layer in self.layers:
            h, x = layer(h, x, edge_index)

        # Global mean pooling
        h_geo = h.mean(dim=0)  # (hidden_dim,)
        return h_geo
