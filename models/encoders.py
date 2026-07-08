"""Physical feature encoders: project raw features to 1024d alignment space.

MPHEncoder includes dispersion regularization to prevent expression collapse
(chemically diverse molecules mapping to indistinguishable latent codes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLP(nn.Module):
    """3-layer residual MLP with LayerNorm and SiLU activation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(out_dim)

        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        h = F.silu(self.norm1(self.fc1(x)))
        h = F.silu(self.norm2(self.fc2(h)))
        h = self.norm_out(self.fc3(h))
        return h + residual


class MPHEncoder(nn.Module):
    """Encode Persistence Image → 1024d with dispersion regularization.

    The dispersion loss penalizes inter-molecule cosine similarity above a
    threshold, preventing collapse of diverse topologies into indistinguishable
    latent codes that starve the MoE gate of routing signal.
    """

    def __init__(self, in_dim: int = 19200, hidden_dim: int = 1024, out_dim: int = 1024):
        super().__init__()
        self.mlp = ResidualMLP(in_dim, hidden_dim, out_dim)

    def forward(self, mph: torch.Tensor) -> torch.Tensor:
        return self.mlp(mph)

    def dispersion_loss(self, h: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
        """Penalize high cosine similarity between different molecules' codes.

        Args:
            h: (B, D) encoded MPH representations.
            margin: cosine similarity above this value is penalized.

        Returns:
            scalar dispersion loss (0 when all pairwise sims ≤ margin or B < 2).
        """
        if h.shape[0] < 2:
            return torch.tensor(0.0, device=h.device, dtype=h.dtype)

        h_norm = F.normalize(h, dim=-1)
        sim = torch.matmul(h_norm, h_norm.T)  # (B, B)

        mask = ~torch.eye(h.shape[0], dtype=torch.bool, device=h.device)
        off_diag = sim[mask]

        violation = F.relu(off_diag - margin)
        return violation.mean()


class PLSEncoder(nn.Module):
    """Encode Persistent Laplacian spectra (256d) → 1024d."""

    def __init__(self, in_dim: int = 256, hidden_dim: int = 512, out_dim: int = 1024):
        super().__init__()
        self.mlp = ResidualMLP(in_dim, hidden_dim, out_dim)

    def forward(self, pls: torch.Tensor) -> torch.Tensor:
        return self.mlp(pls)


class EGNNEncoder(nn.Module):
    """Encode EGNN global features (128d) → 1024d."""

    def __init__(self, in_dim: int = 128, hidden_dim: int = 512, out_dim: int = 1024):
        super().__init__()
        self.mlp = ResidualMLP(in_dim, hidden_dim, out_dim)

    def forward(self, egnn: torch.Tensor) -> torch.Tensor:
        return self.mlp(egnn)


class RSASEncoder(nn.Module):
    """Encode atom-level RSAS features → global 1024d via attention pooling."""

    def __init__(self, in_dim: int = 64, hidden_dim: int = 256, out_dim: int = 1024):
        super().__init__()
        self.atom_mlp = ResidualMLP(in_dim, hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)
        self.out_mlp = ResidualMLP(hidden_dim, hidden_dim, out_dim)

    def forward(self, rsas: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        h = self.atom_mlp(rsas)
        attn_scores = self.attn(h).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~atom_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
        h_pooled = (h * attn_weights).sum(dim=1)
        return self.out_mlp(h_pooled)
