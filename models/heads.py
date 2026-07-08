"""Task-specific prediction heads."""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Multi-label classification head: outputs raw logits.

    BCEWithLogitsLoss is used downstream (NOT BCE after Sigmoid) for
    numerical stability via the log-sum-exp trick.
    """

    def __init__(self, in_dim: int = 1024, num_classes: int = 1):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # raw logits — no sigmoid


class RegressionHead(nn.Module):
    """Regression head with LayerNorm for numerical stability."""

    def __init__(self, in_dim: int = 1024, out_dim: int = 1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc(x))
