"""Sparse Mixture-of-Experts with Top-2 gating and load balancing."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """Single expert: 2-layer FFN with SiLU activation."""

    def __init__(self, dim: int = 1024, expansion: int = 2):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseMoE(nn.Module):
    """Top-K sparse Mixture-of-Experts.

    Args:
        dim: Input/output dimension.
        num_experts: Total number of experts (default 4).
        top_k: Number of experts activated per token (default 2).
    """

    def __init__(
        self,
        dim: int = 1024,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Router / gate
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # Expert FFNs
        self.experts = nn.ModuleList([
            ExpertFFN(dim) for _ in range(num_experts)
        ])

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, dim) input tokens.

        Returns:
            output: (B, dim) expert-processed output.
            aux_loss: Load balancing auxiliary loss.
        """
        B, D = x.shape
        device = x.device

        # Gate logits and routing weights
        gate_logits = self.gate(x)  # (B, num_experts)

        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)  # normalize within top-k

        # Normalize gate logits for load balancing
        gate_probs = F.softmax(gate_logits, dim=-1)  # (B, num_experts)

        # Dispatch tokens to experts
        output = torch.zeros_like(x)

        for expert_idx in range(self.num_experts):
            # Find which tokens are routed to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # (B,)
            if not expert_mask.any():
                continue

            # Get the weight for this expert
            weight_idx = (top_k_indices == expert_idx).float().argmax(dim=-1)  # (B,)
            expert_weights = top_k_weights[expert_mask, weight_idx[expert_mask]]  # (N_selected,)

            # Process through expert
            expert_input = x[expert_mask]  # (N_selected, D)
            expert_output = self.experts[expert_idx](expert_input)  # (N_selected, D)

            # Apply routing weight
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            output[expert_mask] += weighted_output

        output = self.out_proj(output)

        # Load balancing auxiliary loss
        # f_i = fraction of tokens dispatched to expert i
        expert_counts = torch.zeros(self.num_experts, device=device)
        for i in range(self.num_experts):
            expert_counts[i] = (top_k_indices == i).sum().float()
        f = expert_counts / (B * self.top_k)

        # P_i = mean gate probability for expert i
        P = gate_probs.mean(dim=0)

        aux_loss = self.num_experts * (f * P).sum()

        return output, aux_loss
