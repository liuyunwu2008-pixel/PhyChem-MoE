"""Optimizer and learning rate scheduler construction."""

from typing import Tuple
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    total_steps: int = 50000,
) -> Tuple[AdamW, LambdaLR]:
    """Build AdamW optimizer with cosine LR schedule & linear warmup.

    Returns:
        (optimizer, scheduler) tuple.
    """
    # Separate parameters for weight decay (don't decay biases & norms)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler
