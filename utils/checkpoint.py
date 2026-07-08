"""Model checkpoint save/load utilities."""

import gc
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    global_step: int,
    metrics: Dict[str, float],
    path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, path)
    gc.collect()


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "metrics": checkpoint.get("metrics", {}),
    }


def save_best_checkpoint(
    model: nn.Module,
    metric_value: float,
    current_best: float,
    mode: str = "min",
    path: str = "./outputs/best.pt",
) -> float:
    is_better = (mode == "min" and metric_value < current_best) or \
                (mode == "max" and metric_value > current_best)
    if is_better:
        shutil.copy(path, str(Path(path).parent / "best.pt"))
        return metric_value
    return current_best
