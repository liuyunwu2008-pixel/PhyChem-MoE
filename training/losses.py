"""Loss functions: uncertainty-weighted multi-task loss."""

from typing import Dict, List
import torch
import torch.nn as nn


class UncertaintyWeightedLoss(nn.Module):
    """Multi-task uncertainty weighting following Kendall et al. (CVPR 2018).

    Regression:  L_reg / (2σ²) + log σ     (Gaussian likelihood)
    Classification: L_cls / σ² + log σ     (Boltzmann likelihood)

    The factor-of-2 difference comes from the derivation: Gaussian log-likelihood
    for regression yields 1/(2σ²) * MSE; softmax-with-temperature for classification
    yields 1/σ² * CE.
    """

    def __init__(self, task_names: List[str], regression_tasks: List[str]):
        super().__init__()
        self.task_names = task_names
        self.regression_tasks = set(regression_tasks)
        self.log_var = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1))
            for name in task_names
        })

    def forward(self, task_losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        weighted = {}
        total = 0.0
        for name in self.task_names:
            if name in task_losses:
                precision = torch.exp(-self.log_var[name])
                if name in self.regression_tasks:
                    weighted[name] = 0.5 * precision * task_losses[name] + self.log_var[name]
                else:
                    weighted[name] = precision * task_losses[name] + self.log_var[name]
                total = total + weighted[name]
        # Pass through auxiliary losses that don't need uncertainty weighting
        for aux_key in ("_align", "_aux", "_mph_dispersion"):
            if aux_key in task_losses:
                weighted[aux_key] = task_losses[aux_key]
                total = total + task_losses[aux_key]
        weighted["_total"] = total
        return weighted
