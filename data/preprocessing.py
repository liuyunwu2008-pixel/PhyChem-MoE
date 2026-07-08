"""Label standardization for regression targets."""

from typing import Dict, Optional, Tuple
import numpy as np


class LabelStandardizer:
    """Z-score standardization: y_norm = (y - mu) / sigma."""

    def __init__(self):
        self.stats: Dict[str, Tuple[float, float]] = {}

    def fit(self, values: np.ndarray, task_name: str) -> None:
        mu = float(np.mean(values))
        sigma = float(np.std(values))
        if sigma < 1e-8:
            sigma = 1.0
        self.stats[task_name] = (mu, sigma)

    def transform(self, values: np.ndarray, task_name: str) -> np.ndarray:
        mu, sigma = self.stats[task_name]
        return (values - mu) / sigma

    def inverse_transform(self, values: np.ndarray, task_name: str) -> np.ndarray:
        mu, sigma = self.stats[task_name]
        return values * sigma + mu

    def fit_transform(self, values: np.ndarray, task_name: str) -> np.ndarray:
        self.fit(values, task_name)
        return self.transform(values, task_name)

    def get_stats(self, task_name: str) -> Optional[Tuple[float, float]]:
        return self.stats.get(task_name)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {k: {"mu": mu, "sigma": sigma} for k, (mu, sigma) in self.stats.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, float]]) -> "LabelStandardizer":
        std = cls()
        std.stats = {k: (v["mu"], v["sigma"]) for k, v in data.items()}
        return std
