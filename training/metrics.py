"""Evaluation metrics: ROC-AUC for classification, MAE/RMSE for regression."""

from typing import Dict
import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error


def compute_metrics(
    predictions: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    task_types: Dict[str, str],
) -> Dict[str, float]:
    """Compute per-dataset evaluation metrics.

    Args:
        predictions: {task_name: np.ndarray (N, *)} raw model outputs.
        labels: {task_name: np.ndarray (N, *)} ground truth.
        task_types: {task_name: "classification" | "regression"}.

    Returns:
        Dict of metric_name → value.
    """
    metrics = {}

    for name in predictions:
        if name not in labels or name.startswith("_"):
            continue

        pred = predictions[name]
        true = labels[name]
        task_type = task_types.get(name, "regression")

        if task_type == "classification":
            try:
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                if true.ndim == 1:
                    true = true.reshape(-1, 1)

                # Per-task AUC, then macro average
                aucs = []
                for c in range(true.shape[1]):
                    if len(np.unique(true[:, c])) >= 2:
                        aucs.append(roc_auc_score(true[:, c], pred[:, c]))
                metrics[f"{name}_auc"] = float(np.mean(aucs)) if aucs else 0.5
            except Exception:
                metrics[f"{name}_auc"] = 0.5
        else:
            pred_flat = pred.reshape(-1)
            true_flat = true.reshape(-1)
            mask = ~np.isnan(true_flat)
            if mask.sum() > 0:
                metrics[f"{name}_mae"] = mean_absolute_error(true_flat[mask], pred_flat[mask])
                metrics[f"{name}_rmse"] = np.sqrt(mean_squared_error(true_flat[mask], pred_flat[mask]))
            else:
                metrics[f"{name}_mae"] = float("nan")
                metrics[f"{name}_rmse"] = float("nan")

    return metrics


def aggregate_metrics(all_metrics: Dict[str, float], task_types: Dict[str, str]) -> Dict[str, float]:
    """Compute aggregated metrics across classification and regression tasks."""
    agg = {}
    clf_aucs = [v for k, v in all_metrics.items() if k.endswith("_auc") and not np.isnan(v)]
    reg_maes = [v for k, v in all_metrics.items() if k.endswith("_mae") and not np.isnan(v)]
    reg_rmses = [v for k, v in all_metrics.items() if k.endswith("_rmse") and not np.isnan(v)]

    if clf_aucs:
        agg["avg_auc"] = float(np.mean(clf_aucs))
    if reg_maes:
        agg["avg_mae"] = float(np.mean(reg_maes))
    if reg_rmses:
        agg["avg_rmse"] = float(np.mean(reg_rmses))

    return agg
