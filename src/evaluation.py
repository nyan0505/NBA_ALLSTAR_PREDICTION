"""
evaluation.py – Ranking-oriented evaluation metrics for the All-Star pipeline.

Metrics implemented
-------------------
* PR-AUC  (Average Precision)
* ROC-AUC
* Precision@K and Recall@K  (default K = 24, optionally K = 12)
* Confusion matrix at a given decision threshold

All public functions accept numpy arrays / pandas Series and return plain
Python scalars / dicts so they are easy to print in notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


def precision_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int = 24) -> float:
    """Fraction of the top-K predicted players that are actual All-Stars."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    top_k_idx = np.argsort(y_proba)[::-1][:k]
    return float(y_true[top_k_idx].sum() / k)


def recall_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int = 24) -> float:
    """Fraction of actual All-Stars that appear in the top-K predictions."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    total_positives = y_true.sum()
    if total_positives == 0:
        return 0.0
    top_k_idx = np.argsort(y_proba)[::-1][:k]
    return float(y_true[top_k_idx].sum() / total_positives)


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    threshold: float = 0.5,
    k_values: list[int] | None = None,
) -> dict:
    """Compute a full suite of evaluation metrics.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth binary labels (0 / 1).
    y_proba : array-like of float
        Predicted probabilities for the positive class.
    threshold : float
        Decision threshold for the confusion matrix.
    k_values : list[int] | None
        K values for Precision@K / Recall@K.  Defaults to ``[24, 12]``.

    Returns
    -------
    dict
        Dictionary with keys: ``pr_auc``, ``roc_auc``,
        ``precision_at_<k>``, ``recall_at_<k>`` for each k,
        ``confusion_matrix``.
    """
    if k_values is None:
        k_values = [24, 12]

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    results: dict = {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    for k in k_values:
        results[f"precision_at_{k}"] = precision_at_k(y_true, y_proba, k=k)
        results[f"recall_at_{k}"] = recall_at_k(y_true, y_proba, k=k)

    return results


def print_evaluation(results: dict) -> None:
    """Pretty-print an evaluation dict returned by :func:`evaluate`."""
    print(f"  PR-AUC (Average Precision) : {results['pr_auc']:.4f}")
    print(f"  ROC-AUC                    : {results['roc_auc']:.4f}")
    for key, val in results.items():
        if key.startswith("precision_at_"):
            k = key.split("_")[-1]
            print(f"  Precision@{k:<3}               : {val:.4f}")
        if key.startswith("recall_at_"):
            k = key.split("_")[-1]
            print(f"  Recall@{k:<3}                  : {val:.4f}")
    print("  Confusion matrix (rows=actual, cols=pred):")
    for row in results["confusion_matrix"]:
        print(f"    {row}")
