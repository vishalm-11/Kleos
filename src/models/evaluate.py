"""Metrics for the hit/flop classifier and the profit-multiple regressor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from config import PROFIT_MULTIPLE_CAP


def classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series,
) -> Dict[str, Any]:
    """Compute standard binary classification metrics.

    Parameters
    ----------
    y_true :
        0/1 hit labels.
    y_pred :
        0/1 predicted labels (thresholded).
    y_proba :
        P(hit).

    Returns
    -------
    dict
        Suggested keys: accuracy, precision, recall, f1, roc_auc, log_loss.
        Add a confusion-matrix artifact in the implementation if useful.
    """
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    proba = np.asarray(y_proba, dtype=float)
    majority_accuracy = float(max(np.mean(true == 0), np.mean(true == 1)))
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "majority_baseline_accuracy": majority_accuracy,
        "roc_auc": float(roc_auc_score(true, proba)),
        "pr_auc": float(average_precision_score(true, proba)),
        "precision": float(precision_score(true, pred, zero_division=0)),
        "recall": float(recall_score(true, pred, zero_division=0)),
        "f1": float(f1_score(true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(true, pred, labels=[0, 1]).tolist(),
        "threshold_tradeoffs": threshold_tradeoff_metrics(true, proba),
    }


def threshold_tradeoff_metrics(
    y_true: Iterable[int],
    y_proba: Iterable[float],
    thresholds: Iterable[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
) -> list[Dict[str, float]]:
    """Return precision/recall/F1 at operational probability thresholds."""
    true = np.asarray(y_true, dtype=int)
    proba = np.asarray(y_proba, dtype=float)
    rows = []
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(true, pred, zero_division=0)),
                "recall": float(recall_score(true, pred, zero_division=0)),
                "f1": float(f1_score(true, pred, zero_division=0)),
                "predicted_positive_rate": float(pred.mean()),
            }
        )
    return rows


def regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    training_median: float,
    target_cap: float = PROFIT_MULTIPLE_CAP,
) -> Dict[str, float]:
    """Compute standard regression metrics for profit multiple.

    Parameters
    ----------
    y_true :
        Actual inflation-adjusted profit multiple.
    y_pred :
        Predicted profit multiple.

    Returns
    -------
    dict
        Suggested keys: mae, rmse, r2, medae. Consider a log-space MAE as well
        because multiples are right-skewed.
    """
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    true_log = np.log1p(np.clip(true, 0, target_cap))
    pred_log = np.log1p(np.clip(pred, 0, None))
    baseline = np.full_like(true, float(training_median))
    ranks_true = pd.Series(true).rank(method="average")
    ranks_pred = pd.Series(pred).rank(method="average")
    return {
        "rmse": float(mean_squared_error(true, pred) ** 0.5),
        "r2": float(r2_score(true, pred)),
        "mae": float(mean_absolute_error(true, pred)),
        "rmse_log": float(mean_squared_error(true_log, pred_log) ** 0.5),
        "r2_log": float(r2_score(true_log, pred_log)),
        "baseline_training_median": float(training_median),
        "baseline_rmse": float(mean_squared_error(true, baseline) ** 0.5),
        "baseline_mae": float(mean_absolute_error(true, baseline)),
        "spearman": float(ranks_true.corr(ranks_pred)),
        "log_target_cap": float(target_cap),
    }


def evaluate_models(
    clf_true: pd.Series,
    clf_pred: pd.Series,
    clf_proba: pd.Series,
    reg_true: pd.Series,
    reg_pred: pd.Series,
    training_median: float,
    target_cap: float = PROFIT_MULTIPLE_CAP,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate both models on the same split (test or backtest).

    Parameters
    ----------
    clf_true, clf_pred, clf_proba :
        Classifier targets and outputs.
    reg_true, reg_pred :
        Regressor targets and outputs.

    Returns
    -------
    dict
        ``{"classifier": {...}, "regressor": {...}}``.
    """
    return {
        "classifier": classification_metrics(clf_true, clf_pred, clf_proba),
        "regressor": regression_metrics(
            reg_true,
            reg_pred,
            training_median=training_median,
            target_cap=target_cap,
        ),
    }


def write_evaluation_report(
    metrics: Dict[str, Dict[str, Any]],
    markdown_path: Path,
    json_path: Path,
) -> None:
    """Persist matching Markdown and machine-readable JSON test reports."""
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    classifier = metrics["classifier"]
    regressor = metrics["regressor"]
    matrix = classifier["confusion_matrix"]
    tradeoffs = "\n".join(
        f"| {row['threshold']:.1f} | {row['precision']:.4f} | "
        f"{row['recall']:.4f} | {row['f1']:.4f} | "
        f"{row['predicted_positive_rate']:.4f} |"
        for row in classifier["threshold_tradeoffs"]
    )
    report = f"""# Kleos test-set evaluation

## Classifier

- Accuracy: {classifier['accuracy']:.4f}
- Majority-class baseline accuracy: {classifier['majority_baseline_accuracy']:.4f}
- ROC-AUC: {classifier['roc_auc']:.4f}
- PR-AUC: {classifier['pr_auc']:.4f}
- Precision at 0.5: {classifier['precision']:.4f}
- Recall at 0.5: {classifier['recall']:.4f}
- F1 at 0.5: {classifier['f1']:.4f}

Confusion matrix (`[[TN, FP], [FN, TP]]`):

```text
{matrix}
```

| Threshold | Precision | Recall | F1 | Predicted positive rate |
|---:|---:|---:|---:|---:|
{tradeoffs}

## Regressor

- RMSE (original profit-multiple space): {regressor['rmse']:.4f}
- R² (original space): {regressor['r2']:.4f}
- MAE (original space): {regressor['mae']:.4f}
- RMSE (log1p space, actual capped at {regressor['log_target_cap']:.0f}): {regressor['rmse_log']:.4f}
- R² (log1p space): {regressor['r2_log']:.4f}
- Spearman rank correlation: {regressor['spearman']:.4f}
- Training-median baseline prediction: {regressor['baseline_training_median']:.4f}
- Baseline RMSE: {regressor['baseline_rmse']:.4f}
- Baseline MAE: {regressor['baseline_mae']:.4f}
- RMSE improvement over baseline: {regressor['baseline_rmse'] - regressor['rmse']:.4f}
- MAE improvement over baseline: {regressor['baseline_mae'] - regressor['mae']:.4f}
"""
    markdown_path.write_text(report, encoding="utf-8")
