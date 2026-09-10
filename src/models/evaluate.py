"""Metrics for the hit/flop classifier and the profit-multiple regressor."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series,
) -> Dict[str, float]:
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
    raise NotImplementedError("Return a dict of sklearn classification metrics.")


def regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
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
    raise NotImplementedError("Return a dict of sklearn regression metrics.")


def evaluate_models(
    clf_true: pd.Series,
    clf_pred: pd.Series,
    clf_proba: pd.Series,
    reg_true: pd.Series,
    reg_pred: pd.Series,
) -> Dict[str, Dict[str, float]]:
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
    raise NotImplementedError("Compose classification_metrics and regression_metrics.")
