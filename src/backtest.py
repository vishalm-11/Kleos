"""Held-out evaluation on the most recent calendar year of real releases.

The backtest frame must never appear in training, hyperparameter search, or
median / wide-release threshold fitting. ``pipeline.split_train_test_backtest``
is the only supported way to obtain it.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import pandas as pd


def run_backtest(
    classifier,
    regressor,
    backtest_df: pd.DataFrame,
    feature_cols: Iterable[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Score both models on the held-out year and return predictions + metrics.

    Parameters
    ----------
    classifier :
        Fitted XGBClassifier from ``models.classifier``.
    regressor :
        Fitted XGBRegressor from ``models.regressor``.
    backtest_df :
        Rows whose release year is the most recent year in the dataset.
        Must not have been used to fit models or imputation constants.
    feature_cols :
        Training feature list, same order.

    Returns
    -------
    predictions, metrics : tuple
        ``predictions`` has at least title, release_date, y_true / y_pred for
        both tasks, and hit probability.
        ``metrics`` is the dict from ``models.evaluate.evaluate_models``.
    """
    raise NotImplementedError(
        "Predict on backtest_df, attach actuals, call evaluate_models."
    )


def compare_to_test_split(
    backtest_metrics: Dict[str, Dict[str, float]],
    test_metrics: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Side-by-side test vs backtest metrics (distribution shift check).

    Parameters
    ----------
    backtest_metrics, test_metrics :
        Outputs of ``evaluate_models`` on each split.

    Returns
    -------
    pd.DataFrame
        Metric name × {test, backtest, delta} for classifier and regressor.
    """
    raise NotImplementedError("Build a comparison table of test vs backtest metrics.")
