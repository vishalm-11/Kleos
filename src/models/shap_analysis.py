"""SHAP explanations for both the classifier and the regressor.

Use TreeExplainer (XGBoost). Keep plots optional so this module can run
headless in backtest.py and with Streamlit in app/streamlit_app.py.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def shap_values_classifier(
    model,
    df: pd.DataFrame,
    feature_cols: Iterable[str],
):
    """Compute SHAP values for the hit/flop model.

    Parameters
    ----------
    model :
        Fitted XGBClassifier.
    df :
        Feature rows to explain.
    feature_cols :
        Columns in training order.

    Returns
    -------
    shap.Explanation or ndarray
        SHAP values for the positive (hit) class. Document the exact type
        when implementing (Explanation vs raw array).
    """
    raise NotImplementedError("TreeExplainer(model).shap_values on the classifier.")


def shap_values_regressor(
    model,
    df: pd.DataFrame,
    feature_cols: Iterable[str],
):
    """Compute SHAP values for the profit-multiple model.

    Parameters
    ----------
    model :
        Fitted XGBRegressor.
    df :
        Feature rows to explain.
    feature_cols :
        Columns in training order.

    Returns
    -------
    shap.Explanation or ndarray
        Per-row, per-feature contributions to the predicted multiple.
    """
    raise NotImplementedError("TreeExplainer(model).shap_values on the regressor.")


def summary_plot(
    shap_values,
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    title: Optional[str] = None,
) -> None:
    """Global SHAP summary (beeswarm / bar) for a model.

    Parameters
    ----------
    shap_values :
        Output of one of the shap_values_* helpers.
    df :
        Feature frame used to compute those values.
    feature_cols :
        Feature names for the plot.
    title :
        Optional figure title.

    Returns
    -------
    None
        Displays or saves a matplotlib figure. Prefer returning the figure
        handle if Streamlit will call this.
    """
    raise NotImplementedError("shap.summary_plot (or equivalent) via matplotlib.")


def local_explanation(
    shap_values,
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    row_index: object,
):
    """Per-title SHAP breakdown for the Streamlit explanation pane.

    Parameters
    ----------
    shap_values :
        Precomputed values aligned to ``df``.
    df :
        Feature frame.
    feature_cols :
        Feature names.
    row_index :
        Index label of the movie to explain.

    Returns
    -------
    pd.DataFrame
        Feature name, feature value, SHAP value — sorted by absolute impact.
    """
    raise NotImplementedError("Return a tidy local-explanation table for one row.")
