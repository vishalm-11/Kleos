"""XGBoost regressor for inflation-adjusted profit multiple (revenue_adj / budget_adj).

Train on the training split only. The most recent year is reserved for backtest.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd


def train_regressor(
    train_df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str = "profit_multiple",
    params: Optional[dict] = None,
):
    """Fit an XGBoost regressor on the training split.

    Parameters
    ----------
    train_df :
        Training rows only.
    feature_cols :
        Predictor columns. Must not include revenue, profit_multiple, or is_hit.
    target_col :
        Inflation-adjusted profit multiple.
    params :
        Optional XGBoost hyperparameter dict.

    Returns
    -------
    xgboost.XGBRegressor
        Fitted model.
    """
    raise NotImplementedError("Fit XGBRegressor on train_df[feature_cols] -> target_col.")


def predict_profit_multiple(
    model,
    df: pd.DataFrame,
    feature_cols: Iterable[str],
) -> pd.Series:
    """Return predicted profit multiples.

    Parameters
    ----------
    model :
        Fitted XGBRegressor.
    df :
        Feature frame.
    feature_cols :
        Same columns, same order as training.

    Returns
    -------
    pd.Series
        Predictions aligned to ``df.index``.
    """
    raise NotImplementedError("Return model.predict(...) as a Series.")


def save_regressor(model, path: Union[str, Path]) -> None:
    """Serialize the regressor to disk."""
    raise NotImplementedError("Save the XGBoost regressor.")


def load_regressor(path: Union[str, Path]):
    """Load a previously saved regressor."""
    raise NotImplementedError("Load the XGBoost regressor.")
