"""XGBoost hit / flop classifier.

Target is a binary label: profit_multiple > config.HIT_PROFIT_MULTIPLE_THRESHOLD.
Do not train on the backtest year; that frame is reserved for backtest.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

from config import HIT_PROFIT_MULTIPLE_THRESHOLD


def add_hit_label(
    df: pd.DataFrame,
    profit_col: str = "profit_multiple",
    threshold: float = HIT_PROFIT_MULTIPLE_THRESHOLD,
    out_col: str = "is_hit",
) -> pd.DataFrame:
    """Add the binary hit/flop target.

    Parameters
    ----------
    df :
        Frame with ``profit_col``.
    profit_col :
        Inflation-adjusted revenue / budget.
    threshold :
        Hit if profit_multiple > threshold (default ``config.HIT_PROFIT_MULTIPLE_THRESHOLD``).
    out_col :
        Label column name.

    Returns
    -------
    pd.DataFrame
        Copy with ``out_col`` as 0/1.
    """
    raise NotImplementedError("is_hit = (profit_multiple > threshold).astype(int)")


def train_classifier(
    train_df: pd.DataFrame,
    feature_cols: Iterable[str],
    label_col: str = "is_hit",
    params: Optional[dict] = None,
):
    """Fit an XGBoost classifier on the training split.

    Parameters
    ----------
    train_df :
        Training rows only (no test, no backtest year).
    feature_cols :
        Columns to use as X. Exclude targets, raw ids, and post-release leaks
        (nominal revenue, unadjusted dollars if adj columns exist, etc.).
    label_col :
        Binary hit/flop column.
    params :
        Optional XGBoost hyperparameter dict.

    Returns
    -------
    xgboost.XGBClassifier
        Fitted model. Persist path is the caller's concern.
    """
    raise NotImplementedError("Fit XGBClassifier on train_df[feature_cols] -> label_col.")


def predict_hit_proba(
    model,
    df: pd.DataFrame,
    feature_cols: Iterable[str],
) -> pd.Series:
    """Return P(hit) for each row.

    Parameters
    ----------
    model :
        Fitted XGBClassifier.
    df :
        Feature frame (test, backtest, or a single inference row).
    feature_cols :
        Same columns, same order as training.

    Returns
    -------
    pd.Series
        Hit probabilities aligned to ``df.index``.
    """
    raise NotImplementedError("Return model.predict_proba(...)[:, 1] as a Series.")


def save_classifier(model, path: Union[str, Path]) -> None:
    """Serialize the classifier to disk."""
    raise NotImplementedError("Save the XGBoost classifier.")


def load_classifier(path: Union[str, Path]):
    """Load a previously saved classifier."""
    raise NotImplementedError("Load the XGBoost classifier.")
