"""XGBoost regressor for inflation-adjusted profit multiple (revenue_adj / budget_adj).

Train on the training split only. The most recent year is reserved for backtest.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

from config import PROFIT_MULTIPLE_CAP, RANDOM_SEED

REGRESSOR_PARAM_DISTRIBUTIONS = {
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.02, 0.04, 0.06, 0.1, 0.15],
    "n_estimators": [200, 350, 500, 700, 900],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.75, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 8, 12],
}

def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[dict] = None,
    *,
    search_trials: int = 25,
    cv_folds: int = 5,
    validation_fraction: float = 0.15,
    early_stopping_rounds: int = 30,
    target_cap: float = PROFIT_MULTIPLE_CAP,
    random_state: int = RANDOM_SEED,
) -> XGBRegressor:
    """Fit an XGBoost regressor on the training split.

    Parameters
    ----------
    X_train, y_train :
        Training split only. Targets are clipped at ``target_cap`` and then
        transformed with ``log1p``.
    params :
        Optional XGBoost hyperparameter dict.

    Returns
    -------
    xgboost.XGBRegressor
        Fitted model.
    """
    raw_y = pd.to_numeric(pd.Series(y_train), errors="coerce").reset_index(drop=True)
    if raw_y.isna().any() or not np.isfinite(raw_y).all() or (raw_y < 0).any():
        raise ValueError("Regression targets must be finite and non-negative")
    X = X_train.reset_index(drop=True)
    log_y = np.log1p(raw_y.clip(upper=target_cap))
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X,
        log_y,
        test_size=validation_fraction,
        random_state=random_state,
    )

    fixed = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": 1,
    }
    selected = dict(params or {})
    if search_trials > 0:
        search = RandomizedSearchCV(
            estimator=XGBRegressor(**fixed),
            param_distributions=REGRESSOR_PARAM_DISTRIBUTIONS,
            n_iter=search_trials,
            scoring="neg_root_mean_squared_error",
            cv=KFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=random_state,
            ),
            random_state=random_state,
            n_jobs=1,
            refit=False,
            verbose=1,
        )
        search.fit(X_fit, y_fit)
        selected.update(search.best_params_)
    if "n_estimators" not in selected:
        selected["n_estimators"] = 500

    model = XGBRegressor(
        **fixed,
        **selected,
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )
    model.kleos_best_params_ = {
        **selected,
        "target_cap": float(target_cap),
        "best_iteration": int(model.best_iteration),
    }
    model.kleos_training_median_ = float(raw_y.median())
    return model


def predict_profit_multiple(
    model,
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
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
    log_predictions = predict_log_profit_multiple(model, df, feature_cols)
    return pd.Series(
        np.expm1(log_predictions),
        index=df.index,
        name="predicted_profit_multiple",
    )


def predict_log_profit_multiple(
    model,
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> pd.Series:
    """Return model predictions in the fitted ``log1p`` target space."""
    X = df.loc[:, feature_cols] if feature_cols is not None else df
    return pd.Series(model.predict(X), index=df.index, name="predicted_log_profit_multiple")


def save_regressor(model, path: Union[str, Path]) -> None:
    """Serialize the regressor to disk."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)


def load_regressor(path: Union[str, Path]):
    """Load a previously saved regressor."""
    return joblib.load(path)


def save_regressor_params(model: XGBRegressor, path: Union[str, Path]) -> None:
    """Write selected hyperparameters and target metadata as JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = getattr(model, "kleos_best_params_", {})
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
