"""XGBoost hit / flop classifier.

Target is a binary label: profit_multiple >= config.HIT_PROFIT_MULTIPLE_THRESHOLD.
Do not train on the backtest year; that frame is reserved for backtest.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from config import HIT_PROFIT_MULTIPLE_THRESHOLD, RANDOM_SEED

CLASSIFIER_PARAM_DISTRIBUTIONS = {
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.02, 0.04, 0.06, 0.1, 0.15],
    "n_estimators": [200, 350, 500, 700, 900],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.75, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 8, 12],
}


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
        Hit if profit_multiple >= threshold (default ``config.HIT_PROFIT_MULTIPLE_THRESHOLD``).
    out_col :
        Label column name.

    Returns
    -------
    pd.DataFrame
        Copy with ``out_col`` as 0/1.
    """
    result = df.copy()
    result[out_col] = (result[profit_col] >= threshold).astype("int8")
    return result


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[dict] = None,
    *,
    search_trials: int = 25,
    cv_folds: int = 5,
    validation_fraction: float = 0.15,
    early_stopping_rounds: int = 30,
    random_state: int = RANDOM_SEED,
) -> XGBClassifier:
    """Fit an XGBoost classifier on the training split.

    Parameters
    ----------
    X_train, y_train :
        Training split only. The test/backtest matrices must not be passed.
    params :
        Optional XGBoost hyperparameter dict.

    Returns
    -------
    xgboost.XGBClassifier
        Fitted model. Persist path is the caller's concern.
    """
    y = pd.Series(y_train).astype(int).reset_index(drop=True)
    X = X_train.reset_index(drop=True)
    counts = y.value_counts()
    if set(counts.index) != {0, 1}:
        raise ValueError("Classifier training labels must contain both 0 and 1")
    scale_pos_weight = float(counts.loc[0] / counts.loc[1])

    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X,
        y,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=y,
    )
    fixed = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": 1,
        "scale_pos_weight": scale_pos_weight,
    }
    selected = dict(params or {})
    if search_trials > 0:
        search = RandomizedSearchCV(
            estimator=XGBClassifier(**fixed),
            param_distributions=CLASSIFIER_PARAM_DISTRIBUTIONS,
            n_iter=search_trials,
            scoring="roc_auc",
            cv=StratifiedKFold(
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

    model = XGBClassifier(
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
        "scale_pos_weight": scale_pos_weight,
        "best_iteration": int(model.best_iteration),
    }
    return model


def predict_hit_proba(
    model,
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
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
    X = df.loc[:, feature_cols] if feature_cols is not None else df
    return pd.Series(model.predict_proba(X)[:, 1], index=df.index, name="hit_probability")


def save_classifier(model, path: Union[str, Path]) -> None:
    """Serialize the classifier to disk."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)


def load_classifier(path: Union[str, Path]):
    """Load a previously saved classifier."""
    return joblib.load(path)


def save_classifier_params(model: XGBClassifier, path: Union[str, Path]) -> None:
    """Write the selected hyperparameters and imbalance weight as JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = getattr(model, "kleos_best_params_", {})
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
