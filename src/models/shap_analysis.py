"""SHAP explanations for both the classifier and the regressor.

Use TreeExplainer (XGBoost). Keep plots optional so this module can run
headless in backtest.py and with Streamlit in app/streamlit_app.py.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "kleos-matplotlib"),
)

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from config import (
    CLASSIFIER_EXPLAINER_PATH,
    REGRESSOR_EXPLAINER_PATH,
    REPORTS_DIR,
)


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
    columns = list(feature_cols)
    explainer = shap.TreeExplainer(model)
    return explainer(df.loc[:, columns])


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
    columns = list(feature_cols)
    explainer = shap.TreeExplainer(model)
    return explainer(df.loc[:, columns])


def summary_plot(
    shap_values,
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
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
    columns = list(feature_cols)
    plt.figure()
    shap.summary_plot(
        _values_array(shap_values),
        df.loc[:, columns],
        feature_names=columns,
        show=False,
        max_display=20,
    )
    figure = plt.gcf()
    if title:
        figure.axes[0].set_title(title)
    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    return figure


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
    columns = list(feature_cols)
    position = df.index.get_loc(row_index)
    values = _values_array(shap_values)[position]
    result = pd.DataFrame(
        {
            "feature": columns,
            "feature_value": df.loc[row_index, columns].to_numpy(),
            "shap_value": values,
        }
    )
    return result.reindex(
        result["shap_value"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def _values_array(explanation) -> np.ndarray:
    values = np.asarray(
        explanation.values if hasattr(explanation, "values") else explanation
    )
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.ndim != 2:
        raise ValueError(f"Expected two-dimensional SHAP values, got {values.shape}")
    return values


def mean_absolute_importance(
    explanation,
    feature_cols: Iterable[str],
) -> pd.DataFrame:
    """Return mean absolute SHAP contribution per feature, descending."""
    columns = list(feature_cols)
    values = _values_array(explanation)
    if values.shape[1] != len(columns):
        raise ValueError("SHAP feature dimension does not match feature columns")
    return (
        pd.DataFrame(
            {
                "feature": columns,
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def analyze_and_save_shap(
    classifier,
    regressor,
    X_test: pd.DataFrame,
    feature_cols: Iterable[str],
    reports_dir: Path = REPORTS_DIR,
    classifier_explainer_path: Path = CLASSIFIER_EXPLAINER_PATH,
    regressor_explainer_path: Path = REGRESSOR_EXPLAINER_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute test SHAP values and persist plots, rankings, and explainers."""
    columns = list(feature_cols)
    classifier_explainer = shap.TreeExplainer(classifier)
    regressor_explainer = shap.TreeExplainer(regressor)
    classifier_values = classifier_explainer(X_test.loc[:, columns])
    regressor_values = regressor_explainer(X_test.loc[:, columns])

    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_plot(
        classifier_values,
        X_test,
        columns,
        title="Kleos hit classifier — test SHAP",
        output_path=reports_dir / "shap_classifier.png",
    )
    plt.close()
    summary_plot(
        regressor_values,
        X_test,
        columns,
        title="Kleos profit-multiple regressor — test SHAP",
        output_path=reports_dir / "shap_regressor.png",
    )
    plt.close()

    classifier_importance = mean_absolute_importance(
        classifier_values,
        columns,
    )
    regressor_importance = mean_absolute_importance(
        regressor_values,
        columns,
    )
    classifier_importance.to_csv(
        reports_dir / "shap_importance_cls.csv",
        index=False,
    )
    regressor_importance.to_csv(
        reports_dir / "shap_importance_reg.csv",
        index=False,
    )

    classifier_explainer_path.parent.mkdir(parents=True, exist_ok=True)
    regressor_explainer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier_explainer, classifier_explainer_path)
    joblib.dump(regressor_explainer, regressor_explainer_path)

    print("\nTop 10 classifier features by mean |SHAP|:")
    print(classifier_importance.head(10).to_string(index=False))
    print("\nTop 10 regressor features by mean |SHAP|:")
    print(regressor_importance.head(10).to_string(index=False))
    return classifier_importance, regressor_importance
