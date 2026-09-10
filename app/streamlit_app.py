"""Streamlit front end: pre-release inputs, dual predictions, SHAP explanation.

Do not implement the full UI yet. Wire these stubs once the models exist.
Inference must use the same feature builders as ``src.pipeline.build_features``
so the app cannot drift from training.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def render_input_form() -> Dict[str, Any]:
    """Collect pre-release fields from the user.

    Expected inputs (names are suggestions; match pipeline columns when implementing):
    budget, runtime, release date, genres, top-billed cast, director,
    production companies, belongs-to-collection flag.

    Returns
    -------
    dict
        Raw form values, not yet feature-engineered.
    """
    raise NotImplementedError("st.form with the pre-release feature inputs.")


def features_from_form(raw: Dict[str, Any]) -> pd.DataFrame:
    """Turn form values into a one-row feature frame the models can score.

    Parameters
    ----------
    raw :
        Output of ``render_input_form``.

    Returns
    -------
    pd.DataFrame
        Single row, same columns/order as training ``feature_cols``.
        Star-power and studio encodings must use the *training-fitted*
        vocabularies and median, not re-fit on this row.
    """
    raise NotImplementedError("Map form fields through the same feature builders as the pipeline.")


def render_predictions(hit_proba: float, profit_multiple: float) -> None:
    """Show hit/flop probability and predicted profit multiple.

    Parameters
    ----------
    hit_proba :
        P(hit) from the classifier.
    profit_multiple :
        Predicted inflation-adjusted revenue / budget.
    """
    raise NotImplementedError("Display the two model outputs.")


def render_shap_explanation(local_table: pd.DataFrame) -> None:
    """Show a local SHAP breakdown for the submitted title.

    Parameters
    ----------
    local_table :
        Output of ``src.models.shap_analysis.local_explanation``.
    """
    raise NotImplementedError("Plot or table the local SHAP values.")


def main() -> None:
    """App entry point: form → features → both models → SHAP.

    Run later with: ``streamlit run app/streamlit_app.py``
    """
    raise NotImplementedError("Compose form, predict, and explanation panes.")


if __name__ == "__main__":
    main()
