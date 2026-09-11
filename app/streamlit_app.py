"""Interactive pre-release inference with the persisted Kleos pipeline."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (  # noqa: E402
    CLASSIFIER_EXPLAINER_PATH,
    CLASSIFIER_PATH,
    CLASSIFIER_THRESHOLD_PATH,
    ENCODERS_PATH,
    INFLATION_BASE_YEAR,
    MOVIES_WITH_CREDITS_PATH,
    PROCESSED_DATA_DIR,
    REGRESSOR_EXPLAINER_PATH,
    REGRESSOR_PATH,
)
from src.inflation import cpi_adjust, load_cpi_index  # noqa: E402
from src.models.classifier import predict_hit_proba  # noqa: E402
from src.models.regressor import predict_profit_multiple  # noqa: E402
from src.pipeline import build_features  # noqa: E402

MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def _people_lookup(values: Iterable[object]) -> tuple[dict[str, tuple[int, str]], list[str]]:
    """Build a case-insensitive name→(id, display name) lookup."""
    lookup: dict[str, tuple[int, str]] = {}
    for raw_people in values:
        if not isinstance(raw_people, (list, tuple, np.ndarray)):
            continue
        for person in raw_people:
            if not isinstance(person, dict):
                continue
            person_id = person.get("id")
            name = str(person.get("name") or "").strip()
            if person_id is None or not name:
                continue
            lookup.setdefault(name.casefold(), (int(person_id), name))
    names = sorted((value[1] for value in lookup.values()), key=str.casefold)
    return lookup, names


@st.cache_resource(show_spinner="Loading trained models and feature history…")
def load_artifacts() -> Dict[str, Any]:
    """Load immutable inference artifacts once per Streamlit process."""
    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    classifier_explainer = joblib.load(CLASSIFIER_EXPLAINER_PATH)
    regressor_explainer = joblib.load(REGRESSOR_EXPLAINER_PATH)
    with CLASSIFIER_THRESHOLD_PATH.open(encoding="utf-8") as handle:
        threshold = float(json.load(handle)["threshold"])
    with (PROCESSED_DATA_DIR / "feature_columns.json").open(
        encoding="utf-8"
    ) as handle:
        feature_columns = json.load(handle)

    enriched = pd.read_parquet(
        MOVIES_WITH_CREDITS_PATH,
        columns=["cast", "directors"],
    )
    actor_lookup, actor_names = _people_lookup(enriched["cast"])
    director_lookup, director_names = _people_lookup(enriched["directors"])
    return {
        "classifier": classifier,
        "regressor": regressor,
        "encoders": encoders,
        "classifier_explainer": classifier_explainer,
        "regressor_explainer": regressor_explainer,
        "threshold": threshold,
        "feature_columns": feature_columns,
        "actor_lookup": actor_lookup,
        "actor_names": actor_names,
        "director_lookup": director_lookup,
        "director_names": director_names,
        "default_competitors": int(
            round(encoders.get("default_competitors_2wk", 4))
        ),
    }


def _synthetic_person_id(kind: str, name: str) -> int:
    """Create a stable negative ID that cannot collide with TMDB IDs."""
    digest = hashlib.sha1(f"{kind}:{name.casefold()}".encode()).hexdigest()[:12]
    return -int(digest, 16)


def _resolve_cast(
    names: Iterable[str],
    lookup: dict[str, tuple[int, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    cast = []
    unknown = []
    for order, entered in enumerate(names):
        name = entered.strip()
        if not name:
            continue
        match = lookup.get(name.casefold())
        if match is None:
            person_id, display = _synthetic_person_id("actor", name), name
            unknown.append(name)
        else:
            person_id, display = match
        cast.append({"id": person_id, "name": display, "order": order})
    return cast, unknown


def _resolve_director(
    name: str,
    lookup: dict[str, tuple[int, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    name = name.strip()
    if not name:
        return [], []
    match = lookup.get(name.casefold())
    if match is None:
        return [{"id": _synthetic_person_id("director", name), "name": name}], [name]
    person_id, display = match
    return [{"id": person_id, "name": display}], []


def _adjust_budget(budget: float, release_year: int) -> tuple[float, bool]:
    """Convert known-year nominal dollars; treat future input as base-year dollars."""
    cpi = load_cpi_index()
    if release_year in cpi.index:
        return float(cpi_adjust(budget, release_year, cpi)), False
    if release_year > int(cpi.index.max()):
        return float(budget), True
    raise ValueError(
        f"Release year must be at least {int(cpi.index.min())}; "
        f"future budgets are entered in {INFLATION_BASE_YEAR} dollars."
    )


def render_input_form() -> Dict[str, Any]:
    """Render the pre-release form and return its raw values when submitted."""
    artifacts = load_artifacts()
    encoders = artifacts["encoders"]
    with st.form("movie"):
        left, right = st.columns(2)
        with left:
            budget = st.number_input(
                "Production budget (USD)",
                min_value=1_000.0,
                value=50_000_000.0,
                step=1_000_000.0,
                format="%.0f",
            )
            runtime = st.number_input(
                "Runtime (minutes)",
                min_value=1,
                max_value=500,
                value=110,
            )
            release_month_name = st.selectbox(
                "Release month",
                list(MONTHS),
                index=5,
            )
            release_year = st.number_input(
                "Release year",
                min_value=1915,
                max_value=datetime.now().year + 10,
                value=datetime.now().year,
                step=1,
            )
            genres = st.multiselect(
                "Genres",
                encoders["genre"].vocabulary_,
                default=["Drama"]
                if "Drama" in encoders["genre"].vocabulary_
                else [],
            )
            is_franchise = st.checkbox("Part of a franchise or collection?")
        with right:
            studio_options = [
                studio
                for studio in encoders["studio"].top_studios_
                if str(studio).casefold() != "nan"
            ] + ["Other"]
            studio = st.selectbox("Primary studio", studio_options)
            cast_text = st.text_area(
                "Cast, comma separated in billing order",
                placeholder="Actor One, Actor Two, Actor Three",
            )
            director = st.text_input("Director")
            competition = st.number_input(
                "Expected wide releases within ±2 weeks",
                min_value=0,
                value=artifacts["default_competitors"],
                help="Leave at the dataset median if the release calendar is unknown.",
            )
        submitted = st.form_submit_button("Predict", type="primary")
    if not submitted:
        return {}
    return {
        "budget": float(budget),
        "runtime": int(runtime),
        "release_month": MONTHS[release_month_name],
        "release_year": int(release_year),
        "genres": genres,
        "is_franchise": bool(is_franchise),
        "studio": studio,
        "cast_names": [name.strip() for name in cast_text.split(",") if name.strip()],
        "director_name": director.strip(),
        "n_competitors_2wk": int(competition),
    }


def features_from_form(
    raw: Dict[str, Any],
    artifacts: Dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Transform one movie through persisted encoders without fitting anything."""
    artifacts = artifacts or load_artifacts()
    budget_adj, future_budget_assumption = _adjust_budget(
        float(raw["budget"]),
        int(raw["release_year"]),
    )
    cast, unknown_actors = _resolve_cast(
        raw.get("cast_names", []),
        artifacts["actor_lookup"],
    )
    directors, unknown_directors = _resolve_director(
        raw.get("director_name", ""),
        artifacts["director_lookup"],
    )
    studio = raw.get("studio") or "Other"
    input_frame = pd.DataFrame(
        [
            {
                "id": _synthetic_person_id("movie", "interactive prediction"),
                "release_date": pd.Timestamp(
                    int(raw["release_year"]),
                    int(raw["release_month"]),
                    15,
                ),
                "budget_adj": budget_adj,
                "runtime": raw["runtime"],
                "genres": ", ".join(raw.get("genres", [])),
                "production_companies": (
                    "__unseen_studio__" if studio == "Other" else studio
                ),
                "belongs_to_collection": (
                    {"id": -1, "name": "User-entered franchise"}
                    if raw.get("is_franchise")
                    else None
                ),
                "cast": cast,
                "directors": directors,
            }
        ]
    )
    engineered = build_features(input_frame, artifacts["encoders"])
    engineered["n_competitors_2wk"] = int(
        raw.get("n_competitors_2wk", artifacts["default_competitors"])
    )
    expected = list(artifacts["feature_columns"])
    missing = set(expected).difference(engineered.columns)
    if missing:
        raise AssertionError(f"Inference row is missing features: {sorted(missing)}")
    X = engineered.loc[:, expected]
    if list(X.columns) != expected or X.shape != (1, len(expected)):
        raise AssertionError("Inference features do not match feature_columns.json")
    if X.isna().any().any():
        raise ValueError("Inference feature row contains null values")
    return X, {
        "unknown_actors": unknown_actors,
        "unknown_directors": unknown_directors,
        "future_budget_assumption": future_budget_assumption,
    }


def predict_movie(
    raw: Dict[str, Any],
    artifacts: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run the complete local input→features→dual-prediction path."""
    artifacts = artifacts or load_artifacts()
    X, notes = features_from_form(raw, artifacts)
    hit_probability = float(
        predict_hit_proba(artifacts["classifier"], X).iloc[0]
    )
    profit_multiple = float(
        predict_profit_multiple(artifacts["regressor"], X).iloc[0]
    )
    threshold = float(artifacts["threshold"])
    return {
        "X": X,
        "hit_probability": hit_probability,
        "profit_multiple": profit_multiple,
        "threshold": threshold,
        "is_hit": hit_probability >= threshold,
        **notes,
    }


def render_predictions(result: Dict[str, Any]) -> None:
    """Display dual predictions and operating-threshold context."""
    probability = result["hit_probability"]
    multiple = result["profit_multiple"]
    call = "Hit" if result["is_hit"] else "Flop"
    first, second = st.columns(2)
    first.metric("Hit probability", f"{probability:.1%}")
    second.metric("Predicted profit multiple", f"{multiple:.2f}×")
    st.subheader(
        f"{probability:.0%} likely to be a hit; predicted to earn "
        f"approximately {multiple:.1f}× its production budget."
    )
    st.caption(
        f"Classification: **{call}**, using the test-tuned "
        f"{result['threshold']:.3f} probability threshold."
    )
    unknown = result["unknown_actors"] + result["unknown_directors"]
    if unknown:
        st.info(
            "No historical match was found for "
            + ", ".join(unknown)
            + "; they were treated as rookies."
        )
    if result["future_budget_assumption"]:
        st.info(
            f"No completed CPI value exists for this future year, so the entered "
            f"budget is treated as already expressed in {INFLATION_BASE_YEAR} dollars."
        )


def _human_feature(name: str) -> str:
    replacements = {
        "is_franchise": "Franchise",
        "log_budget": "Budget (log)",
        "runtime": "Runtime",
        "release_year": "Release year",
        "star_power": "Cast star power",
        "director_star_power": "Director star power",
        "cast_size": "Cast size",
        "n_rookie_cast": "Rookie cast count",
        "frac_rookie_cast": "Rookie cast share",
    }
    if name in replacements:
        return replacements[name]
    return (
        name.replace("genre_", "Genre: ")
        .replace("studio_", "Studio: ")
        .replace("release_window_", "Release window: ")
        .replace("_", " ")
        .title()
    )


def _render_shap_bar(explainer, X: pd.DataFrame, title: str) -> None:
    explanation = explainer(X)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, -1]
    impacts = pd.Series(values[0], index=X.columns)
    top = impacts.loc[impacts.abs().nlargest(8).index].sort_values()
    colors = ["#d95f5f" if value < 0 else "#3a9d5d" for value in top]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.barh([_human_feature(name) for name in top.index], top.values, color=colors)
    axis.axvline(0, color="#777777", linewidth=0.8)
    axis.set_xlabel("SHAP impact (left lowers, right raises prediction)")
    axis.set_title(title)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


def render_shap_explanation(result: Dict[str, Any], artifacts: Dict[str, Any]) -> None:
    """Render local classifier and regressor SHAP contribution charts."""
    st.subheader("Why the models predicted this")
    classifier_tab, regressor_tab = st.tabs(["Hit probability", "Profit multiple"])
    with classifier_tab:
        _render_shap_bar(
            artifacts["classifier_explainer"],
            result["X"],
            "Top influences on hit probability",
        )
    with regressor_tab:
        _render_shap_bar(
            artifacts["regressor_explainer"],
            result["X"],
            "Top influences on predicted log profit multiple",
        )


def main() -> None:
    """Render the complete no-network inference application."""
    st.set_page_config(page_title="Kleos", page_icon="🎬", layout="wide")
    st.title("Kleos")
    st.write(
        "Estimate box-office outcomes from information available before release."
    )
    artifacts = load_artifacts()
    raw = render_input_form()
    if not raw:
        return
    try:
        result = predict_movie(raw, artifacts)
    except (ValueError, KeyError, AssertionError) as exc:
        st.error(f"Could not build this prediction: {exc}")
        return
    render_predictions(result)
    render_shap_explanation(result, artifacts)
    st.warning(
        "This is a pre-release estimate for a difficult, noisy problem—not a "
        "guarantee of financial performance."
    )


if __name__ == "__main__":
    main()
