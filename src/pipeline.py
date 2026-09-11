"""Orchestrate feature construction and the train / test / backtest split.

Chronological order is mandatory before any rolling or historical feature
(star power, competition windows that depend on prior context, etc.).
Call ``assert_chronologically_sorted`` after the sort; do not skip it.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import joblib
import pandas as pd

from config import (
    BACKTEST_YEAR,
    ENCODERS_PATH,
    HIT_THRESHOLD,
    MOVIES_WITH_CREDITS_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    SPLIT_STRATEGY,
    TRAIN_FRACTION,
)
from src.data_loading import filter_adjusted_financials
from src.features.competition import CompetitionEncoder
from src.features.core import add_release_year, build_core_features
from src.features.franchise import add_belongs_to_collection
from src.features.genre import GenreEncoder
from src.features.star_power import FEATURE_COLUMNS as STAR_POWER_FEATURE_COLUMNS
from src.features.star_power import StarPowerEncoder
from src.features.studio import StudioEncoder
from src.features.timing import RELEASE_WINDOWS, add_release_timing
from src.inflation import apply_inflation_adjustment, load_cpi_index

LOGGER = logging.getLogger(__name__)

COMPETITION_FEATURE_COLUMNS = [
    "n_competitors_2wk",
    "n_competitors_1wk",
    "n_same_genre_competitors_2wk",
]
CORE_FEATURE_COLUMNS = [
    "log_budget",
    "runtime",
    "runtime_imputed",
    "release_year",
]


def sort_chronologically(
    df: pd.DataFrame,
    date_col: str = "release_date",
) -> pd.DataFrame:
    """Return ``df`` sorted by release date (stable, oldest first).

    Parameters
    ----------
    df :
        Frame with a parsed datetime column.
    date_col :
        Sort key.

    Returns
    -------
    pd.DataFrame
        Sorted copy with a reset or preserved index — document the choice.
    """
    if date_col not in df.columns:
        raise KeyError(f"Missing release-date column: {date_col}")
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    if result[date_col].isna().any():
        bad = result.index[result[date_col].isna()].tolist()[:10]
        raise ValueError(f"Invalid or missing release dates at rows: {bad}")
    return result.sort_values(date_col, kind="stable").reset_index(drop=True)


def assert_chronologically_sorted(
    df: pd.DataFrame,
    date_col: str = "release_date",
) -> None:
    """Raise if rows are not in non-decreasing release-date order.

    This is the leakage-prevention test stub: run it after
    ``sort_chronologically`` and before star-power / any look-back feature.

    Parameters
    ----------
    df :
        Candidate frame.
    date_col :
        Date column that must be monotonic non-decreasing.

    Raises
    ------
    AssertionError
        If ``date_col`` is not sorted or contains nulls.
    """
    if date_col not in df.columns:
        raise KeyError(f"Missing release-date column: {date_col}")
    dates = pd.to_datetime(df[date_col], errors="coerce")
    assert dates.notna().all(), f"{date_col} contains null or invalid dates"
    assert dates.is_monotonic_increasing, f"{date_col} is not chronological"


def backtest_year(df: pd.DataFrame, date_col: str = "release_date") -> int:
    """Return the most recent calendar year present in ``df``.

    Parameters
    ----------
    df :
        Frame with parsed dates (full financially valid set, pre-split).
    date_col :
        Release date.

    Returns
    -------
    int
        Hold-out year for ``backtest.py``.
    """
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().all():
        raise ValueError("Cannot determine backtest year without release dates")
    return BACKTEST_YEAR


def split_backtest_and_model_pool(
    df: pd.DataFrame,
    date_col: str = "release_date",
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Hold out the most recent year entirely; return the remaining model pool.

    Parameters
    ----------
    df :
        Full processed (or pre-feature) frame.
    date_col :
        Release date.

    Returns
    -------
    model_pool, backtest_df, holdout_year : tuple
        ``backtest_df`` is every row in ``holdout_year``.
        ``model_pool`` is every earlier year (this is what gets 75/25 split).
        ``holdout_year`` is the integer year that was peeled off.
    """
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError("Cannot split rows with missing release dates")
    years = dates.dt.year
    holdout_year = backtest_year(df, date_col)
    if (years > holdout_year).any():
        future_years = sorted(years.loc[years > holdout_year].unique().tolist())
        raise ValueError(
            f"Partial/future years must be removed before splitting: {future_years}"
        )
    model_pool = df.loc[years < holdout_year].copy()
    backtest_df = df.loc[years == holdout_year].copy()
    if model_pool.empty or backtest_df.empty:
        raise ValueError(
            f"Need both pre-{holdout_year} rows and {holdout_year} backtest rows"
        )
    return model_pool, backtest_df, holdout_year


def split_train_test(
    model_pool: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    date_col: str = "release_date",
    split_strategy: str = SPLIT_STRATEGY,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the non-backtest pool 75/25 for train and test.

    Prefer a chronological split (earliest ``train_fraction`` of dates → train,
    remainder → test) so test does not sit in the middle of the training era.
    If a random split is used instead, document why and still keep the
    backtest year completely out of both sides.

    Parameters
    ----------
    model_pool :
        Output of ``split_backtest_and_model_pool`` (no hold-out year rows).
    train_fraction :
        Default ``config.TRAIN_FRACTION`` (0.75). Test is 1 - train_fraction.
    date_col :
        Used if the split is chronological.

    Returns
    -------
    train_df, test_df : tuple
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if split_strategy not in {"chronological", "random"}:
        raise ValueError(
            "split_strategy must be either 'chronological' or 'random', "
            f"got {split_strategy!r}"
        )
    ordered = sort_chronologically(model_pool, date_col)
    split_at = int(len(ordered) * train_fraction)
    if split_at == 0 or split_at == len(ordered):
        raise ValueError("Model pool is too small for non-empty train/test splits")
    if split_strategy == "random":
        ordered = ordered.sample(frac=1, random_state=random_state)
    return (
        sort_chronologically(ordered.iloc[:split_at], date_col),
        sort_chronologically(ordered.iloc[split_at:], date_col),
    )


def split_train_test_backtest(
    df: pd.DataFrame,
    date_col: str = "release_date",
    split_strategy: str = SPLIT_STRATEGY,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reusable three-way split: train, test, backtest.

    Parameters
    ----------
    df :
        Full frame (features optional; dates required).
    date_col :
        Release date.

    Returns
    -------
    train_df, test_df, backtest_df : tuple
        Backtest = most recent year. Train/test = 75/25 of the rest.
    """
    model_pool, backtest_df, _ = split_backtest_and_model_pool(df, date_col)
    train_df, test_df = split_train_test(
        model_pool,
        date_col=date_col,
        split_strategy=split_strategy,
    )
    return (
        train_df,
        test_df,
        sort_chronologically(backtest_df, date_col),
    )


def fit_encoders(
    train_df: pd.DataFrame,
    all_movies: pd.DataFrame,
) -> Dict[str, Any]:
    """Fit train-only state and retain chronological pools for app inference."""
    valid_runtime = pd.to_numeric(train_df["runtime"], errors="coerce")
    valid_runtime = valid_runtime.loc[valid_runtime > 0]
    if valid_runtime.empty:
        raise ValueError("Cannot fit runtime imputation without positive runtimes")

    genre = GenreEncoder().fit(train_df)
    studio = StudioEncoder().fit(train_df)
    star_power = StarPowerEncoder().fit(train_df)
    competition = CompetitionEncoder().fit(train_df)
    return {
        "genre": genre,
        "studio": studio,
        "star_power": star_power,
        "competition": competition,
        "median_runtime": float(valid_runtime.median()),
        # Outcomes are used only when their release_date is strictly before a
        # future prediction. StarPowerEncoder enforces that date boundary.
        "history_pool": all_movies[
            ["id", "release_date", "profit_multiple", "cast", "directors"]
        ].copy(),
        "competitor_pool": all_movies[
            ["id", "release_date", "budget_adj", "genres"]
        ].copy(),
    }


def feature_columns(encoders: Mapping[str, Any]) -> list[str]:
    """Return the sole approved model-input columns in stable order."""
    genre_columns = [
        f"genre_{genre}" for genre in encoders["genre"].vocabulary_
    ]
    studio_columns = [
        f"studio_{studio}" for studio in encoders["studio"].top_studios_
    ] + ["studio_other"]
    timing_columns = [f"release_window_{window}" for window in RELEASE_WINDOWS]
    return (
        CORE_FEATURE_COLUMNS
        + timing_columns
        + genre_columns
        + ["is_franchise"]
        + studio_columns
        + STAR_POWER_FEATURE_COLUMNS
        + COMPETITION_FEATURE_COLUMNS
    )


def build_features(
    df: pd.DataFrame,
    encoders: Mapping[str, Any],
    *,
    history_df: Optional[pd.DataFrame] = None,
    competitor_pool: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Apply fitted encoders and return a frame containing engineered columns.

    For app inference, omit both optional pools to use the historical and
    release-calendar pools persisted with ``encoders.joblib``. Pipeline calls
    pass split-appropriate star history explicitly.
    """
    result = build_core_features(
        df,
        median_runtime=float(encoders["median_runtime"]),
    )
    result = add_release_timing(result)
    result = add_belongs_to_collection(result)
    result = encoders["genre"].transform(result)
    result = encoders["studio"].transform(result)

    star_history = (
        encoders["history_pool"] if history_df is None else history_df
    )
    result = encoders["star_power"].transform(
        result,
        history_df=star_history,
    )
    competition_pool = (
        encoders["competitor_pool"]
        if competitor_pool is None
        else competitor_pool
    )
    result = encoders["competition"].transform(
        result,
        competitor_pool=competition_pool,
    )
    return result


def _prepare_movies(df: pd.DataFrame) -> pd.DataFrame:
    """Validate dates, remove non-model years, inflation-adjust, and sort."""
    required = {
        "id",
        "release_date",
        "budget",
        "revenue",
        "runtime",
        "genres",
        "production_companies",
        "belongs_to_collection",
        "cast",
        "directors",
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Input parquet is missing columns: {sorted(missing)}")

    result = df.copy()
    result["release_date"] = pd.to_datetime(result["release_date"], errors="coerce")
    invalid_dates = int(result["release_date"].isna().sum())
    if invalid_dates:
        LOGGER.warning("Dropping %s rows with invalid release_date", invalid_dates)
        result = result.loc[result["release_date"].notna()].copy()
    result = add_release_year(result)

    cpi = load_cpi_index()
    min_year = int(cpi.index.min())
    valid_year = result["release_year"].between(min_year, BACKTEST_YEAR)
    dropped_years = int((~valid_year).sum())
    if dropped_years:
        values = sorted(result.loc[~valid_year, "release_year"].unique().tolist())
        LOGGER.warning(
            "Dropping %s rows outside model years %s–%s: %s",
            dropped_years,
            min_year,
            BACKTEST_YEAR,
            values,
        )
        result = result.loc[valid_year].copy()

    result = apply_inflation_adjustment(result)
    before_financial_floor = len(result)
    result = filter_adjusted_financials(result)
    LOGGER.info(
        "Adjusted-dollar filters retained %s rows and dropped %s",
        len(result),
        before_financial_floor - len(result),
    )
    result = sort_chronologically(result)
    assert_chronologically_sorted(result)
    result["is_hit"] = (
        result["profit_multiple"] >= HIT_THRESHOLD
    ).astype("int8")
    return result


def _log_split_summary(name: str, frame: pd.DataFrame) -> None:
    hit_rate = float(frame["is_hit"].mean())
    LOGGER.info(
        "%s: %s rows; hit=%s (%.2f%%), flop=%s (%.2f%%)",
        name,
        len(frame),
        int(frame["is_hit"].sum()),
        hit_rate * 100,
        int((1 - frame["is_hit"]).sum()),
        (1 - hit_rate) * 100,
    )


def _select_feature_matrix(
    engineered: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    missing = set(columns).difference(engineered.columns)
    if missing:
        raise KeyError(f"Engineered frame is missing features: {sorted(missing)}")
    if len(columns) != len(set(columns)):
        raise AssertionError("Feature column list contains duplicates")
    forbidden = {"profit_multiple", "is_hit"}
    assert forbidden.isdisjoint(columns), "Target leaked into feature columns"
    assert not any(
        column.casefold().startswith("revenue") for column in columns
    ), "Revenue leaked into feature columns"

    matrix = engineered.loc[:, columns].copy().reset_index(drop=True)
    all_null = matrix.columns[matrix.isna().all()].tolist()
    if all_null:
        raise AssertionError(f"All-null feature columns: {all_null}")
    return matrix


def _persist_pipeline_outputs(
    output_dir: Path,
    matrices: Mapping[str, pd.DataFrame],
    split_frames: Mapping[str, pd.DataFrame],
    columns: list[str],
    encoders: Dict[str, Any],
    encoders_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, matrix in matrices.items():
        matrix.to_parquet(output_dir / f"X_{split}.parquet", index=False)
        frame = split_frames[split].reset_index(drop=True)
        pd.DataFrame({"is_hit": frame["is_hit"]}).to_parquet(
            output_dir / f"y_{split}_cls.parquet",
            index=False,
        )
        pd.DataFrame({"profit_multiple": frame["profit_multiple"]}).to_parquet(
            output_dir / f"y_{split}_reg.parquet",
            index=False,
        )
        metadata_columns = [
            column
            for column in ("id", "title", "release_date", "release_year")
            if column in frame.columns
        ]
        frame[metadata_columns].rename(columns={"id": "movie_id"}).to_parquet(
            output_dir / f"movie_ids_{split}.parquet",
            index=False,
        )

    with (output_dir / "feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(columns, handle, indent=2)
        handle.write("\n")

    encoders_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, encoders_path)
    LOGGER.info("Persisted fitted encoders to %s", encoders_path)


def run_pipeline(
    persist: bool = True,
    df: Optional[pd.DataFrame] = None,
    output_dir: Path = PROCESSED_DATA_DIR,
    encoders_path: Path = ENCODERS_PATH,
    split_strategy: str = SPLIT_STRATEGY,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and optionally persist model-ready train/test/backtest matrices."""
    started = time.perf_counter()
    if df is None:
        LOGGER.info("Loading %s", MOVIES_WITH_CREDITS_PATH)
        df = pd.read_parquet(MOVIES_WITH_CREDITS_PATH)
    movies = _prepare_movies(df)
    LOGGER.info("Using %s train/test split strategy", split_strategy)
    train_df, test_df, backtest_df = split_train_test_backtest(
        movies,
        split_strategy=split_strategy,
    )
    split_frames = {
        "train": train_df,
        "test": test_df,
        "backtest": backtest_df,
    }
    for name, frame in split_frames.items():
        _log_split_summary(name, frame)

    all_movies = pd.concat(
        [train_df, test_df, backtest_df],
        ignore_index=True,
        sort=False,
    )
    all_movies = sort_chronologically(all_movies)
    assert_chronologically_sorted(all_movies)
    encoders = fit_encoders(train_df, all_movies)
    LOGGER.info(
        "Fitted median profit multiple %.4f and wide-release budget cutoff $%s",
        encoders["star_power"].median_profit_multiple_,
        f"{encoders['competition'].fitted_threshold_:,.2f}",
    )

    empty_history = train_df.iloc[0:0].copy()
    train_engineered = build_features(
        train_df,
        encoders,
        history_df=empty_history,
        competitor_pool=all_movies,
    )
    test_engineered = build_features(
        test_df,
        encoders,
        history_df=train_df,
        competitor_pool=all_movies,
    )
    pre_backtest = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    backtest_engineered = build_features(
        backtest_df,
        encoders,
        history_df=pre_backtest,
        competitor_pool=all_movies,
    )
    engineered = {
        "train": train_engineered,
        "test": test_engineered,
        "backtest": backtest_engineered,
    }

    columns = feature_columns(encoders)
    encoders["feature_columns"] = columns
    encoders["backtest_year"] = BACKTEST_YEAR
    encoders["split_strategy"] = split_strategy
    matrices = {
        name: _select_feature_matrix(frame, columns)
        for name, frame in engineered.items()
    }

    for name, matrix in matrices.items():
        null_rates = matrix.isna().mean()
        high_null = null_rates.loc[null_rates > 0.05]
        if high_null.empty:
            LOGGER.info("%s: no feature columns above 5%% null", name)
        else:
            LOGGER.warning(
                "%s features above 5%% null: %s",
                name,
                {key: round(value, 4) for key, value in high_null.items()},
            )
    LOGGER.info("Final feature count: %s", len(columns))

    if persist:
        _persist_pipeline_outputs(
            output_dir,
            matrices,
            split_frames,
            columns,
            encoders,
            encoders_path,
        )

    LOGGER.info("Pipeline completed in %.2fs", time.perf_counter() - started)
    return matrices["train"], matrices["test"], matrices["backtest"]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_pipeline()


if __name__ == "__main__":
    main()
