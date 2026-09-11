"""
Project-wide constants, flags, and paths for Kleos.

Numeric thresholds that affect feature meaning live here (named, documented)
rather than as magic numbers in feature modules. Values that must be *fitted*
from data (e.g. the median profit multiple used for cold-start imputation)
are computed at pipeline time — never hardcoded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# Expected location of the untouched TMDB Movies Dataset download.
# Replace the filename once the exact Kaggle dump is placed in data/raw/.
RAW_TMDB_PATH: Path = RAW_DATA_DIR / "tmdb_movies.csv"
CREDITS_CACHE_PATH: Path = RAW_DATA_DIR / "credits_cache.jsonl"
DETAILS_CACHE_PATH: Path = RAW_DATA_DIR / "details_cache.jsonl"
MOVIES_WITH_CREDITS_PATH: Path = PROCESSED_DATA_DIR / "movies_with_credits.parquet"
PROCESSED_FEATURES_PATH: Path = PROCESSED_DATA_DIR / "features.parquet"

# ---------------------------------------------------------------------------
# TMDB credits fetch (src/fetch_credits.py)
# ---------------------------------------------------------------------------

TMDB_CREDITS_URL_TEMPLATE: str = "https://api.themoviedb.org/3/movie/{movie_id}/credits"
TMDB_DETAILS_URL_TEMPLATE: str = "https://api.themoviedb.org/3/movie/{movie_id}"
# ~20 req/s. Official cap is higher; stay polite and recover via 429 backoff.
TMDB_REQUEST_DELAY_SECONDS: float = 0.05
TMDB_PROGRESS_EVERY: int = 200
TMDB_REQUEST_TIMEOUT_SECONDS: float = 30.0
TMDB_MAX_RETRIES: int = 6

# ---------------------------------------------------------------------------
# Data filters (data_loading.py)
# ---------------------------------------------------------------------------

# Drop rows that cannot support a financial target.
MIN_BUDGET: float = 0.0  # keep rows with budget > MIN_BUDGET
MIN_REVENUE: float = 0.0  # keep rows with revenue > MIN_REVENUE

# ---------------------------------------------------------------------------
# Inflation (inflation.py)
# ---------------------------------------------------------------------------

# Dollar figures are expressed in this calendar year's dollars after CPI adjust.
# 2025 is the most recent complete year in data/cpi_u_annual.csv.
INFLATION_BASE_YEAR: int = 2025

# ---------------------------------------------------------------------------
# Classification target
# ---------------------------------------------------------------------------

# Hit if inflation-adjusted profit multiple (revenue / budget) exceeds this.
# 1.0 = broke even on reported TMDB figures; tune after EDA.
HIT_PROFIT_MULTIPLE_THRESHOLD: float = 1.0

# ---------------------------------------------------------------------------
# Train / test / backtest split (pipeline.py)
# ---------------------------------------------------------------------------

# Most recent calendar year of releases is held out entirely for backtest.py.
# Remaining rows are split 75/25 train/test. Split helpers must be reused
# (not reimplemented inside notebooks).
TRAIN_FRACTION: float = 0.75
TEST_FRACTION: float = 0.25
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Star power (features/star_power.py)
# ---------------------------------------------------------------------------

# Set by notebooks/eda_cast_order.ipynb (20/20 lead-slot agreement, 0/20 alphabetical).
# True  -> Path A: inverse-cast_order weighted average over the full cast
# False -> Path B: unweighted mean of the top-N billed names
# Implement Path A as 1/(order+1): TMDB order is 0-based.
USE_WEIGHTED_STAR_POWER: bool = True

# Used only when USE_WEIGHTED_STAR_POWER is False (Path B).
STAR_POWER_TOP_N: int = 5

# Rookie if the person has strictly fewer than this many *prior* films
# (released before the current movie) in the dataset.
MIN_PRIOR_FILMS: int = 2
# Backward-compatible name used by the initial scaffold.
ROOKIE_PRIOR_FILM_THRESHOLD: int = MIN_PRIOR_FILMS

# Clip each historical outcome before adding it to actor/director running
# averages so one microbudget viral hit cannot dominate career history.
PROFIT_MULTIPLE_CAP: float = 50.0

# Fitted from the training pool only (see compute_median_profit_multiple).
# Cold-start / missing historical scores impute to this median — not mean, not 0.
# Must remain None until the pipeline fits it; do not bake in a numeric default.
MEDIAN_PROFIT_MULTIPLE: Optional[float] = None

# ---------------------------------------------------------------------------
# Studio encoding (features/studio.py)
# ---------------------------------------------------------------------------

# Keep the N most frequent production companies; remaining labels collapse to "other".
TOP_N_STUDIOS: int = 20

# Keep genres represented by at least this many training-set movies.
MIN_GENRE_COUNT: int = 50

# Release month -> model-facing release window. Keep all timing policy here so
# feature code does not contain scattered month literals.
RELEASE_MONTH_TO_WINDOW: dict[int, str] = {
    1: "dump",
    2: "dump",
    3: "spring",
    4: "spring",
    5: "summer",
    6: "summer",
    7: "summer",
    8: "late_summer",
    9: "awards",
    10: "awards",
    11: "holiday",
    12: "holiday",
}

# ---------------------------------------------------------------------------
# Competition density (features/competition.py)
# ---------------------------------------------------------------------------

# A release counts as "wide" if its pre-release, inflation-adjusted budget is
# at or above
# this percentile of the (non-backtest) training distribution.
#
# Justification: a raw same-week title count treats a 50-screen indie the same
# as a tentpole. Budget is known before release and avoids leaking box-office
# outcomes while keeping the +/-2 week window focused on major launches.
WIDE_RELEASE_BUDGET_PERCENTILE: float = 75.0

# Inclusive window around a title's release date, in days (2 weeks either side).
COMPETITION_WINDOW_DAYS: int = 14

# Fitted wide-release dollar cutoff (inflation-adjusted). Set by the pipeline
# from WIDE_RELEASE_BUDGET_PERCENTILE on training data; not a magic number.
WIDE_RELEASE_BUDGET_THRESHOLD: Optional[float] = None

# ---------------------------------------------------------------------------
# Fitted-value helpers (call from pipeline.py after the backtest year is held out)
# ---------------------------------------------------------------------------


def compute_median_profit_multiple(train_df: pd.DataFrame, profit_col: str = "profit_multiple") -> float:
    """Return the training-set median profit multiple for cold-start imputation.

    Parameters
    ----------
    train_df :
        Training rows only. Do not pass test or backtest frames — that would
        leak future outcomes into the imputation constant.
    profit_col :
        Column holding inflation-adjusted revenue / inflation-adjusted budget.

    Returns
    -------
    float
        Median of ``profit_col``. Caller should assign this to
        ``config.MEDIAN_PROFIT_MULTIPLE`` (or thread it through the pipeline)
        rather than hardcoding a number in feature code.
    """
    if profit_col not in train_df.columns:
        raise KeyError(f"Training frame is missing {profit_col!r}")
    values = pd.to_numeric(train_df[profit_col], errors="coerce")
    values = values.replace([float("inf"), -float("inf")], pd.NA).dropna()
    if values.empty:
        raise ValueError("Cannot fit median profit multiple without finite values")
    return float(values.median())


def compute_wide_release_budget_threshold(
    train_df: pd.DataFrame,
    budget_col: str = "budget_adj",
    percentile: float = WIDE_RELEASE_BUDGET_PERCENTILE,
) -> float:
    """Return the inflation-adjusted budget cutoff that defines a wide release.

    Parameters
    ----------
    train_df :
        Training rows only (backtest year already removed).
    budget_col :
        Inflation-adjusted budget column.
    percentile :
        Percentile in [0, 100]; default is ``WIDE_RELEASE_BUDGET_PERCENTILE``.

    Returns
    -------
    float
        Dollar threshold. Store on ``config.WIDE_RELEASE_BUDGET_THRESHOLD``.
    """
    if budget_col not in train_df.columns:
        raise KeyError(f"Training frame is missing {budget_col!r}")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    budget = pd.to_numeric(train_df[budget_col], errors="coerce")
    budget = budget.replace([float("inf"), -float("inf")], pd.NA).dropna()
    if budget.empty:
        raise ValueError("Cannot fit wide-release threshold without budget values")
    return float(budget.quantile(percentile / 100.0))
