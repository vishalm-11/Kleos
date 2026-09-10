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
PROCESSED_FEATURES_PATH: Path = PROCESSED_DATA_DIR / "features.parquet"

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
CPI_BASE_YEAR: int = 2024

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

# Flip after notebooks/eda_cast_order.ipynb validates TMDB cast_order.
# True  -> Path A: inverse-cast_order weighted average over the full cast
# False -> Path B: unweighted mean of the top-N billed names
USE_WEIGHTED_STAR_POWER: bool = False

# Used only when USE_WEIGHTED_STAR_POWER is False (Path B).
STAR_POWER_TOP_N: int = 5

# Rookie if the person has strictly fewer than this many *prior* films
# (released before the current movie) in the dataset.
ROOKIE_PRIOR_FILM_THRESHOLD: int = 2

# Fitted from the training pool only (see compute_median_profit_multiple).
# Cold-start / missing historical scores impute to this median — not mean, not 0.
# Must remain None until the pipeline fits it; do not bake in a numeric default.
MEDIAN_PROFIT_MULTIPLE: Optional[float] = None

# ---------------------------------------------------------------------------
# Studio encoding (features/studio.py)
# ---------------------------------------------------------------------------

# Keep the N most frequent production companies; remaining labels collapse to "other".
TOP_N_STUDIOS: int = 20

# ---------------------------------------------------------------------------
# Competition density (features/competition.py)
# ---------------------------------------------------------------------------

# A release counts as "wide" if its inflation-adjusted revenue is at or above
# this percentile of the (non-backtest) training distribution.
#
# Justification: a raw same-week title count treats a 50-screen indie the same
# as a tentpole. Gating on inflation-adjusted revenue percentile keeps the
# +/-2 week window focused on commercially relevant openers.
WIDE_RELEASE_REVENUE_PERCENTILE: float = 75.0

# Inclusive window around a title's release date, in days (2 weeks either side).
COMPETITION_WINDOW_DAYS: int = 14

# Fitted wide-release dollar cutoff (inflation-adjusted). Set by the pipeline
# from WIDE_RELEASE_REVENUE_PERCENTILE on training data; not a magic number.
WIDE_RELEASE_REVENUE_THRESHOLD: Optional[float] = None

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
    raise NotImplementedError(
        "Fit the median on train_df[profit_col] and store it on "
        "config.MEDIAN_PROFIT_MULTIPLE or pass it into star_power builders."
    )


def compute_wide_release_revenue_threshold(
    train_df: pd.DataFrame,
    revenue_col: str = "revenue_adj",
    percentile: float = WIDE_RELEASE_REVENUE_PERCENTILE,
) -> float:
    """Return the inflation-adjusted revenue cutoff that defines a wide release.

    Parameters
    ----------
    train_df :
        Training rows only (backtest year already removed).
    revenue_col :
        Inflation-adjusted revenue column.
    percentile :
        Percentile in [0, 100]; default is ``WIDE_RELEASE_REVENUE_PERCENTILE``.

    Returns
    -------
    float
        Dollar threshold. Store on ``config.WIDE_RELEASE_REVENUE_THRESHOLD``.
    """
    raise NotImplementedError(
        "Compute the training-set percentile of inflation-adjusted revenue "
        "and store it on config.WIDE_RELEASE_REVENUE_THRESHOLD."
    )
