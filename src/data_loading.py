"""Load the raw TMDB Movies Dataset and apply financial-validity filters.

Use the 900k+ TMDB dump (not the old TMDB 5000 set). Raw files stay in
``data/raw/``; this module must not write processed features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from config import MIN_BUDGET, MIN_REVENUE, RAW_TMDB_PATH


def load_raw_tmdb(
    path: Path = RAW_TMDB_PATH,
    usecols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Read the untouched TMDB Movies Dataset from disk.

    Parameters
    ----------
    path :
        Location of the downloaded dump (CSV or other tabular format).
        Defaults to ``config.RAW_TMDB_PATH``.
    usecols :
        Optional column subset (e.g. ``["id", "budget", "revenue"]`` when
        only financial filters are needed).

    Returns
    -------
    pd.DataFrame
        Unfiltered raw rows. Column names should match the source file
        (typically ``budget``, ``revenue``, ``release_date``, ``genres``,
        ``production_companies``, ``belongs_to_collection``, ``runtime``,
        ``title``, ``id``, etc.). This dump does **not** include cast/crew;
        those come from ``credits_cache.jsonl`` via ``src/merge_credits.py``.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw TMDB dump not found at {path}. "
            "Place asaniczka's tmdb_movies.csv in data/raw/."
        )
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def filter_positive_financials(
    df: pd.DataFrame,
    budget_col: str = "budget",
    revenue_col: str = "revenue",
    min_budget: float = MIN_BUDGET,
    min_revenue: float = MIN_REVENUE,
) -> pd.DataFrame:
    """Keep rows with reported budget and revenue strictly above the minima.

    Parameters
    ----------
    df :
        Raw TMDB frame.
    budget_col, revenue_col :
        Source column names.
    min_budget, min_revenue :
        Rows must satisfy budget > min_budget and revenue > min_revenue.
        Defaults come from ``config`` (0.0 → keep strictly positive figures).

    Returns
    -------
    pd.DataFrame
        Filtered copy. Rows dropped are logged by callers that care
        (fetch/merge print counts).
    """
    budget = pd.to_numeric(df[budget_col], errors="coerce")
    revenue = pd.to_numeric(df[revenue_col], errors="coerce")
    mask = (budget > min_budget) & (revenue > min_revenue)
    return df.loc[mask].copy()


def parse_release_dates(df: pd.DataFrame, date_col: str = "release_date") -> pd.DataFrame:
    """Coerce ``release_date`` to datetime and drop rows that cannot be parsed.

    Parameters
    ----------
    df :
        Frame with a raw date column (string or mixed).
    date_col :
        Name of the release-date column.

    Returns
    -------
    pd.DataFrame
        Copy with ``date_col`` as timezone-naive datetime64. Needed before
        chronological sorting and year-based splits.
    """
    raise NotImplementedError("Parse release dates; drop unparseable rows.")


def load_and_filter(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Convenience: load raw TMDB, parse dates, keep positive budget & revenue.

    Parameters
    ----------
    path :
        Optional override for ``config.RAW_TMDB_PATH``.

    Returns
    -------
    pd.DataFrame
        Financially valid rows with parsed ``release_date``. Still unadjusted
        for inflation and without engineered features.
    """
    raise NotImplementedError("Compose load_raw_tmdb -> parse_release_dates -> filter_positive_financials.")
