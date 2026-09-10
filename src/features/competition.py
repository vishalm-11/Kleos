"""Wide-release competition density in a +/- 2 week window.

"Wide release" is *not* "any title that opened the same week". It is a title
whose inflation-adjusted revenue meets ``config.WIDE_RELEASE_REVENUE_THRESHOLD``,
a cutoff fitted as ``config.WIDE_RELEASE_REVENUE_PERCENTILE`` of the training
distribution (see the justification on that constant in config.py).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from config import (
    COMPETITION_WINDOW_DAYS,
    WIDE_RELEASE_REVENUE_THRESHOLD,
)


def is_wide_release(
    revenue_adj: float,
    threshold: Optional[float] = WIDE_RELEASE_REVENUE_THRESHOLD,
) -> bool:
    """True if inflation-adjusted revenue is at or above the wide-release cutoff.

    Parameters
    ----------
    revenue_adj :
        CPI-adjusted box office.
    threshold :
        Fitted dollar cutoff from ``config.compute_wide_release_revenue_threshold``.
        Must not be a magic number inlined here.

    Returns
    -------
    bool
    """
    raise NotImplementedError("Return revenue_adj >= threshold; threshold must be fitted.")


def add_wide_release_flag(
    df: pd.DataFrame,
    revenue_col: str = "revenue_adj",
    threshold: Optional[float] = WIDE_RELEASE_REVENUE_THRESHOLD,
    out_col: str = "is_wide_release",
) -> pd.DataFrame:
    """Mark each row as wide or not using the fitted revenue threshold.

    Parameters
    ----------
    df :
        Inflation-adjusted frame.
    revenue_col :
        Inflation-adjusted revenue.
    threshold :
        ``config.WIDE_RELEASE_REVENUE_THRESHOLD`` after the pipeline fits it.
    out_col :
        Binary flag name.

    Returns
    -------
    pd.DataFrame
        Copy with ``out_col``.
    """
    raise NotImplementedError("Attach is_wide_release from the fitted threshold.")


def competition_density(
    df: pd.DataFrame,
    row_index: object,
    window_days: int = COMPETITION_WINDOW_DAYS,
    date_col: str = "release_date",
    wide_col: str = "is_wide_release",
) -> int:
    """Count other wide releases in ``[+/- window_days]`` around this title.

    The title itself is excluded. Non-wide titles do not increment the count.

    Parameters
    ----------
    df :
        Frame that already has ``is_wide_release`` and parsed dates.
    row_index :
        Index label of the movie being scored.
    window_days :
        Half-width of the window (default 14 = two weeks either side).
    date_col :
        Release date.
    wide_col :
        Wide-release flag.

    Returns
    -------
    int
        Number of *other* wide releases in the closed window.
    """
    raise NotImplementedError("Count other is_wide_release rows within +/- window_days.")


def add_competition_density(
    df: pd.DataFrame,
    window_days: int = COMPETITION_WINDOW_DAYS,
    out_col: str = "competition_density",
) -> pd.DataFrame:
    """Attach a wide-release density feature for every row.

    Parameters
    ----------
    df :
        Frame with dates and ``is_wide_release``.
    window_days :
        See ``config.COMPETITION_WINDOW_DAYS``.
    out_col :
        Feature name.

    Returns
    -------
    pd.DataFrame
        Copy with ``out_col``. Implementation may vectorize (e.g. sweep-line
        or merge_asof) rather than calling ``competition_density`` row-wise.
    """
    raise NotImplementedError("Add competition_density for all rows.")
