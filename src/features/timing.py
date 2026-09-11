"""Release-month timing: map calendar month to a bucketed category."""

from __future__ import annotations

import pandas as pd

from config import RELEASE_MONTH_TO_WINDOW

RELEASE_WINDOWS = tuple(dict.fromkeys(RELEASE_MONTH_TO_WINDOW.values()))


def month_to_bucket(month: int) -> str:
    """Map a 1–12 calendar month to a named release-window bucket.

    Parameters
    ----------
    month :
        Calendar month, January = 1.

    Returns
    -------
    str
        Bucket label (e.g. ``"summer"``, ``"awards"``).
    """
    try:
        return RELEASE_MONTH_TO_WINDOW[int(month)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"month must be an integer from 1 through 12; got {month!r}") from exc


def add_release_timing(
    df: pd.DataFrame,
    date_col: str = "release_date",
    out_col: str = "release_window",
) -> pd.DataFrame:
    """Add a categorical timing feature derived from release month.

    Parameters
    ----------
    df :
        Frame with parsed ``release_date``.
    date_col :
        Datetime source column.
    out_col :
        Name of the categorical bucket column.

    Returns
    -------
    pd.DataFrame
        Copy with ``out_col`` plus one stable binary column per configured
        bucket, named ``release_window_<bucket>``.
    """
    if date_col not in df.columns:
        raise KeyError(f"Missing release-date column: {date_col}")
    result = df.copy()
    dates = pd.to_datetime(result[date_col], errors="coerce")
    if dates.isna().any():
        bad_rows = dates.index[dates.isna()].tolist()[:10]
        raise ValueError(f"Invalid or missing release dates at rows: {bad_rows}")

    result[date_col] = dates
    result[out_col] = dates.dt.month.map(month_to_bucket)
    for window in RELEASE_WINDOWS:
        result[f"release_window_{window}"] = (result[out_col] == window).astype("int8")
    return result
