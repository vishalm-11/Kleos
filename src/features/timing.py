"""Release-month timing: map calendar month to a bucketed category."""

from __future__ import annotations

import pandas as pd

# Suggested buckets (finalize during EDA; keep the mapping in one place):
#   winter      : Jan–Feb
#   spring      : Mar–Apr
#   summer      : May–Aug
#   dump_month  : Sep (post-summer / pre-Oscar)
#   awards      : Oct–Dec (festival + awards corridor)
# Do not scatter month integers across model code.


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
    raise NotImplementedError("Return the timing bucket for `month`.")


def add_release_timing(
    df: pd.DataFrame,
    date_col: str = "release_date",
    out_col: str = "release_timing",
) -> pd.DataFrame:
    """Add a categorical timing feature derived from release month.

    Parameters
    ----------
    df :
        Frame with parsed ``release_date``.
    date_col :
        Datetime source column.
    out_col :
        Name of the bucket column. One-hot encoding can happen here or
        in the model preprocessor — pick one place and document it.

    Returns
    -------
    pd.DataFrame
        Copy with ``out_col`` (and optional dummy columns).
    """
    raise NotImplementedError("Extract month, map through month_to_bucket, attach column(s).")
