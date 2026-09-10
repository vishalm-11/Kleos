"""Franchise / collection membership as a binary flag."""

from __future__ import annotations

import pandas as pd


def add_belongs_to_collection(
    df: pd.DataFrame,
    collection_col: str = "belongs_to_collection",
    out_col: str = "is_franchise",
) -> pd.DataFrame:
    """Flag titles that TMDB lists as part of a collection.

    Parameters
    ----------
    df :
        Raw or partially processed frame.
    collection_col :
        TMDB field: typically a JSON object when in a collection, null otherwise.
    out_col :
        Binary flag name.

    Returns
    -------
    pd.DataFrame
        Copy with ``out_col`` = 1 if the title belongs to a collection, else 0.
        Do not expand collection identity into high-cardinality IDs here.
    """
    raise NotImplementedError("Set is_franchise from belongs_to_collection nullness/payload.")
