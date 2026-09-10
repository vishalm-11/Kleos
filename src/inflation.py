"""CPI-adjust budget and revenue to a common base year.

All downstream financial features and targets (profit multiple, hit/flop,
wide-release threshold) should use inflation-adjusted dollars so a 1995
hit is comparable to a 2023 hit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from config import CPI_BASE_YEAR


def load_cpi_index(path: Optional[Path] = None) -> pd.Series:
    """Load a year → CPI index mapping.

    Parameters
    ----------
    path :
        Optional path to a CPI table (CSV with year and index columns).
        If omitted, implementation may fetch or embed a standard series
        (e.g. US CPI-U, annual average).

    Returns
    -------
    pd.Series
        Index is calendar year (int), values are CPI levels. Must cover
        every ``release_year`` present in the filtered TMDB frame.
    """
    raise NotImplementedError("Return a year-indexed CPI series.")


def cpi_adjust(
    amount: Union[float, pd.Series],
    year: Union[int, pd.Series],
    cpi_by_year: pd.Series,
    base_year: int = CPI_BASE_YEAR,
) -> Union[float, pd.Series]:
    """Convert nominal dollars in ``year`` to ``base_year`` dollars.

    Parameters
    ----------
    amount :
        Nominal budget or revenue.
    year :
        Calendar year the amount was reported (typically release year).
    cpi_by_year :
        Output of ``load_cpi_index``.
    base_year :
        Target year; default ``config.CPI_BASE_YEAR``.

    Returns
    -------
    float or pd.Series
        ``amount * (cpi[base_year] / cpi[year])``.
    """
    raise NotImplementedError("Scale amount by CPI[base_year] / CPI[year].")


def apply_inflation_adjustment(
    df: pd.DataFrame,
    budget_col: str = "budget",
    revenue_col: str = "revenue",
    year_col: str = "release_year",
    base_year: int = CPI_BASE_YEAR,
) -> pd.DataFrame:
    """Add inflation-adjusted budget/revenue columns and a profit multiple.

    Parameters
    ----------
    df :
        Frame with nominal ``budget``, ``revenue``, and a release year.
    budget_col, revenue_col, year_col :
        Source column names.
    base_year :
        Dollars are expressed in this year's terms.

    Returns
    -------
    pd.DataFrame
        Copy with at least:
        - ``budget_adj``
        - ``revenue_adj``
        - ``profit_multiple`` = revenue_adj / budget_adj
    """
    raise NotImplementedError("Attach budget_adj, revenue_adj, and profit_multiple.")
