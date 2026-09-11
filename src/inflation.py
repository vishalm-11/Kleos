"""CPI-adjust budget and revenue to a common base year.

All downstream financial features and targets (profit multiple, hit/flop,
wide-release threshold) should use inflation-adjusted dollars so a 1995
hit is comparable to a recent hit. The bundled table is the U.S. Bureau of
Labor Statistics CPI-U annual average, all items, U.S. city average.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from config import DATA_DIR, INFLATION_BASE_YEAR

CPI_DATA_PATH = DATA_DIR / "cpi_u_annual.csv"


def load_cpi_index(path: Optional[Path] = None) -> pd.Series:
    """Load a year → CPI index mapping.

    Parameters
    ----------
    path :
        Optional path to a CSV with ``year`` and ``cpi`` columns. Defaults to
        the bundled BLS CPI-U annual-average table; no network call is made.

    Returns
    -------
    pd.Series
        Index is calendar year (int), values are CPI levels. Must cover
        every ``release_year`` present in the filtered TMDB frame.
    """
    cpi_path = path or CPI_DATA_PATH
    if not cpi_path.exists():
        raise FileNotFoundError(f"CPI table not found: {cpi_path}")

    table = pd.read_csv(cpi_path)
    required = {"year", "cpi"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"CPI table {cpi_path} is missing columns: {sorted(missing)}")

    years = pd.to_numeric(table["year"], errors="raise").astype(int)
    values = pd.to_numeric(table["cpi"], errors="raise").astype(float)
    if years.duplicated().any():
        duplicates = sorted(years[years.duplicated()].unique().tolist())
        raise ValueError(f"CPI table contains duplicate years: {duplicates}")
    if (values <= 0).any():
        raise ValueError("CPI values must all be positive")

    return pd.Series(values.to_numpy(), index=years.to_numpy(), name="cpi").sort_index()


def cpi_adjust(
    amount: Union[float, pd.Series],
    year: Union[int, pd.Series],
    cpi_by_year: pd.Series,
    base_year: int = INFLATION_BASE_YEAR,
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
        Target year; default ``config.INFLATION_BASE_YEAR``.

    Returns
    -------
    float or pd.Series
        ``amount * (cpi[base_year] / cpi[year])``.
    """
    if base_year not in cpi_by_year.index:
        raise ValueError(
            f"Inflation base year {base_year} is outside the CPI table range "
            f"{int(cpi_by_year.index.min())}–{int(cpi_by_year.index.max())}"
        )

    if isinstance(year, pd.Series):
        numeric_year = pd.to_numeric(year, errors="coerce")
        if numeric_year.isna().any():
            bad_rows = numeric_year.index[numeric_year.isna()].tolist()[:10]
            raise ValueError(f"Release year is missing or invalid at rows: {bad_rows}")
        int_year = numeric_year.astype(int)
        missing_years = sorted(set(int_year.unique()).difference(cpi_by_year.index))
        if missing_years:
            raise ValueError(
                f"Release years outside CPI table range "
                f"{int(cpi_by_year.index.min())}–{int(cpi_by_year.index.max())}: "
                f"{missing_years}"
            )
        multipliers = int_year.map(cpi_by_year)
        numeric_amount = pd.to_numeric(amount, errors="coerce")
        return numeric_amount * (float(cpi_by_year.loc[base_year]) / multipliers)

    if pd.isna(year):
        raise ValueError("Release year is missing")
    int_year = int(year)
    if int_year not in cpi_by_year.index:
        raise ValueError(
            f"Release year {int_year} is outside the CPI table range "
            f"{int(cpi_by_year.index.min())}–{int(cpi_by_year.index.max())}"
        )
    return float(amount) * (
        float(cpi_by_year.loc[base_year]) / float(cpi_by_year.loc[int_year])
    )


def apply_inflation_adjustment(
    df: pd.DataFrame,
    budget_col: str = "budget",
    revenue_col: str = "revenue",
    year_col: str = "release_year",
    base_year: int = INFLATION_BASE_YEAR,
    cpi_path: Optional[Path] = None,
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
    cpi_path :
        Optional alternate CPI CSV, primarily useful for testing.

    Returns
    -------
    pd.DataFrame
        Copy with at least:
        - ``budget_adj``
        - ``revenue_adj``
        - ``profit_multiple`` = revenue_adj / budget_adj
    """
    required = {budget_col, revenue_col, year_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns required for inflation adjustment: {sorted(missing)}")

    result = df.copy()
    cpi = load_cpi_index(cpi_path)
    result["budget_adj"] = cpi_adjust(
        result[budget_col], result[year_col], cpi, base_year
    )
    result["revenue_adj"] = cpi_adjust(
        result[revenue_col], result[year_col], cpi, base_year
    )

    if (result["budget_adj"] <= 0).any():
        raise ValueError("budget must be positive before computing profit_multiple")
    result["profit_multiple"] = result["revenue_adj"] / result["budget_adj"]
    return result