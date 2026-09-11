"""Core numeric features: log budget, runtime, release year."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def add_release_year(
    df: pd.DataFrame,
    date_col: str = "release_date",
    out_col: str = "release_year",
) -> pd.DataFrame:
    """Extract calendar year from ``release_date``.

    Parameters
    ----------
    df :
        Frame with a parsed datetime ``date_col``.
    date_col :
        Source datetime column.
    out_col :
        Name of the integer year column to add.

    Returns
    -------
    pd.DataFrame
        Copy with ``out_col`` populated.
    """
    if date_col not in df.columns:
        raise KeyError(f"Missing release-date column: {date_col}")
    result = df.copy()
    dates = pd.to_datetime(result[date_col], errors="coerce")
    if dates.isna().any():
        bad_rows = dates.index[dates.isna()].tolist()[:10]
        raise ValueError(f"Invalid or missing release dates at rows: {bad_rows}")
    result[date_col] = dates
    result[out_col] = dates.dt.year.astype(int)
    return result


def add_log_budget(
    df: pd.DataFrame,
    budget_col: str = "budget_adj",
    out_col: str = "log_budget",
) -> pd.DataFrame:
    """Add log-transformed inflation-adjusted budget.

    Parameters
    ----------
    df :
        Frame with a strictly positive budget column.
    budget_col :
        Prefer ``budget_adj`` so the feature is in constant dollars.
    out_col :
        Name of the log-budget column.

    Returns
    -------
    pd.DataFrame
        Copy with ``out_col`` = ``log1p(budget_adj)``.
    """
    if budget_col not in df.columns:
        raise KeyError(f"Missing budget column: {budget_col}")
    result = df.copy()
    budget = pd.to_numeric(result[budget_col], errors="coerce")
    if budget.isna().any() or (budget < 0).any():
        raise ValueError(f"{budget_col} must contain non-negative numeric values")
    result[out_col] = np.log1p(budget)
    return result


def add_runtime_feature(
    df: pd.DataFrame,
    runtime_col: str = "runtime",
    out_col: str = "runtime",
    median_runtime: Optional[float] = None,
    flag_col: str = "runtime_imputed",
) -> pd.DataFrame:
    """Clean runtime (minutes) for modeling.

    Parameters
    ----------
    df :
        Frame with raw TMDB runtime.
    runtime_col :
        Source column.
    out_col :
        Output column (may overwrite ``runtime`` after cleaning).
    median_runtime :
        Optional fitted median to reuse on test/backtest. If omitted, compute
        the median from positive runtimes in ``df``.
    flag_col :
        Binary indicator for rows whose runtime was null, nonnumeric, or <= 0.

    Returns
    -------
    pd.DataFrame
        Copy with invalid/zero runtimes median-imputed and ``flag_col`` added.
    """
    if runtime_col not in df.columns:
        raise KeyError(f"Missing runtime column: {runtime_col}")
    result = df.copy()
    runtime = pd.to_numeric(result[runtime_col], errors="coerce")
    invalid = runtime.isna() | (runtime <= 0)

    if median_runtime is None:
        valid = runtime[~invalid]
        if valid.empty:
            raise ValueError("Cannot impute runtime: no positive runtimes available")
        median_runtime = float(valid.median())
    elif not np.isfinite(median_runtime) or median_runtime <= 0:
        raise ValueError("median_runtime must be a positive finite number")

    result[flag_col] = invalid.astype("int8")
    result[out_col] = runtime.mask(invalid, float(median_runtime))
    return result


def build_core_features(
    df: pd.DataFrame,
    median_runtime: Optional[float] = None,
) -> pd.DataFrame:
    """Compose year, log budget, and runtime onto ``df``.

    Parameters
    ----------
    df :
        Inflation-adjusted frame (needs ``release_date`` and ``budget_adj``).

    Returns
    -------
    pd.DataFrame
        Copy with ``release_year``, ``log_budget``, and cleaned ``runtime``.
    """
    result = add_release_year(df)
    result = add_log_budget(result)
    return add_runtime_feature(result, median_runtime=median_runtime)
