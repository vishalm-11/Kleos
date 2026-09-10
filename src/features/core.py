"""Core numeric features: log budget, runtime, release year."""

from __future__ import annotations

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
    raise NotImplementedError("Add integer release_year from release_date.")


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
        Copy with ``out_col`` = log(budget). Choose log vs log1p at implement time.
    """
    raise NotImplementedError("Add log(budget_adj).")


def add_runtime_feature(
    df: pd.DataFrame,
    runtime_col: str = "runtime",
    out_col: str = "runtime",
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

    Returns
    -------
    pd.DataFrame
        Copy with invalid/zero runtimes handled (drop, impute, or flag —
        decide at implement time and document the choice).
    """
    raise NotImplementedError("Clean and attach runtime.")


def build_core_features(df: pd.DataFrame) -> pd.DataFrame:
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
    raise NotImplementedError("Call add_release_year, add_log_budget, add_runtime_feature.")
