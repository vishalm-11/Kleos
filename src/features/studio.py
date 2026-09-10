"""Top-N production-company encoding; remainder collapsed to 'other'."""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd

from config import TOP_N_STUDIOS


def parse_company_names(raw: object) -> List[str]:
    """Parse a TMDB production_companies cell into company name strings.

    Parameters
    ----------
    raw :
        JSON list of ``{"id": ..., "name": ...}`` objects, often stringified.

    Returns
    -------
    list of str
        Company names. Empty list if missing or unparseable.
    """
    raise NotImplementedError("Parse TMDB production_companies into a list of names.")


def fit_top_studios(
    df: pd.DataFrame,
    n: int = TOP_N_STUDIOS,
    company_col: str = "production_companies",
) -> List[str]:
    """Choose the N most frequent production companies on training data.

    Parameters
    ----------
    df :
        Training frame only.
    n :
        How many names to keep as their own dummy columns.
    company_col :
        Raw TMDB production-companies column.

    Returns
    -------
    list of str
        Ordered top-N company names. Everything else becomes ``studio_other``.
    """
    raise NotImplementedError("Return the n most frequent studio names from train_df.")


def encode_studios(
    df: pd.DataFrame,
    top_studios: Iterable[str],
    company_col: str = "production_companies",
    prefix: str = "studio",
) -> pd.DataFrame:
    """Multi-label one-hot for top studios plus a single 'other' flag.

    A title can have several production companies. If *any* company is outside
    ``top_studios``, ``studio_other`` is 1 (in addition to any top-N hits).

    Parameters
    ----------
    df :
        Frame to encode (train, test, or backtest) using a vocabulary fit on train.
    top_studios :
        Output of ``fit_top_studios``.
    company_col :
        Source column.
    prefix :
        Dummy prefix.

    Returns
    -------
    pd.DataFrame
        Copy with ``studio_<name>`` columns and ``studio_other``.
    """
    raise NotImplementedError("One-hot top-N studios; collapse the rest to studio_other.")
