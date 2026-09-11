"""Primary-studio encoding fitted on training data."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional

import pandas as pd

from config import TOP_N_STUDIOS


def parse_company_names(raw: object) -> List[str]:
    """Parse comma-separated production-company names.

    Parameters
    ----------
    raw :
        Comma-separated string from the asaniczka dataset.

    Returns
    -------
    list of str
        Stripped names in source order. Empty for null/blank.
    """
    if raw is None or (not isinstance(raw, (list, tuple, set, dict)) and pd.isna(raw)):
        return []
    if not isinstance(raw, str):
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def primary_studio(raw: object) -> Optional[str]:
    """Return the first listed production company, or ``None``."""
    names = parse_company_names(raw)
    return names[0] if names else None


def fit_top_studios(
    df: pd.DataFrame,
    n: int = TOP_N_STUDIOS,
    company_col: str = "production_companies",
) -> List[str]:
    """Choose the N most frequent primary studios on training data.

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
        Frequency-descending top-N primary studios; ties are alphabetical.
        Everything else becomes ``studio_other`` during transform.
    """
    if company_col not in df.columns:
        raise KeyError(f"Missing production-company column: {company_col}")
    if n < 1:
        raise ValueError("n must be at least 1")
    counts = Counter(
        studio
        for studio in df[company_col].map(primary_studio)
        if studio is not None
    )
    return [
        studio
        for studio, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:n]
    ]


def encode_studios(
    df: pd.DataFrame,
    top_studios: Iterable[str],
    company_col: str = "production_companies",
    prefix: str = "studio",
) -> pd.DataFrame:
    """One-hot the primary studio against a training-fitted top-N list.

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
        Copy with exactly one active studio column per row. Missing and
        non-top-N primary studios map to ``studio_other``.
    """
    if company_col not in df.columns:
        raise KeyError(f"Missing production-company column: {company_col}")
    result = df.copy()
    top = list(top_studios)
    primary = result[company_col].map(primary_studio)
    for studio in top:
        result[f"{prefix}_{studio}"] = (primary == studio).astype("int8")
    result[f"{prefix}_other"] = (~primary.isin(top)).astype("int8")
    return result


def add_studio_features(
    df: pd.DataFrame,
    top_studios: Iterable[str],
    company_col: str = "production_companies",
) -> pd.DataFrame:
    """Transform ``df`` with primary studios fitted on training data."""
    return encode_studios(df, top_studios=top_studios, company_col=company_col)


class StudioEncoder:
    """Train-fitted primary-studio encoder."""

    def __init__(self, n: int = TOP_N_STUDIOS) -> None:
        self.n = n
        self.top_studios_: List[str] = []
        self.is_fitted_: bool = False

    def fit(
        self,
        train_df: pd.DataFrame,
        company_col: str = "production_companies",
    ) -> "StudioEncoder":
        self.top_studios_ = fit_top_studios(
            train_df,
            n=self.n,
            company_col=company_col,
        )
        self.is_fitted_ = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        company_col: str = "production_companies",
    ) -> pd.DataFrame:
        if not self.is_fitted_:
            raise ValueError("StudioEncoder must be fitted before transform")
        return encode_studios(
            df,
            top_studios=self.top_studios_,
            company_col=company_col,
        )
