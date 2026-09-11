"""Wide-release competition density in a +/- 2 week window.

"Wide release" is *not* "any title that opened the same week". It is a title
whose inflation-adjusted budget meets ``config.WIDE_RELEASE_BUDGET_THRESHOLD``,
a cutoff fitted as ``config.WIDE_RELEASE_BUDGET_PERCENTILE`` of the training
distribution (see the justification on that constant in config.py).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import (
    COMPETITION_WINDOW_DAYS,
    WIDE_RELEASE_BUDGET_PERCENTILE,
    WIDE_RELEASE_BUDGET_THRESHOLD,
    compute_wide_release_budget_threshold,
)
from src.features.genre import parse_genre_names

ONE_WEEK_DAYS = 7


def is_wide_release(
    budget_adj: float,
    threshold: Optional[float] = WIDE_RELEASE_BUDGET_THRESHOLD,
) -> bool:
    """True if inflation-adjusted budget is at or above the wide-release cutoff.

    Parameters
    ----------
    budget_adj :
        CPI-adjusted production budget, known before release.
    threshold :
        Fitted dollar cutoff from ``config.compute_wide_release_budget_threshold``.
        Must not be a magic number inlined here.

    Returns
    -------
    bool
    """
    if threshold is None or not np.isfinite(threshold):
        raise ValueError("wide-release threshold must be fitted before transform")
    if pd.isna(budget_adj):
        return False
    return float(budget_adj) >= float(threshold)


def add_wide_release_flag(
    df: pd.DataFrame,
    budget_col: str = "budget_adj",
    threshold: Optional[float] = WIDE_RELEASE_BUDGET_THRESHOLD,
    out_col: str = "is_wide_release",
) -> pd.DataFrame:
    """Mark each row as wide or not using the fitted budget threshold.

    Parameters
    ----------
    df :
        Inflation-adjusted frame.
    budget_col :
        Inflation-adjusted pre-release budget.
    threshold :
        ``config.WIDE_RELEASE_BUDGET_THRESHOLD`` after the pipeline fits it.
    out_col :
        Binary flag name.

    Returns
    -------
    pd.DataFrame
        Copy with ``out_col``.
    """
    if budget_col not in df.columns:
        raise KeyError(f"Missing budget column: {budget_col}")
    if threshold is None or not np.isfinite(threshold):
        raise ValueError("wide-release threshold must be fitted before transform")
    result = df.copy()
    budget = pd.to_numeric(result[budget_col], errors="coerce")
    result[out_col] = (budget >= float(threshold)).astype("int8")
    return result


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
    required = {date_col, wide_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Competition frame is missing columns: {sorted(missing)}")
    if row_index not in df.index:
        raise KeyError(f"Unknown row index: {row_index!r}")
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError("release_date must not contain null or invalid values")
    release_date = dates.loc[row_index]
    in_window = dates.between(
        release_date - pd.Timedelta(days=window_days),
        release_date + pd.Timedelta(days=window_days),
        inclusive="both",
    )
    wide = df[wide_col].astype(bool)
    return int((in_window & wide).sum() - int(wide.loc[row_index]))


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
    required = {"release_date", "is_wide_release"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Competition frame is missing columns: {sorted(missing)}")
    dates = pd.to_datetime(df["release_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("release_date must not contain null or invalid values")

    order = np.argsort(dates.to_numpy(dtype="datetime64[ns]"), kind="stable")
    sorted_dates = dates.to_numpy(dtype="datetime64[ns]")[order]
    sorted_wide = df["is_wide_release"].to_numpy(dtype=bool)[order]
    prefix = np.concatenate(([0], np.cumsum(sorted_wide, dtype=np.int64)))
    delta = np.timedelta64(window_days, "D")
    left = np.searchsorted(sorted_dates, sorted_dates - delta, side="left")
    right = np.searchsorted(sorted_dates, sorted_dates + delta, side="right")
    counts_sorted = prefix[right] - prefix[left] - sorted_wide.astype(np.int64)
    counts = np.empty(len(df), dtype=np.int64)
    counts[order] = counts_sorted

    result = df.copy()
    result[out_col] = counts
    return result


def _genre_set(raw: object) -> frozenset[str]:
    if isinstance(raw, str) or raw is None:
        return frozenset(parse_genre_names(raw))
    if isinstance(raw, (list, tuple, set, np.ndarray)):
        return frozenset(
            str(value).strip()
            for value in raw
            if value is not None and str(value).strip()
        )
    return frozenset()


class CompetitionEncoder:
    """Train-fitted wide-release threshold plus cross-split date-window counts.

    Fit only on training budget. For each transform, pass the concatenated
    train + test + backtest frame as ``competitor_pool`` so every movie sees
    the real launch calendar across all splits.
    """

    def __init__(
        self,
        percentile: float = WIDE_RELEASE_BUDGET_PERCENTILE,
    ) -> None:
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")
        self.percentile = float(percentile)
        self.fitted_threshold_: Optional[float] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        budget_col: str = "budget_adj",
    ) -> "CompetitionEncoder":
        """Fit the wide-release dollar cutoff on training rows only."""
        if budget_col not in train_df.columns:
            raise KeyError(f"Training frame is missing {budget_col!r}")
        budget = pd.to_numeric(train_df[budget_col], errors="coerce")
        budget = budget[np.isfinite(budget)]
        if budget.empty:
            raise ValueError("Cannot fit competition threshold without finite budget")
        self.fitted_threshold_ = compute_wide_release_budget_threshold(
            train_df.loc[budget.index],
            budget_col=budget_col,
            percentile=self.percentile,
        )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        competitor_pool: Optional[pd.DataFrame] = None,
        *,
        id_col: str = "id",
        date_col: str = "release_date",
        budget_col: str = "budget_adj",
        genre_col: str = "genres",
    ) -> pd.DataFrame:
        """Attach ±7/14-day competition counts.

        ``competitor_pool`` defaults to ``df``. For split frames, explicitly
        pass the full dataset so test/backtest movies count titles in every
        split. A movie is excluded from its own counts by ``id``.
        """
        if self.fitted_threshold_ is None:
            raise ValueError("CompetitionEncoder must be fitted before transform")
        pool = df if competitor_pool is None else competitor_pool
        target_required = {id_col, date_col, genre_col}
        pool_required = target_required | {budget_col}
        target_missing = target_required.difference(df.columns)
        pool_missing = pool_required.difference(pool.columns)
        if target_missing:
            raise KeyError(f"Transform frame is missing columns: {sorted(target_missing)}")
        if pool_missing:
            raise KeyError(f"Competitor pool is missing columns: {sorted(pool_missing)}")

        target_dates = pd.to_datetime(df[date_col], errors="coerce")
        pool_dates = pd.to_datetime(pool[date_col], errors="coerce")
        if target_dates.isna().any():
            bad = target_dates.index[target_dates.isna()].tolist()[:10]
            raise ValueError(f"Invalid target release dates at rows: {bad}")
        if pool_dates.isna().any():
            bad = pool_dates.index[pool_dates.isna()].tolist()[:10]
            raise ValueError(f"Invalid competitor release dates at rows: {bad}")
        if pool[id_col].duplicated().any():
            duplicates = pool.loc[pool[id_col].duplicated(), id_col].tolist()[:5]
            raise ValueError(f"Competitor pool contains duplicate movie ids: {duplicates}")

        order = np.argsort(
            pool_dates.to_numpy(dtype="datetime64[ns]"),
            kind="stable",
        )
        sorted_dates = pool_dates.to_numpy(dtype="datetime64[ns]")[order]
        sorted_ids = pool[id_col].to_numpy()[order]
        sorted_budget = pd.to_numeric(
            pool[budget_col], errors="coerce"
        ).to_numpy()[order]
        sorted_genres = [
            _genre_set(value) for value in pool[genre_col].to_numpy()[order]
        ]
        sorted_wide = np.isfinite(sorted_budget) & (
            sorted_budget >= self.fitted_threshold_
        )
        wide_positions = np.flatnonzero(sorted_wide)
        wide_dates = sorted_dates[wide_positions]
        wide_ids = sorted_ids[wide_positions]
        wide_genres = [sorted_genres[position] for position in wide_positions]
        wide_id_set = set(wide_ids.tolist())

        target_date_array = target_dates.to_numpy(dtype="datetime64[ns]")
        target_ids = df[id_col].to_numpy()
        target_genres = [_genre_set(value) for value in df[genre_col]]

        left_14 = np.searchsorted(
            wide_dates,
            target_date_array - np.timedelta64(COMPETITION_WINDOW_DAYS, "D"),
            side="left",
        )
        right_14 = np.searchsorted(
            wide_dates,
            target_date_array + np.timedelta64(COMPETITION_WINDOW_DAYS, "D"),
            side="right",
        )
        left_7 = np.searchsorted(
            wide_dates,
            target_date_array - np.timedelta64(ONE_WEEK_DAYS, "D"),
            side="left",
        )
        right_7 = np.searchsorted(
            wide_dates,
            target_date_array + np.timedelta64(ONE_WEEK_DAYS, "D"),
            side="right",
        )

        self_is_wide = np.fromiter(
            (movie_id in wide_id_set for movie_id in target_ids),
            dtype=bool,
            count=len(df),
        )
        counts_14 = right_14 - left_14 - self_is_wide.astype(np.int64)
        counts_7 = right_7 - left_7 - self_is_wide.astype(np.int64)

        same_genre = np.zeros(len(df), dtype=np.int64)
        for i, genres in enumerate(target_genres):
            if not genres:
                continue
            movie_id = target_ids[i]
            same_genre[i] = sum(
                competitor_id != movie_id and bool(genres.intersection(other_genres))
                for competitor_id, other_genres in zip(
                    wide_ids[left_14[i] : right_14[i]],
                    wide_genres[left_14[i] : right_14[i]],
                )
            )

        result = df.copy()
        result["n_competitors_2wk"] = counts_14
        result["n_competitors_1wk"] = counts_7
        result["n_same_genre_competitors_2wk"] = same_genre
        return result

    def transform_splits(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        backtest_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Transform all splits against one shared competitor pool."""
        pool = pd.concat([train_df, test_df, backtest_df], ignore_index=True)
        return (
            self.transform(train_df, competitor_pool=pool),
            self.transform(test_df, competitor_pool=pool),
            self.transform(backtest_df, competitor_pool=pool),
        )
