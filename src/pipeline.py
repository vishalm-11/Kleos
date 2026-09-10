"""Orchestrate feature construction and the train / test / backtest split.

Chronological order is mandatory before any rolling or historical feature
(star power, competition windows that depend on prior context, etc.).
Call ``assert_chronologically_sorted`` after the sort; do not skip it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from config import (
    PROCESSED_FEATURES_PATH,
    TRAIN_FRACTION,
)


def sort_chronologically(
    df: pd.DataFrame,
    date_col: str = "release_date",
) -> pd.DataFrame:
    """Return ``df`` sorted by release date (stable, oldest first).

    Parameters
    ----------
    df :
        Frame with a parsed datetime column.
    date_col :
        Sort key.

    Returns
    -------
    pd.DataFrame
        Sorted copy with a reset or preserved index — document the choice.
    """
    raise NotImplementedError("Sort by release_date ascending.")


def assert_chronologically_sorted(
    df: pd.DataFrame,
    date_col: str = "release_date",
) -> None:
    """Raise if rows are not in non-decreasing release-date order.

    This is the leakage-prevention test stub: run it after
    ``sort_chronologically`` and before star-power / any look-back feature.

    Parameters
    ----------
    df :
        Candidate frame.
    date_col :
        Date column that must be monotonic non-decreasing.

    Raises
    ------
    AssertionError
        If ``date_col`` is not sorted or contains nulls.
    """
    raise NotImplementedError(
        "Assert df[date_col] is monotonic non-decreasing and has no nulls. "
        "Treat this as the required leakage guard, not optional debug code."
    )


def backtest_year(df: pd.DataFrame, date_col: str = "release_date") -> int:
    """Return the most recent calendar year present in ``df``.

    Parameters
    ----------
    df :
        Frame with parsed dates (full financially valid set, pre-split).
    date_col :
        Release date.

    Returns
    -------
    int
        Hold-out year for ``backtest.py``.
    """
    raise NotImplementedError("Return the max calendar year of release_date.")


def split_backtest_and_model_pool(
    df: pd.DataFrame,
    date_col: str = "release_date",
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Hold out the most recent year entirely; return the remaining model pool.

    Parameters
    ----------
    df :
        Full processed (or pre-feature) frame.
    date_col :
        Release date.

    Returns
    -------
    model_pool, backtest_df, holdout_year : tuple
        ``backtest_df`` is every row in ``holdout_year``.
        ``model_pool`` is every earlier year (this is what gets 75/25 split).
        ``holdout_year`` is the integer year that was peeled off.
    """
    raise NotImplementedError(
        "year = backtest_year(df); split rows by release year == year vs < year."
    )


def split_train_test(
    model_pool: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    date_col: str = "release_date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the non-backtest pool 75/25 for train and test.

    Prefer a chronological split (earliest ``train_fraction`` of dates → train,
    remainder → test) so test does not sit in the middle of the training era.
    If a random split is used instead, document why and still keep the
    backtest year completely out of both sides.

    Parameters
    ----------
    model_pool :
        Output of ``split_backtest_and_model_pool`` (no hold-out year rows).
    train_fraction :
        Default ``config.TRAIN_FRACTION`` (0.75). Test is 1 - train_fraction.
    date_col :
        Used if the split is chronological.

    Returns
    -------
    train_df, test_df : tuple
    """
    raise NotImplementedError("Split model_pool 75/25; keep the logic here, not in a notebook.")


def split_train_test_backtest(
    df: pd.DataFrame,
    date_col: str = "release_date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reusable three-way split: train, test, backtest.

    Parameters
    ----------
    df :
        Full frame (features optional; dates required).
    date_col :
        Release date.

    Returns
    -------
    train_df, test_df, backtest_df : tuple
        Backtest = most recent year. Train/test = 75/25 of the rest.
    """
    raise NotImplementedError("Compose split_backtest_and_model_pool + split_train_test.")


def build_features(
    df: pd.DataFrame,
    median_profit_multiple: Optional[float] = None,
    wide_release_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Run the full feature stack on a chronologically sorted frame.

    Suggested order (implementers should keep look-back features last):
    1. sort + ``assert_chronologically_sorted``
    2. inflation adjustment
    3. core, timing, genre, franchise, studio
    4. star power (needs median_profit_multiple; prior films only)
    5. competition (needs fitted wide-release threshold)

    Parameters
    ----------
    df :
        Financially filtered TMDB rows.
    median_profit_multiple :
        Fitted on *train* only, then reused for test/backtest scoring.
    wide_release_threshold :
        Fitted on *train* only.

    Returns
    -------
    pd.DataFrame
        Model-ready feature frame plus targets
        (``profit_multiple``, and a hit/flop label).
    """
    raise NotImplementedError("Wire feature modules here; do not bury this in a notebook.")


def run_pipeline(persist: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load → filter → split → fit constants on train → build features.

    Fit ``MEDIAN_PROFIT_MULTIPLE`` and ``WIDE_RELEASE_REVENUE_THRESHOLD`` on
    the training split only, then apply ``build_features`` to train, test,
    and backtest with those frozen values.

    Parameters
    ----------
    persist :
        If True, write processed frames under ``config.PROCESSED_DATA_DIR``
        (see ``PROCESSED_FEATURES_PATH``).

    Returns
    -------
    train_df, test_df, backtest_df : tuple
    """
    raise NotImplementedError(
        "load_and_filter -> split_train_test_backtest -> fit medians/thresholds "
        f"on train -> build_features on each split -> optionally save to {PROCESSED_FEATURES_PATH}."
    )
