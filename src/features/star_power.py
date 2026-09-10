"""Actor and director historical performance with chronological (no-leakage) lookup.

***************************************************************************
CAST_ORDER RELIABILITY — READ BEFORE IMPLEMENTING THE WEIGHTED FORMULA
***************************************************************************

Do not implement the star-power aggregation until
``notebooks/eda_cast_order.ipynb`` has been used to spot-check TMDB's
``cast_order`` field against known billing on ~20 familiar movies.

Two implementation paths, selected by ``config.USE_WEIGHTED_STAR_POWER``:

Path A (USE_WEIGHTED_STAR_POWER = True)
    cast_order is a trustworthy billing rank.
    star_power = sum(actor_score_i * (1 / cast_order_i)) / sum(1 / cast_order_i)
    over the *full* billed cast. No top-N cutoff.

Path B (USE_WEIGHTED_STAR_POWER = False)  — default until the notebook says otherwise
    cast_order is noisy, missing, or alphabetical for a meaningful chunk of rows.
    star_power = unweighted mean of actor_score among the top-N names
    ordered by cast_order (N = config.STAR_POWER_TOP_N, typically 5).

***************************************************************************
COLD START
***************************************************************************

``is_rookie_actor`` / ``is_rookie_director`` are True when the person has
fewer than ``config.ROOKIE_PRIOR_FILM_THRESHOLD`` films released *strictly
before* this movie (default: < 2 prior films).

Missing historical scores (rookies, or anyone with no prior scored film)
impute to the *median* profit multiple — never the mean, never zero.
That median is fitted on the training pool
(``config.compute_median_profit_multiple``) and passed in; do not hardcode it.

***************************************************************************
LEAKAGE
***************************************************************************

Every historical average may only use films with release_date < this movie's
release_date. The pipeline must sort chronologically before calling these
helpers; ``assert_no_future_films_in_history`` is the guard.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from config import (
    MEDIAN_PROFIT_MULTIPLE,
    ROOKIE_PRIOR_FILM_THRESHOLD,
    STAR_POWER_TOP_N,
    USE_WEIGHTED_STAR_POWER,
)


def assert_no_future_films_in_history(
    history: pd.DataFrame,
    as_of_date: pd.Timestamp,
    date_col: str = "release_date",
) -> None:
    """Raise if ``history`` contains any film released on or after ``as_of_date``.

    Parameters
    ----------
    history :
        Candidate prior-film rows used to score a person.
    as_of_date :
        Release date of the movie being scored.
    date_col :
        Date column on ``history``.

    Raises
    ------
    AssertionError
        If any row has ``date_col >= as_of_date`` (same-day and future leaks).
    """
    raise NotImplementedError(
        "Assert (history[date_col] < as_of_date).all(); this is the leakage guard."
    )


def prior_films_for_person(
    df: pd.DataFrame,
    person_id: object,
    as_of_date: pd.Timestamp,
    person_id_col: str,
    date_col: str = "release_date",
) -> pd.DataFrame:
    """Return this person's films released strictly before ``as_of_date``.

    Parameters
    ----------
    df :
        Full (chronologically sorted) movie-level or credit-level frame.
    person_id :
        TMDB person id (or name if ids are unavailable — document the choice).
    as_of_date :
        Current movie's release date.
    person_id_col :
        Column identifying the person on ``df``.
    date_col :
        Release-date column.

    Returns
    -------
    pd.DataFrame
        Prior rows only. Caller should run ``assert_no_future_films_in_history``.
    """
    raise NotImplementedError("Filter to person_id and release_date < as_of_date.")


def is_rookie_actor(
    df: pd.DataFrame,
    person_id: object,
    as_of_date: pd.Timestamp,
    threshold: int = ROOKIE_PRIOR_FILM_THRESHOLD,
) -> bool:
    """True if this actor has fewer than ``threshold`` prior films in the dataset.

    Parameters
    ----------
    df :
        Credit-level or movie-level frame covering the actor's appearances.
    person_id :
        Actor identifier.
    as_of_date :
        Current movie release date; only earlier films count.
    threshold :
        Default ``config.ROOKIE_PRIOR_FILM_THRESHOLD`` (2).

    Returns
    -------
    bool
        True for cold-start actors.
    """
    raise NotImplementedError("Count prior actor credits; return count < threshold.")


def is_rookie_director(
    df: pd.DataFrame,
    person_id: object,
    as_of_date: pd.Timestamp,
    threshold: int = ROOKIE_PRIOR_FILM_THRESHOLD,
) -> bool:
    """True if this director has fewer than ``threshold`` prior films in the dataset.

    Parameters
    ----------
    df :
        Frame of directing credits.
    person_id :
        Director identifier.
    as_of_date :
        Current movie release date.
    threshold :
        Default ``config.ROOKIE_PRIOR_FILM_THRESHOLD`` (2).

    Returns
    -------
    bool
        True for cold-start directors.
    """
    raise NotImplementedError("Count prior directing credits; return count < threshold.")


def historical_profit_multiple(
    prior_films: pd.DataFrame,
    profit_col: str = "profit_multiple",
    median_fallback: Optional[float] = MEDIAN_PROFIT_MULTIPLE,
) -> float:
    """Mean profit multiple of ``prior_films``, or the training-set median if empty.

    Parameters
    ----------
    prior_films :
        Strictly earlier titles for this person.
    profit_col :
        Inflation-adjusted revenue / budget.
    median_fallback :
        Fitted median from ``config.compute_median_profit_multiple``. Must be
        provided (or already set on config) before scoring; do not hardcode.

    Returns
    -------
    float
        Historical mean, or ``median_fallback`` when there is no history.
    """
    raise NotImplementedError(
        "If prior_films is empty, return median_fallback; else mean(profit_col)."
    )


def weighted_star_power_path_a(
    cast_scores: pd.DataFrame,
    score_col: str = "actor_score",
    order_col: str = "cast_order",
) -> float:
    """Path A: inverse-cast_order weighted average over the full cast.

    star_power = sum(score_i * w_i) / sum(w_i)  where w_i = 1 / cast_order_i

    Parameters
    ----------
    cast_scores :
        One row per billed actor on *this* movie, each with a historical score
        (already imputed for rookies).
    score_col :
        Per-actor historical profit-multiple score.
    order_col :
        TMDB billing rank; smaller = higher billing. Must be >= 1.

    Returns
    -------
    float
        Weighted star-power scalar for the title.
    """
    raise NotImplementedError("Implement only after eda_cast_order.ipynb supports Path A.")


def top_n_star_power_path_b(
    cast_scores: pd.DataFrame,
    n: int = STAR_POWER_TOP_N,
    score_col: str = "actor_score",
    order_col: str = "cast_order",
) -> float:
    """Path B: unweighted mean of the top-N actors by cast_order.

    Parameters
    ----------
    cast_scores :
        Per-actor scores for this title.
    n :
        How many billed names to average (default ``config.STAR_POWER_TOP_N``).
    score_col, order_col :
        Score and billing-rank columns.

    Returns
    -------
    float
        Unweighted mean of the first ``n`` rows when sorted by ``order_col``.
    """
    raise NotImplementedError("Implement as the fallback if cast_order is unreliable.")


def add_star_power_features(
    df: pd.DataFrame,
    median_profit_multiple: Optional[float] = MEDIAN_PROFIT_MULTIPLE,
    use_weighted: bool = USE_WEIGHTED_STAR_POWER,
) -> pd.DataFrame:
    """Attach actor/director star-power scores and rookie flags to each movie.

    Parameters
    ----------
    df :
        Chronologically sorted, inflation-adjusted movie frame with cast/crew
        payloads. Pipeline must sort before calling this.
    median_profit_multiple :
        Training-set median used for cold-start imputation. Required.
    use_weighted :
        ``True`` → Path A, ``False`` → Path B. Reads
        ``config.USE_WEIGHTED_STAR_POWER`` by default.

    Returns
    -------
    pd.DataFrame
        Copy with at least:
        - ``actor_star_power``
        - ``director_star_power`` (historical mean of the director's prior films)
        - ``is_rookie_actor`` (e.g. lead / any-rookie — document the grain)
        - ``is_rookie_director``
    """
    raise NotImplementedError(
        "Expand cast/crew, score each person on prior films only, aggregate "
        "via Path A or B, impute rookies with median_profit_multiple."
    )
