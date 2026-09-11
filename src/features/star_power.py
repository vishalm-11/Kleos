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
    star_power = sum(actor_score_i * (1 / (cast_order_i + 1))) / sum(weight_i)
    over the *full* billed cast. No top-N cutoff.

Path B (USE_WEIGHTED_STAR_POWER = False)
    cast_order is noisy, missing, or alphabetical for a meaningful chunk of rows.
    star_power = unweighted mean of actor_score among the top-N names
    ordered by cast_order (N = config.STAR_POWER_TOP_N, typically 5).

***************************************************************************
COLD START
***************************************************************************

Actors and directors are rookies when the person has
fewer than ``config.MIN_PRIOR_FILMS`` films released *strictly
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

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    MEDIAN_PROFIT_MULTIPLE,
    MIN_PRIOR_FILMS,
    PROFIT_MULTIPLE_CAP,
    STAR_POWER_TOP_N,
    USE_WEIGHTED_STAR_POWER,
)


FEATURE_COLUMNS = [
    "star_power",
    "actor_star_power",
    "top1_actor_score",
    "top3_star_power",
    "cast_size",
    "n_rookie_cast",
    "frac_rookie_cast",
    "director_star_power",
    "is_rookie_director",
    "director_prior_film_count",
]


def _require_finite_median(value: Optional[float]) -> float:
    if value is None or not np.isfinite(value):
        raise ValueError(
            "median_profit_multiple must be fitted on training data before transform"
        )
    return float(value)


def _credit_ids(raw: object) -> set[object]:
    if not isinstance(raw, (list, tuple, np.ndarray)):
        return set()
    return {
        credit.get("id")
        for credit in raw
        if isinstance(credit, dict) and credit.get("id") is not None
    }


def _is_rookie_from_credit_lists(
    df: pd.DataFrame,
    person_id: object,
    as_of_date: pd.Timestamp,
    credit_col: str,
    threshold: int,
    date_col: str = "release_date",
) -> bool:
    required = {credit_col, date_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"History frame is missing columns: {sorted(missing)}")
    dates = pd.to_datetime(df[date_col], errors="coerce")
    prior = df.loc[dates < pd.Timestamp(as_of_date)]
    count = sum(person_id in _credit_ids(raw) for raw in prior[credit_col])
    return count < threshold


def _normalise_cast(raw: object) -> list[dict]:
    """Return cast entries with usable zero-based orders in stable order."""
    if not isinstance(raw, (list, tuple, np.ndarray)):
        return []
    cast: list[dict] = []
    for position, member in enumerate(raw):
        if not isinstance(member, dict):
            continue
        order = member.get("order")
        try:
            order = int(order)
        except (TypeError, ValueError):
            order = position
        if order < 0:
            order = position
        cast.append(
            {
                "id": member.get("id"),
                "name": member.get("name"),
                "cast_order": order,
                "_position": position,
            }
        )
    return sorted(cast, key=lambda member: (member["cast_order"], member["_position"]))


def _normalise_directors(raw: object) -> list[dict]:
    if not isinstance(raw, (list, tuple, np.ndarray)):
        return []
    result: list[dict] = []
    seen: set[object] = set()
    for member in raw:
        if not isinstance(member, dict):
            continue
        person_id = member.get("id")
        if person_id is None or person_id in seen:
            continue
        seen.add(person_id)
        result.append({"id": person_id, "name": member.get("name")})
    return result


@dataclass
class _RunningHistory:
    total: float = 0.0
    count: int = 0
    last_release_date: Optional[pd.Timestamp] = None

    def score(
        self,
        as_of_date: pd.Timestamp,
        median: float,
        min_prior_films: int,
    ) -> Tuple[float, bool]:
        if self.last_release_date is not None:
            assert self.last_release_date < as_of_date, (
                "History leakage: running state contains a same-day or future film"
            )
        rookie = self.count < min_prior_films
        return (median if rookie else self.total / self.count), rookie

    def update(self, capped_profit_multiple: float, release_date: pd.Timestamp) -> None:
        self.total += capped_profit_multiple
        self.count += 1
        self.last_release_date = release_date


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
    if date_col not in history.columns:
        raise KeyError(f"History is missing date column {date_col!r}")
    dates = pd.to_datetime(history[date_col], errors="coerce")
    if dates.isna().any():
        raise AssertionError("Historical release dates must not be null")
    as_of = pd.Timestamp(as_of_date)
    invalid = history.loc[dates >= as_of]
    assert invalid.empty, (
        f"History leakage: {len(invalid)} film(s) have {date_col} >= {as_of.date()}"
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
    required = {person_id_col, date_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"History frame is missing columns: {sorted(missing)}")
    dates = pd.to_datetime(df[date_col], errors="coerce")
    result = df.loc[
        (df[person_id_col] == person_id) & (dates < pd.Timestamp(as_of_date))
    ].copy()
    assert_no_future_films_in_history(result, pd.Timestamp(as_of_date), date_col)
    return result


def is_rookie_actor(
    df: pd.DataFrame,
    person_id: object,
    as_of_date: pd.Timestamp,
    threshold: int = MIN_PRIOR_FILMS,
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
        Default ``config.MIN_PRIOR_FILMS`` (2).

    Returns
    -------
    bool
        True for cold-start actors.
    """
    return _is_rookie_from_credit_lists(
        df, person_id, as_of_date, credit_col="cast", threshold=threshold
    )


def is_rookie_director(
    df: pd.DataFrame,
    person_id: object,
    as_of_date: pd.Timestamp,
    threshold: int = MIN_PRIOR_FILMS,
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
        Default ``config.MIN_PRIOR_FILMS`` (2).

    Returns
    -------
    bool
        True for cold-start directors.
    """
    return _is_rookie_from_credit_lists(
        df, person_id, as_of_date, credit_col="directors", threshold=threshold
    )


def historical_profit_multiple(
    prior_films: pd.DataFrame,
    profit_col: str = "profit_multiple",
    median_fallback: Optional[float] = MEDIAN_PROFIT_MULTIPLE,
    cap: float = PROFIT_MULTIPLE_CAP,
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
    fallback = _require_finite_median(median_fallback)
    if profit_col not in prior_films.columns:
        raise KeyError(f"History is missing profit column {profit_col!r}")
    values = pd.to_numeric(prior_films[profit_col], errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return fallback
    return float(values.clip(upper=cap).mean())


def weighted_star_power_path_a(
    cast_scores: pd.DataFrame,
    score_col: str = "actor_score",
    order_col: str = "cast_order",
) -> float:
    """Path A: inverse-cast_order weighted average over the full cast.

    star_power = sum(score_i * w_i) / sum(w_i), where
    ``w_i = 1 / (cast_order_i + 1)`` because TMDB order is zero-based.

    Parameters
    ----------
    cast_scores :
        One row per billed actor on *this* movie, each with a historical score
        (already imputed for rookies).
    score_col :
        Per-actor historical profit-multiple score.
    order_col :
        TMDB billing rank; smaller = higher billing. Must be >= 0.

    Returns
    -------
    float
        Weighted star-power scalar for the title.
    """
    if cast_scores.empty:
        raise ValueError("Cannot aggregate an empty cast")
    scores = pd.to_numeric(cast_scores[score_col], errors="raise").to_numpy(float)
    orders = pd.to_numeric(cast_scores[order_col], errors="raise").to_numpy(float)
    if np.any(orders < 0):
        raise ValueError("cast_order must be non-negative")
    weights = 1.0 / (orders + 1.0)
    return float(np.average(scores, weights=weights))


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
    if cast_scores.empty:
        raise ValueError("Cannot aggregate an empty cast")
    top = cast_scores.sort_values(order_col, kind="stable").head(n)
    return float(pd.to_numeric(top[score_col], errors="raise").mean())


class StarPowerEncoder:
    """Fit the train-only fallback median and compute chronological features.

    ``transform`` performs one date-batched forward pass with dictionary-backed
    running sums/counts. When scoring test or backtest separately, provide all
    earlier eligible movies through ``history_df``. Alternatively,
    ``transform_splits`` handles the standard train/test/backtest sequence.
    """

    def __init__(
        self,
        min_prior_films: int = MIN_PRIOR_FILMS,
        profit_multiple_cap: float = PROFIT_MULTIPLE_CAP,
        use_weighted: bool = USE_WEIGHTED_STAR_POWER,
        top_n: int = STAR_POWER_TOP_N,
    ) -> None:
        if min_prior_films < 1:
            raise ValueError("min_prior_films must be at least 1")
        if not np.isfinite(profit_multiple_cap) or profit_multiple_cap <= 0:
            raise ValueError("profit_multiple_cap must be positive and finite")
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        self.min_prior_films = int(min_prior_films)
        self.profit_multiple_cap = float(profit_multiple_cap)
        self.use_weighted = bool(use_weighted)
        self.top_n = int(top_n)
        self.median_profit_multiple_: Optional[float] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        profit_col: str = "profit_multiple",
    ) -> "StarPowerEncoder":
        """Fit the cold-start median from training outcomes only."""
        if profit_col not in train_df.columns:
            raise KeyError(f"Training frame is missing {profit_col!r}")
        values = pd.to_numeric(train_df[profit_col], errors="coerce")
        values = values[np.isfinite(values)]
        if values.empty:
            raise ValueError("Cannot fit star power without finite training outcomes")
        self.median_profit_multiple_ = float(values.median())
        return self

    def transform(
        self,
        df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
        *,
        date_col: str = "release_date",
        profit_col: str = "profit_multiple",
        cast_col: str = "cast",
        directors_col: str = "directors",
        id_col: str = "id",
    ) -> pd.DataFrame:
        """Add star-power features using only films strictly before each row.

        Parameters
        ----------
        df :
            Movies to return features for.
        history_df :
            Optional additional movies whose outcomes may become history. Rows
            on or after a target movie's date cannot leak because updates are
            date-batched. Do not include the same movie in both frames.
        """
        median = _require_finite_median(self.median_profit_multiple_)
        required = {date_col, profit_col, cast_col, directors_col}
        missing = required.difference(df.columns)
        if missing:
            raise KeyError(f"Transform frame is missing columns: {sorted(missing)}")

        target = df.copy()
        target["_sp_target_position"] = np.arange(len(target), dtype=int)
        target["_sp_is_target"] = True

        frames = []
        if history_df is not None:
            history_missing = required.difference(history_df.columns)
            if history_missing:
                raise KeyError(
                    f"History frame is missing columns: {sorted(history_missing)}"
                )
            if id_col in target.columns and id_col in history_df.columns:
                overlap = set(target[id_col].dropna()).intersection(
                    history_df[id_col].dropna()
                )
                if overlap:
                    sample = list(overlap)[:5]
                    raise ValueError(
                        f"history_df overlaps target movies (sample ids: {sample})"
                    )
            history = history_df.copy()
            history["_sp_target_position"] = -1
            history["_sp_is_target"] = False
            frames.append(history)
        frames.append(target)

        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined["_sp_release_date"] = pd.to_datetime(
            combined[date_col], errors="coerce"
        )
        if combined["_sp_release_date"].isna().any():
            bad = combined.index[combined["_sp_release_date"].isna()].tolist()[:10]
            raise ValueError(f"Invalid or missing release dates at rows: {bad}")
        combined = combined.sort_values("_sp_release_date", kind="stable")

        actor_history: Dict[object, _RunningHistory] = {}
        director_history: Dict[object, _RunningHistory] = {}
        output: Dict[int, Dict[str, float]] = {}

        records = combined.to_dict("records")
        start = 0
        while start < len(records):
            release_date = records[start]["_sp_release_date"]
            end = start + 1
            while (
                end < len(records)
                and records[end]["_sp_release_date"] == release_date
            ):
                end += 1
            date_batch = records[start:end]

            # Score every movie on this date before updating any history from
            # this date. This is what enforces strict release_date < D.
            for row in date_batch:
                if not row["_sp_is_target"]:
                    continue
                cast = _normalise_cast(row.get(cast_col))
                cast_rows = []
                rookie_count = 0
                for member in cast:
                    person_id = member["id"]
                    state = (
                        actor_history.get(person_id, _RunningHistory())
                        if person_id is not None
                        else _RunningHistory()
                    )
                    actor_score, rookie = state.score(
                        release_date,
                        median,
                        self.min_prior_films,
                    )
                    rookie_count += int(rookie)
                    cast_rows.append(
                        {
                            "actor_score": actor_score,
                            "cast_order": member["cast_order"],
                        }
                    )

                cast_scores = pd.DataFrame(
                    cast_rows, columns=["actor_score", "cast_order"]
                )
                if cast_scores.empty:
                    star_power = median
                    top1_score = median
                    top3_power = median
                else:
                    if self.use_weighted:
                        star_power = weighted_star_power_path_a(cast_scores)
                    else:
                        star_power = top_n_star_power_path_b(
                            cast_scores, n=self.top_n
                        )
                    top1 = cast_scores.loc[cast_scores["cast_order"] == 0]
                    top1_score = (
                        float(top1.iloc[0]["actor_score"])
                        if not top1.empty
                        else float(cast_scores.iloc[0]["actor_score"])
                    )
                    top3 = cast_scores.loc[cast_scores["cast_order"].between(0, 2)]
                    top3_power = (
                        weighted_star_power_path_a(top3)
                        if not top3.empty
                        else median
                    )

                directors = _normalise_directors(row.get(directors_col))
                director_scores = []
                director_counts = []
                director_rookies = []
                for director in directors:
                    state = director_history.get(director["id"], _RunningHistory())
                    score, rookie = state.score(
                        release_date,
                        median,
                        self.min_prior_films,
                    )
                    director_scores.append(score)
                    director_counts.append(state.count)
                    director_rookies.append(rookie)

                output[int(row["_sp_target_position"])] = {
                    "star_power": star_power,
                    "actor_star_power": star_power,
                    "top1_actor_score": top1_score,
                    "top3_star_power": top3_power,
                    "cast_size": len(cast),
                    "n_rookie_cast": rookie_count,
                    "frac_rookie_cast": (
                        rookie_count / len(cast) if cast else 0.0
                    ),
                    "director_star_power": (
                        float(np.mean(director_scores))
                        if director_scores
                        else median
                    ),
                    "is_rookie_director": (
                        int(any(director_rookies)) if directors else 1
                    ),
                    # Multiple directors use the mean, matching aggregation of
                    # their scores. This is an integer for the common one-director case.
                    "director_prior_film_count": (
                        float(np.mean(director_counts)) if directors else 0.0
                    ),
                }

            # Only now can outcomes from this date enter running histories.
            for row in date_batch:
                value = pd.to_numeric(
                    pd.Series([row.get(profit_col)]), errors="coerce"
                ).iloc[0]
                if pd.isna(value) or not np.isfinite(value):
                    continue
                capped = min(float(value), self.profit_multiple_cap)
                for person_id in _credit_ids(row.get(cast_col)):
                    actor_history.setdefault(person_id, _RunningHistory()).update(
                        capped, release_date
                    )
                for director in _normalise_directors(row.get(directors_col)):
                    director_history.setdefault(
                        director["id"], _RunningHistory()
                    ).update(capped, release_date)

            start = end

        result = df.copy()
        for column in FEATURE_COLUMNS:
            result[column] = [output[position][column] for position in range(len(df))]
        for column in (
            "cast_size",
            "n_rookie_cast",
            "is_rookie_director",
        ):
            result[column] = result[column].astype("int64")
        return result

    def transform_splits(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        backtest_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Transform standard chronological splits with all eligible history.

        Training movies build history for later training movies; test movies
        see train plus earlier test releases; backtest movies see train/test
        plus earlier backtest releases.
        """
        train_result = self.transform(train_df)
        test_result = self.transform(test_df, history_df=train_df)
        pre_backtest = pd.concat([train_df, test_df], ignore_index=True, sort=False)
        backtest_result = self.transform(backtest_df, history_df=pre_backtest)
        return train_result, test_result, backtest_result


def add_star_power_features(
    df: pd.DataFrame,
    median_profit_multiple: Optional[float] = MEDIAN_PROFIT_MULTIPLE,
    use_weighted: bool = USE_WEIGHTED_STAR_POWER,
    history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Attach actor/director star-power scores and rookie flags to each movie.

    Parameters
    ----------
    df :
        Inflation-adjusted movie frame with cast/crew payloads. The encoder
        performs its own stable chronological sort.
    median_profit_multiple :
        Training-set median used for cold-start imputation. Required.
    use_weighted :
        ``True`` → Path A, ``False`` → Path B. Reads
        ``config.USE_WEIGHTED_STAR_POWER`` by default.
    history_df :
        Optional earlier movie pool, such as train when transforming test.

    Returns
    -------
    pd.DataFrame
        Copy with at least:
        - ``star_power`` / ``actor_star_power``
        - ``top1_actor_score``, ``top3_star_power``
        - ``cast_size``, ``n_rookie_cast``, ``frac_rookie_cast``
        - ``director_star_power`` (historical mean of the director's prior films)
        - ``is_rookie_director``, ``director_prior_film_count``
    """
    encoder = StarPowerEncoder(use_weighted=use_weighted)
    encoder.median_profit_multiple_ = _require_finite_median(
        median_profit_multiple
    )
    return encoder.transform(df, history_df=history_df)
