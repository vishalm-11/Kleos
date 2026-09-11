"""One-hot encoding for multi-label TMDB genres."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List

import pandas as pd

from config import MIN_GENRE_COUNT


def parse_genre_names(raw: object) -> List[str]:
    """Parse an asaniczka genres cell into unique genre names.

    Parameters
    ----------
    raw :
        Comma-separated string such as
        ``"Action, Adventure, Science Fiction"``.

    Returns
    -------
    list of str
        Stripped genre names, preserving source order. Empty for null/blank.
    """
    if raw is None or (not isinstance(raw, (list, tuple, set, dict)) and pd.isna(raw)):
        return []
    if not isinstance(raw, str):
        return []
    # dict.fromkeys removes accidental duplicate labels within one movie.
    return list(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))


def fit_genre_vocabulary(
    df: pd.DataFrame,
    genre_col: str = "genres",
    min_count: int = MIN_GENRE_COUNT,
) -> List[str]:
    """Learn qualifying genre labels from training rows.

    Parameters
    ----------
    df :
        Training frame only — fitting on test/backtest would leak label space
        in a minor way and makes the encoding unstable across splits.
    genre_col :
        Raw TMDB genres column.
    min_count :
        Minimum number of training movies containing the genre.

    Returns
    -------
    list of str
        Stable frequency-descending column order. Ties are alphabetical.
    """
    if genre_col not in df.columns:
        raise KeyError(f"Missing genres column: {genre_col}")
    if min_count < 1:
        raise ValueError("min_count must be at least 1")

    counts: Counter[str] = Counter()
    for raw in df[genre_col]:
        counts.update(parse_genre_names(raw))
    return [
        name
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_count
    ]


def one_hot_genres(
    df: pd.DataFrame,
    vocabulary: Iterable[str],
    genre_col: str = "genres",
    prefix: str = "genre",
) -> pd.DataFrame:
    """Add multi-label one-hot genre columns.

    A movie may belong to several genres; each vocabulary entry becomes a
    0/1 column. Genres unseen at fit time are ignored (not an "other" bin,
    unless you add one explicitly).

    Parameters
    ----------
    df :
        Frame with a raw ``genre_col``.
    vocabulary :
        Output of ``fit_genre_vocabulary``.
    genre_col :
        Source column.
    prefix :
        Column prefix, e.g. ``genre_Drama``.

    Returns
    -------
    pd.DataFrame
        Copy with one binary column per vocabulary entry. Rare/unseen genres
        are ignored rather than mapped to another bucket.
    """
    if genre_col not in df.columns:
        raise KeyError(f"Missing genres column: {genre_col}")
    result = df.copy()
    parsed = result[genre_col].map(parse_genre_names)
    for genre in list(vocabulary):
        result[f"{prefix}_{genre}"] = parsed.map(lambda names: int(genre in names)).astype("int8")
    return result


def add_genre_features(
    df: pd.DataFrame,
    vocabulary: Iterable[str],
    genre_col: str = "genres",
) -> pd.DataFrame:
    """Transform ``df`` with a genre vocabulary fitted on training data."""
    return one_hot_genres(df, vocabulary=vocabulary, genre_col=genre_col)


class GenreEncoder:
    """Train-fitted multi-label genre encoder."""

    def __init__(self, min_count: int = MIN_GENRE_COUNT) -> None:
        self.min_count = min_count
        self.vocabulary_: List[str] = []
        self.is_fitted_: bool = False

    def fit(
        self,
        train_df: pd.DataFrame,
        genre_col: str = "genres",
    ) -> "GenreEncoder":
        self.vocabulary_ = fit_genre_vocabulary(
            train_df,
            genre_col=genre_col,
            min_count=self.min_count,
        )
        self.is_fitted_ = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        genre_col: str = "genres",
    ) -> pd.DataFrame:
        if not self.is_fitted_:
            raise ValueError("GenreEncoder must be fitted before transform")
        return one_hot_genres(
            df,
            vocabulary=self.vocabulary_,
            genre_col=genre_col,
        )
