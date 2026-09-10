"""One-hot encoding for multi-label TMDB genres."""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd


def parse_genre_names(raw: object) -> List[str]:
    """Parse a TMDB genres cell into a list of genre name strings.

    Parameters
    ----------
    raw :
        Typical TMDB payload is a JSON list of ``{"id": ..., "name": ...}``
        objects, sometimes stored as a string.

    Returns
    -------
    list of str
        Genre names. Empty list if missing or unparseable.
    """
    raise NotImplementedError("Parse TMDB genres JSON/string into a list of names.")


def fit_genre_vocabulary(df: pd.DataFrame, genre_col: str = "genres") -> List[str]:
    """Learn the sorted set of genre labels from training rows.

    Parameters
    ----------
    df :
        Training frame only — fitting on test/backtest would leak label space
        in a minor way and makes the encoding unstable across splits.
    genre_col :
        Raw TMDB genres column.

    Returns
    -------
    list of str
        Stable column order for one-hot output (e.g. ``genre_Action``).
    """
    raise NotImplementedError("Return sorted unique genre names from training data.")


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
        Copy with one binary column per vocabulary entry.
    """
    raise NotImplementedError("Attach one-hot columns for each genre in vocabulary.")
