"""Join cached TMDB credits onto the financially filtered movie frame.

Reads ``data/raw/credits_cache.jsonl`` and ``data/raw/tmdb_movies.csv``,
keeps budget > 0 and revenue > 0, drops rows whose credits fetch was not
``ok`` (TMDB-deleted spam/fake titles with untrustworthy financials), and
writes ``data/processed/movies_with_credits.parquet`` for ``pipeline.py``.

Does not engineer star-power features — that stays in ``src/features/star_power.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (  # noqa: E402
    CREDITS_CACHE_PATH,
    MOVIES_WITH_CREDITS_PATH,
    RAW_TMDB_PATH,
)
from src.data_loading import (  # noqa: E402
    enrichment_candidate_rows,
    filter_min_vote_count,
    load_raw_tmdb,
)

LOGGER = logging.getLogger(__name__)


def load_credits_cache(path: Path = CREDITS_CACHE_PATH) -> pd.DataFrame:
    """Load the JSONL credits cache into a DataFrame keyed by movie ``id``.

    Parameters
    ----------
    path :
        ``credits_cache.jsonl`` from ``src/fetch_credits.py``.

    Returns
    -------
    pd.DataFrame
        Columns: ``id``, ``cast``, ``directors``, ``credits_status``.
        ``cast`` / ``directors`` are object columns holding Python lists.
        If the same id appears more than once, the last line wins.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Credits cache not found at {path}. Run `python -m src.fetch_credits` first."
        )

    by_id: Dict[int, Dict[str, Any]] = {}
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed cache line %s", line_no)
                skipped += 1
                continue
            movie_id = record.get("id")
            if movie_id is None:
                skipped += 1
                continue
            by_id[int(movie_id)] = {
                "id": int(movie_id),
                "cast": record.get("cast") or [],
                "directors": record.get("directors") or [],
                "credits_status": record.get("status", "ok"),
            }

    LOGGER.info("Loaded credits for %s movies from %s (%s bad lines)", len(by_id), path, skipped)
    return pd.DataFrame(by_id.values())


def merge_credits(
    movies: pd.DataFrame,
    credits: pd.DataFrame,
    id_col: str = "id",
) -> pd.DataFrame:
    """Left-join cached credits onto the filtered movie frame.

    Parameters
    ----------
    movies :
        Financially filtered TMDB rows.
    credits :
        Output of ``load_credits_cache``.
    id_col :
        TMDB movie id column on both frames.

    Returns
    -------
    pd.DataFrame
        ``movies`` plus ``cast``, ``directors``, ``credits_status``.
        Rows with no cache hit keep null credit columns. Caller should
        drop non-``ok`` statuses before persisting (see
        ``drop_untrusted_credits``).
    """
    if id_col not in movies.columns:
        raise KeyError(f"Movies frame is missing {id_col!r}")
    movies = movies.copy()
    movies[id_col] = pd.to_numeric(movies[id_col], errors="coerce").astype("Int64")
    credits = credits.copy()
    credits[id_col] = pd.to_numeric(credits[id_col], errors="coerce").astype("Int64")
    merged = movies.merge(credits, on=id_col, how="left")
    return merged


def drop_untrusted_credits(merged: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose credits fetch was not ``ok``.

    A non-ok status (404 / missing cache row) means TMDB has deleted the
    title — typically spam or fake entries — so reported budget and
    revenue are not trustworthy either.
    """
    before = len(merged)
    kept = merged.loc[merged["credits_status"] == "ok"].copy()
    dropped = before - len(kept)
    LOGGER.info(
        "Dropped %s rows with credits_status != 'ok' "
        "(TMDB-deleted / untrusted financials); %s remain",
        dropped,
        len(kept),
    )
    return kept


def write_movies_with_credits(
    movies_path: Path = RAW_TMDB_PATH,
    cache_path: Path = CREDITS_CACHE_PATH,
    out_path: Path = MOVIES_WITH_CREDITS_PATH,
) -> pd.DataFrame:
    """Select enrichment targets, join credits, and write the trusted rows."""
    movies = load_raw_tmdb(movies_path)
    n_raw = len(movies)
    vote_valid = filter_min_vote_count(movies)
    LOGGER.info(
        "Global vote_count filter removed %s rows below MIN_VOTE_COUNT; %s remain",
        n_raw - len(vote_valid),
        len(vote_valid),
    )
    movies = enrichment_candidate_rows(vote_valid)
    movies = movies.drop_duplicates(subset=["id"], keep="first")
    LOGGER.info(
        "Selected %s unique ids from financial and supplemental fetch targets",
        len(movies),
    )

    credits = load_credits_cache(cache_path)
    merged = merge_credits(movies, credits)

    n_matched = merged["credits_status"].notna().sum()
    n_ok = (merged["credits_status"] == "ok").sum()
    LOGGER.info(
        "Joined credits onto %s movies (%s cache hits, %s status=ok)",
        len(merged),
        int(n_matched),
        int(n_ok),
    )
    if n_matched < len(merged):
        LOGGER.warning(
            "%s filtered movies have no cache row — re-run fetch_credits",
            int(len(merged) - n_matched),
        )

    merged = drop_untrusted_credits(merged)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    LOGGER.info("Wrote %s rows to %s", len(merged), out_path)
    return merged


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join credits_cache.jsonl onto filtered movies and write parquet."
    )
    parser.add_argument("--movies", type=Path, default=RAW_TMDB_PATH)
    parser.add_argument("--cache", type=Path, default=CREDITS_CACHE_PATH)
    parser.add_argument("--out", type=Path, default=MOVIES_WITH_CREDITS_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    write_movies_with_credits(
        movies_path=args.movies,
        cache_path=args.cache,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
