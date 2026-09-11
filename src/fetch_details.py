"""Fetch minimal TMDB movie details for credit-validated titles.

Reads movie ids from ``data/processed/movies_with_credits.parquet`` and writes
one compact JSON object per movie to ``data/raw/details_cache.jsonl``. The
append-only cache is resumable: ids already present are skipped on later runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (  # noqa: E402
    DETAILS_CACHE_PATH,
    MOVIES_WITH_CREDITS_PATH,
    RAW_TMDB_PATH,
    TMDB_DETAILS_URL_TEMPLATE,
    TMDB_MAX_RETRIES,
    TMDB_PROGRESS_EVERY,
    TMDB_REQUEST_DELAY_SECONDS,
    TMDB_REQUEST_TIMEOUT_SECONDS,
)
from src.fetch_credits import (  # noqa: E402
    STATUS_NOT_FOUND,
    STATUS_OK,
    _auth_for_key,
    append_jsonl,
    load_api_key,
    load_cached_ids,
    pending_ids,
    pd_to_unique_ids,
)
from src.data_loading import load_raw_tmdb, recent_vote_qualified_rows  # noqa: E402

LOGGER = logging.getLogger(__name__)


def load_credit_validated_ids(
    movies_path: Path = MOVIES_WITH_CREDITS_PATH,
) -> List[int]:
    """Return unique ids whose cached credits status is ``ok``."""
    if not movies_path.exists():
        raise FileNotFoundError(
            f"Credits parquet not found at {movies_path}. "
            "Run `python -m src.merge_credits` first."
        )
    movies = pd.read_parquet(movies_path, columns=["id", "credits_status"])
    movies = movies.loc[movies["credits_status"] == STATUS_OK]
    ids = (
        pd.to_numeric(movies["id"], errors="coerce")
        .dropna()
        .astype("int64")
        .drop_duplicates()
        .tolist()
    )
    LOGGER.info("Loaded %s credit-validated movie ids from %s", len(ids), movies_path)
    return ids


def extract_details(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only fields required by downstream feature engineering."""
    collection = payload.get("belongs_to_collection")
    if isinstance(collection, dict):
        collection = {
            "id": collection.get("id"),
            "name": collection.get("name"),
        }
    else:
        collection = None
    return {
        "release_date": payload.get("release_date") or None,
        "belongs_to_collection": collection,
        "runtime": payload.get("runtime"),
        "budget": payload.get("budget"),
        "revenue": payload.get("revenue"),
    }


def load_supplemental_ids(movies_path: Path = RAW_TMDB_PATH) -> List[int]:
    """Load recent vote-qualified ids regardless of dump financial values."""
    raw = load_raw_tmdb(
        movies_path,
        usecols=["id", "release_date", "vote_count"],
    )
    return pd_to_unique_ids(recent_vote_qualified_rows(raw)["id"])


def fetch_movie_details(
    session: requests.Session,
    movie_id: int,
    api_key: str,
    timeout: float = TMDB_REQUEST_TIMEOUT_SECONDS,
    max_retries: int = TMDB_MAX_RETRIES,
) -> Optional[Dict[str, Any]]:
    """GET ``/movie/{id}``; return JSON, or ``None`` for a 404.

    Rate-limit responses honor ``Retry-After`` when present and otherwise use
    exponential backoff. Network errors and 5xx responses are retried.
    Authentication failures stop immediately.
    """
    url = TMDB_DETAILS_URL_TEMPLATE.format(movie_id=movie_id)
    headers, params = _auth_for_key(api_key)
    delay = 1.0

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(
                url,
                headers=headers,
                params=params,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            LOGGER.warning(
                "Network error for movie %s (attempt %s/%s): %s",
                movie_id,
                attempt,
                max_retries,
                exc,
            )
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay *= 2
            continue

        if response.status_code == 404:
            return None
        if response.status_code in (401, 403):
            raise RuntimeError(
                f"TMDB rejected the API key (HTTP {response.status_code}). "
                "Check TMDB_API_KEY in .env."
            )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            sleep_for = float(retry_after) if retry_after else delay
            LOGGER.warning("429 for movie %s; sleeping %.1fs", movie_id, sleep_for)
            time.sleep(sleep_for)
            delay *= 2
            continue
        if response.status_code >= 500:
            LOGGER.warning(
                "HTTP %s for movie %s (attempt %s/%s)",
                response.status_code,
                movie_id,
                attempt,
                max_retries,
            )
            if attempt == max_retries:
                response.raise_for_status()
            time.sleep(delay)
            delay *= 2
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError(f"Exhausted retries for movie {movie_id}")


def run_fetch(
    movies_path: Path = MOVIES_WITH_CREDITS_PATH,
    raw_movies_path: Path = RAW_TMDB_PATH,
    cache_path: Path = DETAILS_CACHE_PATH,
    delay_seconds: float = TMDB_REQUEST_DELAY_SECONDS,
    progress_every: int = TMDB_PROGRESS_EVERY,
    max_movies: Optional[int] = None,
) -> None:
    """Fetch uncached details for ids present in the credits parquet."""
    api_key = load_api_key()
    primary_ids = load_credit_validated_ids(movies_path)
    supplemental_ids = set(load_supplemental_ids(raw_movies_path))
    movie_ids = list(dict.fromkeys([*primary_ids, *sorted(supplemental_ids)]))
    target_ids = set(movie_ids)
    cached_ids = load_cached_ids(cache_path)
    todo = pending_ids(movie_ids, cached_ids)
    if max_movies is not None:
        if max_movies < 0:
            raise ValueError("max_movies must be non-negative")
        todo = todo[:max_movies]

    completed = len(cached_ids & target_ids)
    total_target = completed + len(todo)
    LOGGER.info(
        "Supplemental 2022–2024 vote-qualified target has %s ids; "
        "%s are new to the details cache before fetching",
        len(supplemental_ids),
        len(supplemental_ids - cached_ids),
    )
    LOGGER.info(
        "Details cache has %s/%s target ids; %s selected to fetch",
        completed,
        len(movie_ids),
        len(todo),
    )
    if not todo:
        LOGGER.info("Nothing to fetch.")
        return

    fetched_ok = 0
    not_found = 0
    with requests.Session() as session:
        for i, movie_id in enumerate(todo, start=1):
            payload = fetch_movie_details(session, movie_id, api_key)
            if payload is None:
                LOGGER.warning("TMDB 404 for movie id %s; caching not_found", movie_id)
                record = {
                    "id": movie_id,
                    "release_date": None,
                    "belongs_to_collection": None,
                    "runtime": None,
                    "budget": None,
                    "revenue": None,
                    "status": STATUS_NOT_FOUND,
                }
                not_found += 1
            else:
                record = {
                    "id": movie_id,
                    **extract_details(payload),
                    "status": STATUS_OK,
                }
                fetched_ok += 1

            append_jsonl(record, cache_path)
            completed += 1
            if i % progress_every == 0 or i == len(todo):
                LOGGER.info(
                    "Progress: %s/%s completed "
                    "(this run: %s ok, %s missing); %s remaining",
                    completed,
                    total_target,
                    fetched_ok,
                    not_found,
                    len(todo) - i,
                )
            if delay_seconds > 0 and i < len(todo):
                time.sleep(delay_seconds)

    LOGGER.info(
        "Done. Wrote %s new rows (%s ok, %s not found) to %s",
        len(todo),
        fetched_ok,
        not_found,
        cache_path,
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch minimal TMDB details for credit-validated movie ids."
    )
    parser.add_argument(
        "--movies",
        type=Path,
        default=MOVIES_WITH_CREDITS_PATH,
        help="Input movies_with_credits parquet",
    )
    parser.add_argument(
        "--raw-movies",
        type=Path,
        default=RAW_TMDB_PATH,
        help="Raw CSV used for the supplemental recent-year target",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DETAILS_CACHE_PATH,
        help="Append-only, resumable details JSONL cache",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=TMDB_REQUEST_DELAY_SECONDS,
        help="Seconds to sleep between requests",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=TMDB_PROGRESS_EVERY,
        help="Log progress every N fetches",
    )
    parser.add_argument(
        "--max-movies",
        type=int,
        default=None,
        help="Fetch at most this many uncached ids (smoke test)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_fetch(
        movies_path=args.movies,
        raw_movies_path=args.raw_movies,
        cache_path=args.cache,
        delay_seconds=args.delay,
        progress_every=args.progress_every,
        max_movies=args.max_movies,
    )


if __name__ == "__main__":
    main()
