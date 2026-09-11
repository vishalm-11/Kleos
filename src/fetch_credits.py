"""Fetch TMDB /movie/{id}/credits for financially valid titles.

Writes one JSON object per movie to ``data/raw/credits_cache.jsonl`` so a
crash mid-run keeps completed rows. Re-running skips ids already in the cache.

This module only caches raw credits. Star-power aggregation lives in
``src/features/star_power.py`` and is not implemented here.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import requests
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (  # noqa: E402
    CREDITS_CACHE_PATH,
    RAW_TMDB_PATH,
    TMDB_CREDITS_URL_TEMPLATE,
    TMDB_MAX_RETRIES,
    TMDB_PROGRESS_EVERY,
    TMDB_REQUEST_DELAY_SECONDS,
    TMDB_REQUEST_TIMEOUT_SECONDS,
)
from src.data_loading import (  # noqa: E402
    enrichment_candidate_rows,
    filter_min_vote_count,
    filter_positive_financials,
    load_raw_tmdb,
    recent_vote_qualified_rows,
)

LOGGER = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_NOT_FOUND = "not_found"


def load_api_key() -> str:
    """Return ``TMDB_API_KEY`` from the environment (after loading ``.env``)."""
    load_dotenv(_PROJECT_ROOT / ".env")
    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "TMDB_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return api_key


def _auth_for_key(api_key: str) -> tuple[Dict[str, str], Dict[str, str]]:
    """v4 JWT → Bearer header; otherwise v3 ``api_key`` query param."""
    if api_key.startswith("eyJ"):
        return {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}, {}
    return {"Accept": "application/json"}, {"api_key": api_key}


def extract_credits(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Pull full cast (id, name, order) and crew members with job == Director.

    Parameters
    ----------
    payload :
        JSON body from ``GET /movie/{id}/credits``.

    Returns
    -------
    dict
        ``{"cast": [...], "directors": [...]}``. Directors keep ``id`` and ``name``.
    """
    cast = [
        {"id": member.get("id"), "name": member.get("name"), "order": member.get("order")}
        for member in payload.get("cast") or []
    ]
    directors = [
        {"id": member.get("id"), "name": member.get("name")}
        for member in payload.get("crew") or []
        if member.get("job") == "Director"
    ]
    return {"cast": cast, "directors": directors}


def load_cached_ids(path: Path = CREDITS_CACHE_PATH) -> Set[int]:
    """Movie ids already written to the JSONL cache (skip on resume)."""
    cached: Set[int] = set()
    if not path.exists():
        return cached
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed cache line %s", line_no)
                continue
            movie_id = record.get("id")
            if movie_id is None:
                continue
            cached.add(int(movie_id))
    return cached


def append_jsonl(record: Dict[str, Any], path: Path = CREDITS_CACHE_PATH) -> None:
    """Append one movie record and fsync so a crash does not lose the line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def financially_valid_movie_ids(movies_path: Path = RAW_TMDB_PATH) -> List[int]:
    """Unique vote-qualified ids with budget > 0 and revenue > 0."""
    raw = load_raw_tmdb(
        movies_path,
        usecols=["id", "budget", "revenue", "vote_count"],
    )
    n_raw = len(raw)
    voted = filter_min_vote_count(raw)
    filtered = filter_positive_financials(voted)
    ids = (
        pd_to_unique_ids(filtered["id"])
    )
    LOGGER.info(
        "Loaded %s raw rows; %s with budget>0 and revenue>0 (%s unique ids)",
        n_raw,
        len(filtered),
        len(ids),
    )
    return ids


def fetch_target_movie_ids(
    movies_path: Path = RAW_TMDB_PATH,
) -> tuple[List[int], Set[int]]:
    """Return union fetch ids and the supplemental recent-year id set."""
    raw = load_raw_tmdb(
        movies_path,
        usecols=[
            "id",
            "budget",
            "revenue",
            "release_date",
            "vote_count",
        ],
    )
    candidates = enrichment_candidate_rows(raw)
    supplemental = set(pd_to_unique_ids(recent_vote_qualified_rows(raw)["id"]))
    return pd_to_unique_ids(candidates["id"]), supplemental


def pd_to_unique_ids(series) -> List[int]:
    """Coerce an id column to unique ints, dropping nulls."""
    return (
        series.dropna()
        .astype("int64")
        .drop_duplicates()
        .tolist()
    )


def fetch_movie_credits(
    session: requests.Session,
    movie_id: int,
    api_key: str,
    timeout: float = TMDB_REQUEST_TIMEOUT_SECONDS,
    max_retries: int = TMDB_MAX_RETRIES,
) -> Optional[Dict[str, Any]]:
    """GET /movie/{id}/credits. Return JSON, or None on 404.

    429s sleep on ``Retry-After`` (or exponential backoff). Persistent 401/403
    raise immediately so a bad key does not hammer the API.
    """
    url = TMDB_CREDITS_URL_TEMPLATE.format(movie_id=movie_id)
    headers, params = _auth_for_key(api_key)
    delay = 1.0

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, headers=headers, params=params, timeout=timeout)
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


def pending_ids(movie_ids: Iterable[int], cached_ids: Set[int]) -> List[int]:
    """Ids still missing from the cache, preserving input order."""
    return [movie_id for movie_id in movie_ids if movie_id not in cached_ids]


def run_fetch(
    movies_path: Path = RAW_TMDB_PATH,
    cache_path: Path = CREDITS_CACHE_PATH,
    delay_seconds: float = TMDB_REQUEST_DELAY_SECONDS,
    progress_every: int = TMDB_PROGRESS_EVERY,
    max_movies: Optional[int] = None,
) -> None:
    """Fetch credits for filtered movie ids that are not already cached."""
    api_key = load_api_key()
    movie_ids, supplemental_ids = fetch_target_movie_ids(movies_path)
    cached_ids = load_cached_ids(cache_path)
    todo = pending_ids(movie_ids, cached_ids)
    if max_movies is not None:
        todo = todo[:max_movies]

    LOGGER.info(
        "Supplemental 2022–2024 vote-qualified target has %s ids; "
        "%s are new to the credits cache before fetching",
        len(supplemental_ids),
        len(supplemental_ids - cached_ids),
    )
    LOGGER.info(
        "Cache has %s ids; %s remaining to fetch",
        len(cached_ids),
        len(todo),
    )
    if not todo:
        LOGGER.info("Nothing to fetch.")
        return

    session = requests.Session()
    completed = len(cached_ids & set(movie_ids))
    total_target = completed + len(todo)
    fetched_ok = 0
    not_found = 0

    for i, movie_id in enumerate(todo, start=1):
        payload = fetch_movie_credits(session, movie_id, api_key)
        if payload is None:
            LOGGER.warning("TMDB 404 for movie id %s; skipping", movie_id)
            record = {"id": movie_id, "cast": [], "directors": [], "status": STATUS_NOT_FOUND}
            not_found += 1
        else:
            extracted = extract_credits(payload)
            record = {"id": movie_id, "status": STATUS_OK, **extracted}
            fetched_ok += 1

        append_jsonl(record, cache_path)
        completed += 1

        if i % progress_every == 0 or i == len(todo):
            LOGGER.info(
                "Progress: %s/%s completed (this run: %s ok, %s missing); %s remaining",
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
        description="Fetch TMDB cast/crew credits for movies with budget>0 and revenue>0."
    )
    parser.add_argument(
        "--movies",
        type=Path,
        default=RAW_TMDB_PATH,
        help="Path to tmdb_movies.csv",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=CREDITS_CACHE_PATH,
        help="JSONL cache path (appended, resumable)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=TMDB_REQUEST_DELAY_SECONDS,
        help="Seconds to sleep between successful requests",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=TMDB_PROGRESS_EVERY,
        help="Log a progress line every N fetches",
    )
    parser.add_argument(
        "--max-movies",
        type=int,
        default=None,
        help="Fetch at most this many uncached ids (for a smoke test)",
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
        cache_path=args.cache,
        delay_seconds=args.delay,
        progress_every=args.progress_every,
        max_movies=args.max_movies,
    )


if __name__ == "__main__":
    main()
