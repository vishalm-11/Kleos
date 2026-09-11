"""Enrich the credits parquet with minimal TMDB movie details.

The output overwrites ``movies_with_credits.parquet`` atomically enough for the
local workflow: all transformations complete in memory before the parquet write.
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

from config import DETAILS_CACHE_PATH, MOVIES_WITH_CREDITS_PATH  # noqa: E402
from src.data_loading import filter_positive_financials  # noqa: E402

LOGGER = logging.getLogger(__name__)


def load_details_cache(path: Path = DETAILS_CACHE_PATH) -> pd.DataFrame:
    """Load the append-only details JSONL cache; the last row per id wins."""
    if not path.exists():
        raise FileNotFoundError(
            f"Details cache not found at {path}. Run `python -m src.fetch_details` first."
        )

    by_id: Dict[int, Dict[str, Any]] = {}
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed details-cache line %s", line_no)
                malformed += 1
                continue
            if record.get("id") is None:
                LOGGER.warning("Skipping details-cache line %s without an id", line_no)
                malformed += 1
                continue

            movie_id = int(record["id"])
            collection = record.get("belongs_to_collection")
            if isinstance(collection, dict):
                collection = {
                    "id": collection.get("id"),
                    "name": collection.get("name"),
                }
            else:
                collection = None
            by_id[movie_id] = {
                "id": movie_id,
                "details_release_date": record.get("release_date") or None,
                "belongs_to_collection": collection,
                "details_runtime": record.get("runtime"),
                "details_budget": record.get("budget"),
                "details_revenue": record.get("revenue"),
                "details_status": record.get("status", "ok"),
            }

    LOGGER.info(
        "Loaded details for %s movies from %s (%s malformed lines)",
        len(by_id),
        path,
        malformed,
    )
    return pd.DataFrame(by_id.values())


def enrich_movies_with_details(
    movies: pd.DataFrame,
    details: pd.DataFrame,
    id_col: str = "id",
) -> pd.DataFrame:
    """Join details, resolve field precedence, and drop undated movies.

    TMDB details dates take precedence over dump dates; dump dates are retained
    only when the details date is null. A positive dump runtime wins, while a
    positive details runtime fills null/zero/non-numeric dump values.
    """
    required_movies = {id_col, "release_date", "runtime", "budget", "revenue"}
    missing_movies = required_movies.difference(movies.columns)
    if missing_movies:
        raise KeyError(f"Movies frame is missing columns: {sorted(missing_movies)}")

    required_details = {
        id_col,
        "details_release_date",
        "belongs_to_collection",
        "details_runtime",
        "details_budget",
        "details_revenue",
        "details_status",
    }
    missing_details = required_details.difference(details.columns)
    if missing_details:
        raise KeyError(f"Details frame is missing columns: {sorted(missing_details)}")

    left = movies.copy()
    # Allow this operation to be safely re-run on an already enriched parquet.
    left = left.drop(
        columns=[
            "belongs_to_collection",
            "details_release_date",
            "details_runtime",
            "details_budget",
            "details_revenue",
            "details_status",
        ],
        errors="ignore",
    )
    left[id_col] = pd.to_numeric(left[id_col], errors="coerce").astype("Int64")
    right = details.copy()
    right[id_col] = pd.to_numeric(right[id_col], errors="coerce").astype("Int64")
    merged = left.merge(right, on=id_col, how="left", validate="one_to_one")

    dump_dates = merged["release_date"].replace(r"^\s*$", pd.NA, regex=True)
    detail_dates = merged["details_release_date"].replace(r"^\s*$", pd.NA, regex=True)
    dump_was_missing = dump_dates.isna()
    merged["release_date"] = detail_dates.combine_first(dump_dates)
    filled = int((dump_was_missing & merged["release_date"].notna()).sum())
    LOGGER.info(
        "Filled %s of %s missing dump release dates from TMDB details",
        filled,
        int(dump_was_missing.sum()),
    )

    dump_runtime = pd.to_numeric(merged["runtime"], errors="coerce")
    detail_runtime = pd.to_numeric(merged["details_runtime"], errors="coerce")
    runtime_needs_fill = dump_runtime.isna() | (dump_runtime <= 0)
    valid_detail_runtime = detail_runtime.where(detail_runtime > 0)
    merged["runtime"] = dump_runtime.mask(
        runtime_needs_fill,
        valid_detail_runtime,
    )
    runtime_filled = int(
        (runtime_needs_fill & valid_detail_runtime.notna()).sum()
    )
    LOGGER.info("Filled %s missing/zero runtimes from TMDB details", runtime_filled)

    for field in ("budget", "revenue"):
        dump_values = pd.to_numeric(merged[field], errors="coerce")
        detail_values = pd.to_numeric(
            merged[f"details_{field}"],
            errors="coerce",
        )
        use_detail = detail_values > 0
        merged[field] = dump_values.mask(use_detail, detail_values)
        LOGGER.info(
            "Preferred positive TMDB details %s for %s rows",
            field,
            int(use_detail.sum()),
        )

    before_drop = len(merged)
    merged = merged.loc[merged["release_date"].notna()].copy()
    dropped = before_drop - len(merged)
    LOGGER.info(
        "Dropped %s rows still missing release_date; %s rows remain",
        dropped,
        len(merged),
    )

    return merged.drop(
        columns=[
            "details_release_date",
            "details_runtime",
            "details_budget",
            "details_revenue",
        ]
    )


def write_enriched_movies(
    movies_path: Path = MOVIES_WITH_CREDITS_PATH,
    cache_path: Path = DETAILS_CACHE_PATH,
    out_path: Path = MOVIES_WITH_CREDITS_PATH,
) -> pd.DataFrame:
    """Read, enrich, and write the processed movie parquet."""
    if not movies_path.exists():
        raise FileNotFoundError(f"Movies parquet not found: {movies_path}")
    movies = pd.read_parquet(movies_path)
    details = load_details_cache(cache_path)
    enriched = enrich_movies_with_details(movies, details)
    before_financial_filter = len(enriched)
    enriched = filter_positive_financials(enriched)
    LOGGER.info(
        "After API financial precedence, dropped %s rows still lacking positive "
        "budget/revenue; %s remain",
        before_financial_filter - len(enriched),
        len(enriched),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(out_path, index=False)
    LOGGER.info("Wrote %s enriched rows to %s", len(enriched), out_path)
    return enriched


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge cached TMDB details into movies_with_credits.parquet."
    )
    parser.add_argument("--movies", type=Path, default=MOVIES_WITH_CREDITS_PATH)
    parser.add_argument("--cache", type=Path, default=DETAILS_CACHE_PATH)
    parser.add_argument("--out", type=Path, default=MOVIES_WITH_CREDITS_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    write_enriched_movies(
        movies_path=args.movies,
        cache_path=args.cache,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
