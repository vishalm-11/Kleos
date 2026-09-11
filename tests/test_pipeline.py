import pandas as pd
import pytest

from config import BACKTEST_YEAR, MOVIES_WITH_CREDITS_PATH
from src.pipeline import run_pipeline, split_train_test


def test_pipeline_real_sample_has_clean_nonempty_splits():
    if not MOVIES_WITH_CREDITS_PATH.exists():
        pytest.skip("real enriched parquet is not available")

    movies = pd.read_parquet(MOVIES_WITH_CREDITS_PATH)
    dates = pd.to_datetime(movies["release_date"], errors="coerce")
    budgets = pd.to_numeric(movies["budget"], errors="coerce")
    revenues = pd.to_numeric(movies["revenue"], errors="coerce")
    votes = pd.to_numeric(movies["vote_count"], errors="coerce")
    eligible = movies.loc[
        dates.notna() & dates.dt.year.between(1915, BACKTEST_YEAR)
        & budgets.ge(1_000_000)
        & revenues.ge(10_000)
        & votes.ge(10)
    ].copy()
    eligible["_year"] = pd.to_datetime(eligible["release_date"]).dt.year
    historical = eligible.loc[eligible["_year"] < BACKTEST_YEAR].head(180)
    backtest = eligible.loc[eligible["_year"] == BACKTEST_YEAR].head(20)
    sample = pd.concat([historical, backtest], ignore_index=True).drop(
        columns="_year"
    )

    X_train, X_test, X_backtest = run_pipeline(
        persist=False,
        df=sample,
    )

    assert not X_train.empty
    assert not X_test.empty
    assert not X_backtest.empty
    for matrix in (X_train, X_test, X_backtest):
        assert "is_hit" not in matrix.columns
        assert "profit_multiple" not in matrix.columns
        assert not any(column.startswith("revenue") for column in matrix.columns)
        assert not matrix.isna().all().any()
    assert X_backtest["release_year"].eq(BACKTEST_YEAR).all()


def test_train_test_split_supports_seeded_random_strategy():
    frame = pd.DataFrame(
        {
            "id": range(20),
            "release_date": pd.date_range("2000-01-01", periods=20),
        }
    )
    chronological_train, _ = split_train_test(
        frame,
        split_strategy="chronological",
    )
    random_train_a, random_test_a = split_train_test(
        frame,
        split_strategy="random",
    )
    random_train_b, random_test_b = split_train_test(
        frame,
        split_strategy="random",
    )

    assert random_train_a["id"].tolist() == random_train_b["id"].tolist()
    assert random_test_a["id"].tolist() == random_test_b["id"].tolist()
    assert set(random_train_a["id"]) != set(chronological_train["id"])
    assert random_train_a["release_date"].is_monotonic_increasing
    assert random_test_a["release_date"].is_monotonic_increasing
