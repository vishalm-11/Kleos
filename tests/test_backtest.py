import pytest

from config import (
    BACKTEST_YEAR,
    CLASSIFIER_PATH,
    CLASSIFIER_THRESHOLD_PATH,
    ENCODERS_PATH,
    MOVIES_WITH_CREDITS_PATH,
    PROCESSED_DATA_DIR,
    REGRESSOR_PATH,
)
from src.backtest import NOTABLE_COLUMNS, run_backtest


def test_backtest_smoke_writes_reports_for_50_rows(tmp_path):
    required = [
        PROCESSED_DATA_DIR / "X_backtest.parquet",
        CLASSIFIER_PATH,
        CLASSIFIER_THRESHOLD_PATH,
        REGRESSOR_PATH,
        ENCODERS_PATH,
        MOVIES_WITH_CREDITS_PATH,
    ]
    if not all(path.exists() for path in required):
        pytest.skip("trained chronological backtest artifacts are unavailable")

    notable, results = run_backtest(
        reports_dir=tmp_path,
        max_rows=50,
    )

    assert results["headline"]["n_movies"] == 50
    assert list(notable.columns) == NOTABLE_COLUMNS
    assert (tmp_path / f"backtest_{BACKTEST_YEAR}.md").exists()
    assert (tmp_path / f"backtest_{BACKTEST_YEAR}.json").exists()
    assert (tmp_path / f"backtest_{BACKTEST_YEAR}_predictions.csv").exists()
