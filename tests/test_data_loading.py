import pandas as pd
import pytest

from src.data_loading import filter_adjusted_financials


def test_adjusted_financial_filter_applies_both_floors():
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "budget_adj": [1_000_000, 999_999, 2_000_000, 2_000_000],
            "revenue_adj": [10_000, 50_000, 9_999, 20_000],
        }
    )

    result = filter_adjusted_financials(frame)

    assert result["id"].tolist() == [1, 4]


def test_adjusted_financial_filter_requires_inflation_columns():
    with pytest.raises(KeyError, match="Inflation adjustment must run"):
        filter_adjusted_financials(pd.DataFrame({"budget": [1], "revenue": [2]}))
