import pandas as pd
import pytest

from config import INFLATION_BASE_YEAR
from src.inflation import apply_inflation_adjustment, load_cpi_index


def test_inflation_adjustment_and_profit_multiple():
    frame = pd.DataFrame(
        {
            "release_year": [2024, INFLATION_BASE_YEAR],
            "budget": [100.0, 50.0],
            "revenue": [250.0, 100.0],
        }
    )

    result = apply_inflation_adjustment(frame)
    cpi = load_cpi_index()

    assert result.loc[0, "budget_adj"] == pytest.approx(
        100.0 * cpi.loc[INFLATION_BASE_YEAR] / cpi.loc[2024]
    )
    assert result["profit_multiple"].tolist() == pytest.approx([2.5, 2.0])
    assert result["budget"].tolist() == [100.0, 50.0]


def test_inflation_rejects_year_outside_table():
    frame = pd.DataFrame({"release_year": [1914], "budget": [1], "revenue": [2]})

    with pytest.raises(ValueError, match="outside CPI table range"):
        apply_inflation_adjustment(frame)
