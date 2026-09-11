import numpy as np
import pandas as pd
import pytest

from src.features.core import build_core_features


def test_core_features_extract_log_and_impute_runtime():
    frame = pd.DataFrame(
        {
            "release_date": ["2020-01-03", "2021-06-10", "2022-12-01"],
            "budget_adj": [99.0, 199.0, 0.0],
            "runtime": [90.0, None, 0.0],
        }
    )

    result = build_core_features(frame)

    assert result["release_year"].tolist() == [2020, 2021, 2022]
    assert result["log_budget"].tolist() == pytest.approx(np.log1p([99, 199, 0]))
    assert result["runtime"].tolist() == [90.0, 90.0, 90.0]
    assert result["runtime_imputed"].tolist() == [0, 1, 1]
