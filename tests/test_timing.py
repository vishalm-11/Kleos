import pandas as pd

from src.features.timing import add_release_timing, month_to_bucket


def test_release_window_buckets_and_one_hot_columns():
    frame = pd.DataFrame(
        {"release_date": ["2024-01-15", "2024-06-01", "2024-08-20", "2024-12-01"]}
    )

    result = add_release_timing(frame)

    assert result["release_window"].tolist() == [
        "dump",
        "summer",
        "late_summer",
        "holiday",
    ]
    assert result.loc[1, "release_window_summer"] == 1
    assert result.loc[1, "release_window_holiday"] == 0
    assert month_to_bucket(10) == "awards"
    assert "release_window_spring" in result
