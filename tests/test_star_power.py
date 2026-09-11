import pandas as pd
import pytest
import numpy as np

from src.features.star_power import (
    StarPowerEncoder,
    assert_no_future_films_in_history,
    top_n_star_power_path_b,
    weighted_star_power_path_a,
)


def _cast(*people):
    return [
        {"id": person_id, "name": name, "order": order}
        for person_id, name, order in people
    ]


def _directors(*people):
    return [{"id": person_id, "name": name} for person_id, name in people]


def test_forward_pass_scores_rookies_caps_outliers_and_blocks_leakage():
    movies = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "release_date": [
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
                "2020-03-01",  # same day as movie 3
                "2020-04-01",
                "2020-05-01",
            ],
            "profit_multiple": [2.0, 4.0, 6.0, 100.0, 14.0, 20.0],
            "cast": [
                _cast((1, "Actor A", 0)),
                _cast((1, "Actor A", 0)),
                _cast((1, "Actor A", 0), (2, "Actor B", 1)),
                _cast((1, "Actor A", 0)),
                _cast((2, "Actor B", 0)),
                _cast((1, "Actor A", 0), (2, "Actor B", 1)),
            ],
            "directors": [
                _directors((10, "Director D")),
                _directors((10, "Director D")),
                _directors((10, "Director D")),
                _directors((10, "Director D")),
                _directors((11, "Director E")),
                _directors((11, "Director E")),
            ],
        }
    )
    # The fallback is fitted from training only, independently of transform history.
    fit_frame = pd.DataFrame({"profit_multiple": [8.0, 10.0, 12.0]})
    encoder = StarPowerEncoder(
        min_prior_films=2,
        profit_multiple_cap=50.0,
        use_weighted=True,
    ).fit(fit_frame)

    result = encoder.transform(movies)

    assert encoder.median_profit_multiple_ == 10.0
    assert result.loc[1, "top1_actor_score"] == 10.0  # one prior film: rookie
    assert result.loc[2, "top1_actor_score"] == 3.0  # two prior: mean(2, 4)
    assert result.loc[2, "n_rookie_cast"] == 1
    assert result.loc[2, "frac_rookie_cast"] == 0.5
    assert result.loc[2, "star_power"] == pytest.approx(16.0 / 3.0)
    assert result.loc[2, "top3_star_power"] == pytest.approx(16.0 / 3.0)
    assert result.loc[2, "director_star_power"] == 3.0
    assert result.loc[2, "director_prior_film_count"] == 2.0
    assert result.loc[2, "is_rookie_director"] == 0

    # Movie 4 is on the same date and its 100x outcome cannot affect movie 3
    # (or vice versa): both see only January and February.
    assert result.loc[3, "top1_actor_score"] == 3.0

    assert result.loc[4, "top1_actor_score"] == 10.0  # B has only movie 3
    # By movie 6, A has four prior films; 100 is capped at 50.
    assert result.loc[5, "top1_actor_score"] == pytest.approx(15.5)
    assert result.loc[5, "n_rookie_cast"] == 0


def test_external_history_is_strictly_earlier_and_future_hit_cannot_leak():
    train = pd.DataFrame(
        {
            "id": [1, 3],
            "release_date": ["2020-01-01", "2021-01-01"],
            "profit_multiple": [2.0, 50.0],
            "cast": [_cast((1, "Actor A", 0)), _cast((1, "Actor A", 0))],
            "directors": [_directors((10, "Director D")), _directors((10, "Director D"))],
        }
    )
    target = pd.DataFrame(
        {
            "id": [2],
            "release_date": ["2020-06-01"],
            "profit_multiple": [3.0],
            "cast": [_cast((1, "Actor A", 0))],
            "directors": [_directors((10, "Director D"))],
        }
    )
    encoder = StarPowerEncoder(min_prior_films=1).fit(train)

    result = encoder.transform(target, history_df=train)

    assert result.loc[0, "star_power"] == 2.0
    assert result.loc[0, "director_star_power"] == 2.0


def test_aggregation_paths_and_explicit_history_guard():
    scores = pd.DataFrame(
        {"actor_score": [2.0, 8.0, 20.0], "cast_order": [0, 1, 2]}
    )
    assert weighted_star_power_path_a(scores) == pytest.approx(
        (2.0 + 4.0 + 20.0 / 3.0) / (1.0 + 0.5 + 1.0 / 3.0)
    )
    assert top_n_star_power_path_b(scores, n=2) == 5.0

    history = pd.DataFrame({"release_date": ["2020-01-01", "2020-03-01"]})
    with pytest.raises(AssertionError, match="History leakage"):
        assert_no_future_films_in_history(history, pd.Timestamp("2020-03-01"))


def test_parquet_style_numpy_credit_arrays_are_supported():
    movies = pd.DataFrame(
        {
            "id": [1],
            "release_date": ["2020-01-01"],
            "profit_multiple": [2.0],
            "cast": [np.array(_cast((1, "Actor A", 0)), dtype=object)],
            "directors": [
                np.array(_directors((10, "Director D")), dtype=object)
            ],
        }
    )
    encoder = StarPowerEncoder().fit(
        pd.DataFrame({"profit_multiple": [3.0, 4.0, 5.0]})
    )

    result = encoder.transform(movies)

    assert result.loc[0, "cast_size"] == 1
    assert result.loc[0, "n_rookie_cast"] == 1
    assert result.loc[0, "star_power"] == 4.0
