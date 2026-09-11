import pandas as pd
import pytest

from src.features.competition import CompetitionEncoder


def _movie(movie_id, date, budget, genres):
    return {
        "id": movie_id,
        "release_date": date,
        "budget_adj": budget,
        "genres": genres,
    }


def test_competition_counts_wide_movies_excludes_self_and_matches_genre():
    # The 75th percentile is exactly 100, so budget 10 is not wide.
    train = pd.DataFrame(
        {
            "budget_adj": [10.0, 100.0, 100.0, 100.0],
        }
    )
    movies = pd.DataFrame(
        [
            _movie(1, "2020-01-01", 100, "Action"),
            _movie(2, "2020-01-08", 100, "Drama"),
            _movie(3, "2020-01-15", 100, "Action, Adventure"),
            _movie(4, "2020-01-16", 100, "Action"),
            _movie(5, "2020-01-05", 10, "Action"),  # below cutoff
            _movie(6, "2019-12-18", 100, "Action"),  # exactly -14 from id=1
            _movie(7, "2019-12-17", 100, "Action"),  # -15 from id=1
            _movie(8, "2020-01-29", 100, "Drama"),  # exactly +14 from id=3
        ]
    )
    encoder = CompetitionEncoder(percentile=75).fit(train)

    result = encoder.transform(movies)

    assert encoder.fitted_threshold_ == 100.0
    first = result.loc[result["id"] == 1].iloc[0]
    assert first["n_competitors_2wk"] == 3  # ids 2, 3, 6
    assert first["n_competitors_1wk"] == 1  # id 2; low-budget id 5 excluded
    assert first["n_same_genre_competitors_2wk"] == 2  # ids 3 and 6

    third = result.loc[result["id"] == 3].iloc[0]
    assert third["n_competitors_2wk"] == 4  # ids 1, 2, 4, 8
    assert third["n_same_genre_competitors_2wk"] == 2  # ids 1 and 4


def test_boundary_14_days_included_and_15_days_excluded():
    train = pd.DataFrame({"budget_adj": [100.0]})
    pool = pd.DataFrame(
        [
            _movie(1, "2020-01-01", 100, "Drama"),
            _movie(2, "2020-01-15", 100, "Drama"),
            _movie(3, "2020-01-16", 100, "Drama"),
        ]
    )
    target = pool.loc[pool["id"] == 1]
    encoder = CompetitionEncoder().fit(train)

    result = encoder.transform(target, competitor_pool=pool)

    assert result.iloc[0]["n_competitors_2wk"] == 1
    assert result.iloc[0]["n_same_genre_competitors_2wk"] == 1


def test_transform_requires_fit():
    movies = pd.DataFrame([_movie(1, "2020-01-01", 100, "Drama")])

    with pytest.raises(ValueError, match="fitted"):
        CompetitionEncoder().transform(movies)
