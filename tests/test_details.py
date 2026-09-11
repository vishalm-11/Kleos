import pandas as pd

from src.fetch_details import extract_details
from src.merge_details import enrich_movies_with_details


def test_extract_details_keeps_only_required_fields():
    payload = {
        "id": 1,
        "title": "Ignored",
        "release_date": "2020-05-01",
        "belongs_to_collection": {"id": 10, "name": "Series", "poster_path": "/x"},
        "runtime": 120,
        "status": "Released",
    }

    assert extract_details(payload) == {
        "release_date": "2020-05-01",
        "belongs_to_collection": {"id": 10, "name": "Series"},
        "runtime": 120,
    }


def test_enrichment_precedence_runtime_fill_and_date_drop():
    movies = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "release_date": [None, "2019-01-01", None],
            "runtime": [0, 90, None],
        }
    )
    details = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "details_release_date": ["2020-05-01", "2019-02-02", None],
            "belongs_to_collection": [
                {"id": 10, "name": "Series"},
                None,
                None,
            ],
            "details_runtime": [120, 95, 80],
            "details_status": ["ok", "ok", "ok"],
        }
    )

    result = enrich_movies_with_details(movies, details)

    assert result["id"].tolist() == [1, 2]
    assert result["release_date"].tolist() == ["2020-05-01", "2019-02-02"]
    assert result["runtime"].tolist() == [120.0, 90.0]
    assert result.loc[result["id"] == 1, "belongs_to_collection"].iloc[0] == {
        "id": 10,
        "name": "Series",
    }
