import pandas as pd

from src.features.genre import fit_genre_vocabulary, one_hot_genres, parse_genre_names


def test_genre_fit_and_transform_excludes_rare_and_unseen_labels():
    train = pd.DataFrame(
        {
            "genres": [
                "Action, Adventure",
                "Action, Drama",
                "Action",
                "Drama",
            ]
        }
    )
    vocabulary = fit_genre_vocabulary(train, min_count=2)
    test = pd.DataFrame({"genres": ["Action, Documentary", None]})

    result = one_hot_genres(test, vocabulary)

    assert vocabulary == ["Action", "Drama"]
    assert result["genre_Action"].tolist() == [1, 0]
    assert result["genre_Drama"].tolist() == [0, 0]
    assert "genre_Documentary" not in result
    assert parse_genre_names("Action, Adventure, Action") == ["Action", "Adventure"]
