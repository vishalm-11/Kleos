import pandas as pd

from src.features.studio import encode_studios, fit_top_studios, primary_studio


def test_studio_fit_uses_primary_company_and_transform_maps_other():
    train = pd.DataFrame(
        {
            "production_companies": [
                "Warner Bros., Syncopy",
                "Warner Bros.",
                "A24, Plan B",
                "Universal Pictures",
            ]
        }
    )
    top = fit_top_studios(train, n=2)
    test = pd.DataFrame(
        {"production_companies": ["Warner Bros., Other Co", "Paramount", None]}
    )

    result = encode_studios(test, top)

    assert top == ["Warner Bros.", "A24"]
    assert result["studio_Warner Bros."].tolist() == [1, 0, 0]
    assert result["studio_A24"].tolist() == [0, 0, 0]
    assert result["studio_other"].tolist() == [0, 1, 1]
    assert primary_studio("A24, Plan B") == "A24"
