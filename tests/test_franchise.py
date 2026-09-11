import pandas as pd

from src.features.franchise import add_belongs_to_collection


def test_franchise_flag_handles_null_blank_and_collection_values():
    frame = pd.DataFrame(
        {"belongs_to_collection": [None, "", "Toy Story Collection", "{}", {"id": 1}]}
    )

    result = add_belongs_to_collection(frame)

    assert result["is_franchise"].tolist() == [0, 0, 1, 0, 1]
