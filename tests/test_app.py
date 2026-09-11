import numpy as np
import pytest
import requests

from app.streamlit_app import load_artifacts, predict_movie
from config import CLASSIFIER_PATH, ENCODERS_PATH, REGRESSOR_PATH


def test_single_movie_inference_is_finite_and_offline(monkeypatch):
    if not all(path.exists() for path in (CLASSIFIER_PATH, REGRESSOR_PATH, ENCODERS_PATH)):
        pytest.skip("trained inference artifacts are unavailable")

    def reject_network(*args, **kwargs):
        raise AssertionError("Inference must not access the network")

    monkeypatch.setattr(requests.sessions.Session, "request", reject_network)
    artifacts = load_artifacts()
    raw = {
        "budget": 25_000_000,
        "runtime": 105,
        "release_month": 10,
        "release_year": 2025,
        "genres": ["Drama", "Thriller"],
        "is_franchise": False,
        "studio": "Other",
        "cast_names": ["Unknown Smoke-Test Actor"],
        "director_name": "Unknown Smoke-Test Director",
        "n_competitors_2wk": artifacts["default_competitors"],
    }

    result = predict_movie(raw, artifacts)

    assert 0 <= result["hit_probability"] <= 1
    assert np.isfinite(result["profit_multiple"])
    assert result["X"].shape == (1, len(artifacts["feature_columns"]))
    assert result["unknown_actors"] == ["Unknown Smoke-Test Actor"]
    assert result["unknown_directors"] == ["Unknown Smoke-Test Director"]
