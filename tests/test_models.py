import pandas as pd
import pytest

from config import PROCESSED_DATA_DIR
from src.models.classifier import predict_hit_proba, train_classifier
from src.models.evaluate import evaluate_models, write_evaluation_report
from src.models.regressor import predict_profit_multiple, train_regressor


def test_model_training_smoke_on_real_300_row_sample(tmp_path):
    required = [
        PROCESSED_DATA_DIR / "X_train.parquet",
        PROCESSED_DATA_DIR / "y_train_cls.parquet",
        PROCESSED_DATA_DIR / "y_train_reg.parquet",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("persisted pipeline matrices are not available")

    X = pd.read_parquet(required[0]).head(300)
    y_cls = pd.read_parquet(required[1])["is_hit"].head(300)
    y_reg = pd.read_parquet(required[2])["profit_multiple"].head(300)
    train_X, test_X = X.iloc[:240], X.iloc[240:]
    train_cls, test_cls = y_cls.iloc[:240], y_cls.iloc[240:]
    train_reg, test_reg = y_reg.iloc[:240], y_reg.iloc[240:]

    classifier = train_classifier(
        train_X,
        train_cls,
        params={
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
        },
        search_trials=0,
        early_stopping_rounds=3,
    )
    regressor = train_regressor(
        train_X,
        train_reg,
        params={
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
        },
        search_trials=0,
        early_stopping_rounds=3,
    )
    probabilities = predict_hit_proba(classifier, test_X)
    regression_predictions = predict_profit_multiple(regressor, test_X)

    assert probabilities.shape == (60,)
    assert regression_predictions.shape == (60,)

    metrics = evaluate_models(
        test_cls,
        (probabilities >= 0.5).astype(int),
        probabilities,
        test_reg,
        regression_predictions,
        training_median=float(train_reg.median()),
    )
    report_path = tmp_path / "eval_test.md"
    write_evaluation_report(
        metrics,
        markdown_path=report_path,
        json_path=tmp_path / "eval_test.json",
    )
    assert report_path.exists()
    assert "Classifier" in report_path.read_text()
