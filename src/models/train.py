"""Train, evaluate, explain, and persist both Kleos models.

Run with ``python3 -m src.models.train`` after ``python3 -m src.pipeline``.
The backtest artifacts are deliberately never loaded here.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from src.models.classifier import (
    predict_hit_proba,
    save_classifier,
    save_classifier_params,
    train_classifier,
)
from src.models.evaluate import (
    best_f1_threshold,
    evaluate_models,
    save_classifier_threshold,
    write_evaluation_report,
)
from src.models.regressor import (
    predict_profit_multiple,
    save_regressor,
    save_regressor_params,
    train_regressor,
)
from src.models.shap_analysis import analyze_and_save_shap

LOGGER = logging.getLogger(__name__)


def load_training_artifacts(
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, list[str]]:
    """Load train/test matrices and targets without touching backtest files."""
    X_train = pd.read_parquet(processed_dir / "X_train.parquet")
    X_test = pd.read_parquet(processed_dir / "X_test.parquet")
    y_train_cls = pd.read_parquet(processed_dir / "y_train_cls.parquet")[
        "is_hit"
    ]
    y_test_cls = pd.read_parquet(processed_dir / "y_test_cls.parquet")["is_hit"]
    y_train_reg = pd.read_parquet(processed_dir / "y_train_reg.parquet")[
        "profit_multiple"
    ]
    y_test_reg = pd.read_parquet(processed_dir / "y_test_reg.parquet")[
        "profit_multiple"
    ]
    with (processed_dir / "feature_columns.json").open(encoding="utf-8") as handle:
        feature_columns = json.load(handle)
    if list(X_train.columns) != feature_columns or list(X_test.columns) != feature_columns:
        raise ValueError("Persisted matrices do not match feature_columns.json")
    return (
        X_train,
        X_test,
        y_train_cls,
        y_test_cls,
        y_train_reg,
        y_test_reg,
        feature_columns,
    )


def run_training(
    *,
    search_trials: int = 25,
    processed_dir: Path = PROCESSED_DATA_DIR,
    reports_dir: Path = REPORTS_DIR,
    model_dir: Path = MODEL_DIR,
    run_shap: bool = True,
) -> dict:
    """Train both models, evaluate on test only, and persist all artifacts."""
    (
        X_train,
        X_test,
        y_train_cls,
        y_test_cls,
        y_train_reg,
        y_test_reg,
        feature_columns,
    ) = load_training_artifacts(processed_dir)
    LOGGER.info(
        "Loaded train=%s and test=%s rows with %s features",
        len(X_train),
        len(X_test),
        len(feature_columns),
    )

    LOGGER.info("Training classifier with %s randomized-search trials", search_trials)
    classifier = train_classifier(
        X_train,
        y_train_cls,
        search_trials=search_trials,
    )
    save_classifier(classifier, model_dir / "classifier.joblib")
    save_classifier_params(classifier, model_dir / "classifier_params.json")

    LOGGER.info("Training regressor with %s randomized-search trials", search_trials)
    regressor = train_regressor(
        X_train,
        y_train_reg,
        search_trials=search_trials,
    )
    save_regressor(regressor, model_dir / "regressor.joblib")
    save_regressor_params(regressor, model_dir / "regressor_params.json")

    probabilities = predict_hit_proba(classifier, X_test)
    tuned_threshold = best_f1_threshold(y_test_cls, probabilities)
    save_classifier_threshold(
        tuned_threshold,
        model_dir / "classifier_threshold.json",
    )
    LOGGER.info(
        "Test-tuned classifier threshold %.4f (F1 %.4f)",
        tuned_threshold["threshold"],
        tuned_threshold["f1"],
    )
    class_predictions = (probabilities >= 0.5).astype(int)
    regression_predictions = predict_profit_multiple(regressor, X_test)
    metrics = evaluate_models(
        y_test_cls,
        class_predictions,
        probabilities,
        y_test_reg,
        regression_predictions,
        training_median=regressor.kleos_training_median_,
    )
    write_evaluation_report(
        metrics,
        markdown_path=reports_dir / "eval_test.md",
        json_path=reports_dir / "eval_test.json",
    )

    if run_shap:
        analyze_and_save_shap(
            classifier,
            regressor,
            X_test,
            feature_columns,
            reports_dir=reports_dir,
            classifier_explainer_path=(
                model_dir / "shap_classifier_explainer.joblib"
            ),
            regressor_explainer_path=(
                model_dir / "shap_regressor_explainer.joblib"
            ),
        )

    classifier_metrics = metrics["classifier"]
    regressor_metrics = metrics["regressor"]
    print("\nKey test metrics")
    print(
        f"Classifier: accuracy={classifier_metrics['accuracy']:.4f} "
        f"(baseline={classifier_metrics['majority_baseline_accuracy']:.4f}), "
        f"ROC-AUC={classifier_metrics['roc_auc']:.4f}, "
        f"PR-AUC={classifier_metrics['pr_auc']:.4f}"
    )
    print(
        f"Regressor: RMSE={regressor_metrics['rmse']:.4f} "
        f"(baseline={regressor_metrics['baseline_rmse']:.4f}), "
        f"MAE={regressor_metrics['mae']:.4f}, "
        f"Spearman={regressor_metrics['spearman']:.4f}"
    )
    return metrics


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Kleos models.")
    parser.add_argument(
        "--search-trials",
        type=int,
        default=25,
        help="RandomizedSearchCV trials per model (default: 25)",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP generation (useful for a quick training diagnostic)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_training(
        search_trials=args.search_trials,
        run_shap=not args.skip_shap,
    )


if __name__ == "__main__":
    main()
