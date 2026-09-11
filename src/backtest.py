"""Evaluate chronological models on the permanently held-out release year."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from config import (
    BACKTEST_YEAR,
    CLASSIFIER_PATH,
    CLASSIFIER_THRESHOLD_PATH,
    ENCODERS_PATH,
    MOVIES_WITH_CREDITS_PATH,
    PROCESSED_DATA_DIR,
    PROFIT_MULTIPLE_CAP,
    REGRESSOR_PATH,
    REPORTS_DIR,
)
from src.features.core import add_release_year
from src.inflation import apply_inflation_adjustment
from src.models.classifier import load_classifier, predict_hit_proba
from src.models.evaluate import classification_metrics, evaluate_models
from src.models.regressor import load_regressor, predict_profit_multiple

LOGGER = logging.getLogger(__name__)
NOTABLE_COLUMNS = [
    "title",
    "budget_adj_m",
    "actual_profit_multiple",
    "predicted_profit_multiple",
    "actual_hit",
    "hit_probability",
    "predicted_hit",
    "correct",
]


def load_backtest_inputs(
    processed_dir: Path = PROCESSED_DATA_DIR,
    enriched_path: Path = MOVIES_WITH_CREDITS_PATH,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Load aligned backtest matrices, targets, metadata, and display data."""
    X = pd.read_parquet(processed_dir / "X_backtest.parquet")
    y_cls = pd.read_parquet(processed_dir / "y_backtest_cls.parquet")["is_hit"]
    y_reg = pd.read_parquet(processed_dir / "y_backtest_reg.parquet")[
        "profit_multiple"
    ]
    metadata = pd.read_parquet(processed_dir / "movie_ids_backtest.parquet")
    enriched = pd.read_parquet(enriched_path)
    if len({len(X), len(y_cls), len(y_reg), len(metadata)}) != 1:
        raise ValueError("Backtest matrix, targets, and metadata are misaligned")
    return X, y_cls, y_reg, metadata, enriched


def validate_chronological_artifacts(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    encoders: Dict[str, Any],
    year: int,
) -> None:
    """Reject random-split or stale-year artifacts before scoring."""
    if encoders.get("split_strategy") != "chronological":
        raise ValueError("Backtest requires chronological fitted encoders")
    if int(encoders.get("backtest_year", -1)) != year:
        raise ValueError("Encoder backtest year does not match BACKTEST_YEAR")
    if not pd.to_numeric(metadata["release_year"], errors="coerce").eq(year).all():
        raise ValueError(f"Backtest metadata contains rows outside {year}")
    if encoders.get("feature_columns") != list(X.columns):
        raise ValueError("Backtest columns do not match fitted encoder columns")


def build_prediction_frame(
    classifier,
    regressor,
    X: pd.DataFrame,
    y_cls: pd.Series,
    y_reg: pd.Series,
    metadata: pd.DataFrame,
    enriched: pd.DataFrame,
    tuned_threshold: float,
) -> pd.DataFrame:
    """Score models and attach title/date/financial display fields by movie ID."""
    probabilities = predict_hit_proba(classifier, X).reset_index(drop=True)
    reg_predictions = predict_profit_multiple(regressor, X).reset_index(drop=True)
    result = metadata.reset_index(drop=True).copy()
    result["actual_hit"] = pd.Series(y_cls).reset_index(drop=True).astype(int)
    result["hit_probability"] = probabilities
    result["predicted_hit"] = (probabilities >= 0.5).astype(int)
    result["correct"] = result["actual_hit"].eq(result["predicted_hit"])
    result["predicted_hit_tuned"] = (
        probabilities >= tuned_threshold
    ).astype(int)
    result["correct_tuned"] = result["actual_hit"].eq(
        result["predicted_hit_tuned"]
    )
    result["actual_profit_multiple"] = pd.Series(y_reg).reset_index(drop=True)
    result["predicted_profit_multiple"] = reg_predictions

    details = enriched.loc[
        enriched["id"].isin(result["movie_id"]),
        ["id", "title", "release_date", "budget", "revenue"],
    ].copy()
    details["release_date"] = pd.to_datetime(details["release_date"], errors="coerce")
    details = apply_inflation_adjustment(add_release_year(details))
    details = details.rename(
        columns={
            "id": "movie_id",
            "title": "enriched_title",
            "release_date": "enriched_release_date",
        }
    )
    result = result.merge(
        details[
            [
                "movie_id",
                "enriched_title",
                "enriched_release_date",
                "budget",
                "revenue",
                "budget_adj",
                "revenue_adj",
            ]
        ],
        on="movie_id",
        how="left",
        validate="one_to_one",
    )
    if result["budget_adj"].isna().any():
        missing = result.loc[result["budget_adj"].isna(), "movie_id"].tolist()
        raise ValueError(f"Missing enriched rows for movie ids: {missing[:10]}")
    result["title"] = result["enriched_title"].fillna(result["title"])
    result["release_date"] = result["enriched_release_date"].fillna(
        result["release_date"]
    )
    return result.drop(columns=["enriched_title", "enriched_release_date"])


def calibration_quintiles(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize predicted versus actual multiples by predicted quintile."""
    buckets = pd.qcut(
        predictions["predicted_profit_multiple"],
        q=5,
        duplicates="drop",
    )
    result = (
        predictions.assign(_bucket=buckets)
        .groupby("_bucket", observed=True)
        .agg(
            n_movies=("movie_id", "size"),
            mean_predicted=("predicted_profit_multiple", "mean"),
            mean_actual=("actual_profit_multiple", "mean"),
        )
        .reset_index(drop=True)
    )
    result.insert(0, "quintile", np.arange(1, len(result) + 1))
    return result


def notable_releases(predictions: pd.DataFrame) -> pd.DataFrame:
    """Select high-budget titles plus the five largest hits and flops."""
    selected = pd.concat(
        [
            predictions.nlargest(10, "budget_adj"),
            predictions.nlargest(5, "actual_profit_multiple"),
            predictions.loc[predictions["actual_hit"].eq(0)].nsmallest(
                5,
                "actual_profit_multiple",
            ),
        ]
    )
    selected = (
        selected.drop_duplicates("movie_id")
        .sort_values("budget_adj", ascending=False)
        .copy()
    )
    selected["budget_adj_m"] = selected["budget_adj"] / 1_000_000
    return selected.loc[:, NOTABLE_COLUMNS].reset_index(drop=True)


def biggest_misses(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return high-confidence classifier misses and largest log-space errors."""
    wrong = predictions.loc[~predictions["correct"]].copy()
    wrong["wrong_confidence"] = np.where(
        wrong["actual_hit"].eq(1),
        1 - wrong["hit_probability"],
        wrong["hit_probability"],
    )
    classifier_misses = wrong.nlargest(5, "wrong_confidence")[
        [
            "title",
            "actual_hit",
            "predicted_hit",
            "hit_probability",
            "actual_profit_multiple",
        ]
    ].reset_index(drop=True)

    regression = predictions.copy()
    actual_log = np.log1p(
        regression["actual_profit_multiple"].clip(0, PROFIT_MULTIPLE_CAP)
    )
    predicted_log = np.log1p(
        regression["predicted_profit_multiple"].clip(lower=0)
    )
    regression["absolute_log_error"] = (actual_log - predicted_log).abs()
    regression_misses = regression.nlargest(5, "absolute_log_error")[
        [
            "title",
            "actual_profit_multiple",
            "predicted_profit_multiple",
            "absolute_log_error",
        ]
    ].reset_index(drop=True)
    return classifier_misses, regression_misses


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _markdown_rows(frame: pd.DataFrame, formats: Dict[str, str]) -> str:
    rows = []
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if column in formats:
                value = formats[column].format(value)
            elif isinstance(value, (bool, np.bool_)):
                value = "yes" if value else "no"
            values.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_backtest_report(
    results: Dict[str, Any],
    notable: pd.DataFrame,
    classifier_misses: pd.DataFrame,
    regression_misses: pd.DataFrame,
    reports_dir: Path,
    year: int,
) -> tuple[Path, Path]:
    """Write the human-readable and machine-readable backtest reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = reports_dir / f"backtest_{year}.md"
    json_path = reports_dir / f"backtest_{year}.json"
    classifier = results["classifier"]
    tuned = results["classifier_tuned_threshold"]
    regressor = results["regressor"]
    tn, fp = classifier["confusion_matrix"][0]
    fn, tp = classifier["confusion_matrix"][1]
    threshold_rows = _markdown_rows(
        pd.DataFrame(classifier["threshold_tradeoffs"]),
        {
            "threshold": "{:.1f}",
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1": "{:.4f}",
            "predicted_positive_rate": "{:.4f}",
        },
    )
    calibration_rows = _markdown_rows(
        pd.DataFrame(results["calibration"]),
        {"mean_predicted": "{:.3f}", "mean_actual": "{:.3f}"},
    )
    notable_rows = _markdown_rows(
        notable,
        {
            "budget_adj_m": "${:.1f}M",
            "actual_profit_multiple": "{:.2f}",
            "predicted_profit_multiple": "{:.2f}",
            "hit_probability": "{:.3f}",
        },
    )
    classifier_miss_rows = _markdown_rows(
        classifier_misses,
        {
            "hit_probability": "{:.3f}",
            "actual_profit_multiple": "{:.2f}",
        },
    )
    regression_miss_rows = _markdown_rows(
        regression_misses,
        {
            "actual_profit_multiple": "{:.2f}",
            "predicted_profit_multiple": "{:.2f}",
            "absolute_log_error": "{:.3f}",
        },
    )
    report = f"""# Kleos {year} backtest

## Headline
- Movies: {results['headline']['n_movies']}
- Actual hit rate: {results['headline']['actual_hit_rate']:.2%}
- Majority-class baseline accuracy: {results['headline']['majority_baseline_accuracy']:.4f}

## Classifier
- Default threshold: 0.5000
- Accuracy: {classifier['accuracy']:.4f}
- ROC-AUC: {classifier['roc_auc']:.4f}
- PR-AUC: {classifier['pr_auc']:.4f}
- Precision / recall / F1 at 0.5: {classifier['precision']:.4f} / {classifier['recall']:.4f} / {classifier['f1']:.4f}
- Confusion matrix (`[[TN, FP], [FN, TP]]`): `{classifier['confusion_matrix']}`

Of the {fn + tp} actual hits, the model flagged {tp} in advance. Of the
{tn + fp} actual flops, it correctly called {tn}.

### Test-tuned operating threshold
- Threshold loaded from training artifact: {tuned['threshold']:.4f}
- Accuracy: {tuned['accuracy']:.4f}
- Precision: {tuned['precision']:.4f}
- Recall: {tuned['recall']:.4f}
- F1: {tuned['f1']:.4f}
- Confusion matrix: `{tuned['confusion_matrix']}`

| threshold | precision | recall | f1 | predicted_positive_rate |
|---:|---:|---:|---:|---:|
{threshold_rows}

## Regressor
- Log-RMSE: {regressor['rmse_log']:.4f}
- Spearman: {regressor['spearman']:.4f}
- MAE: {regressor['mae']:.4f}
- Training-median baseline: {regressor['baseline_training_median']:.4f}
- Baseline RMSE / model RMSE: {regressor['baseline_rmse']:.4f} / {regressor['rmse']:.4f}
- Baseline MAE / model MAE: {regressor['baseline_mae']:.4f} / {regressor['mae']:.4f}

### Calibration by predicted quintile
| quintile | n_movies | mean_predicted | mean_actual |
|---:|---:|---:|---:|
{calibration_rows}

## Notable releases
| title | budget_adj_m | actual_profit_multiple | predicted_profit_multiple | actual_hit | hit_probability | predicted_hit | correct |
|---|---:|---:|---:|---:|---:|---:|---:|
{notable_rows}

## Biggest classifier misses
| title | actual_hit | predicted_hit | hit_probability | actual_profit_multiple |
|---|---:|---:|---:|---:|
{classifier_miss_rows}

## Biggest regressor misses
| title | actual_profit_multiple | predicted_profit_multiple | absolute_log_error |
|---|---:|---:|---:|
{regression_miss_rows}
"""
    markdown_path.write_text(report, encoding="utf-8")
    payload = {
        **results,
        "notable_releases": _records(notable),
        "biggest_classifier_misses": _records(classifier_misses),
        "biggest_regressor_misses": _records(regression_misses),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return markdown_path, json_path


def run_backtest(
    processed_dir: Path = PROCESSED_DATA_DIR,
    reports_dir: Path = REPORTS_DIR,
    enriched_path: Path = MOVIES_WITH_CREDITS_PATH,
    classifier_path: Path = CLASSIFIER_PATH,
    threshold_path: Path = CLASSIFIER_THRESHOLD_PATH,
    regressor_path: Path = REGRESSOR_PATH,
    encoders_path: Path = ENCODERS_PATH,
    year: int = BACKTEST_YEAR,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Score chronological models and persist held-out-year artifacts."""
    X, y_cls, y_reg, metadata, enriched = load_backtest_inputs(
        processed_dir,
        enriched_path,
    )
    classifier = load_classifier(classifier_path)
    regressor = load_regressor(regressor_path)
    encoders = joblib.load(encoders_path)
    with threshold_path.open(encoding="utf-8") as handle:
        threshold_artifact = json.load(handle)
    tuned_threshold = float(threshold_artifact["threshold"])
    if not 0 <= tuned_threshold <= 1:
        raise ValueError("Persisted classifier threshold must be between 0 and 1")
    validate_chronological_artifacts(X, metadata, encoders, year)
    if max_rows is not None:
        X = X.iloc[:max_rows].reset_index(drop=True)
        y_cls = y_cls.iloc[:max_rows].reset_index(drop=True)
        y_reg = y_reg.iloc[:max_rows].reset_index(drop=True)
        metadata = metadata.iloc[:max_rows].reset_index(drop=True)

    predictions = build_prediction_frame(
        classifier,
        regressor,
        X,
        y_cls,
        y_reg,
        metadata,
        enriched,
        tuned_threshold,
    )
    metrics = evaluate_models(
        predictions["actual_hit"],
        predictions["predicted_hit"],
        predictions["hit_probability"],
        predictions["actual_profit_multiple"],
        predictions["predicted_profit_multiple"],
        training_median=regressor.kleos_training_median_,
    )
    tuned_predictions = predictions["predicted_hit_tuned"]
    tuned_metrics = classification_metrics(
        predictions["actual_hit"],
        tuned_predictions,
        predictions["hit_probability"],
    )
    calibration = calibration_quintiles(predictions)
    results = {
        "year": year,
        "headline": {
            "n_movies": len(predictions),
            "actual_hit_rate": float(predictions["actual_hit"].mean()),
            "majority_baseline_accuracy": metrics["classifier"][
                "majority_baseline_accuracy"
            ],
        },
        **metrics,
        "classifier_tuned_threshold": {
            "threshold": tuned_threshold,
            "accuracy": tuned_metrics["accuracy"],
            "precision": tuned_metrics["precision"],
            "recall": tuned_metrics["recall"],
            "f1": tuned_metrics["f1"],
            "confusion_matrix": tuned_metrics["confusion_matrix"],
        },
        "calibration": _records(calibration),
    }
    notable = notable_releases(predictions)
    classifier_misses, regression_misses = biggest_misses(predictions)
    markdown_path, json_path = write_backtest_report(
        results,
        notable,
        classifier_misses,
        regression_misses,
        reports_dir,
        year,
    )
    predictions_path = reports_dir / f"backtest_{year}_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    LOGGER.info("Wrote %s, %s, and %s", markdown_path, json_path, predictions_path)
    return notable, results


def compare_to_test_split(
    backtest_metrics: Dict[str, Dict[str, Any]],
    test_metrics: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Build a numeric test-versus-backtest distribution-shift comparison."""
    rows = []
    for model_name in ("classifier", "regressor"):
        shared = set(backtest_metrics[model_name]) & set(test_metrics[model_name])
        for metric in sorted(shared):
            backtest_value = backtest_metrics[model_name][metric]
            test_value = test_metrics[model_name][metric]
            if isinstance(backtest_value, (int, float)) and isinstance(
                test_value,
                (int, float),
            ):
                rows.append(
                    {
                        "model": model_name,
                        "metric": metric,
                        "test": test_value,
                        "backtest": backtest_value,
                        "delta": backtest_value - test_value,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    notable, results = run_backtest()
    classifier = results["classifier"]
    tuned = results["classifier_tuned_threshold"]
    regressor = results["regressor"]
    print(
        f"{BACKTEST_YEAR} backtest: {results['headline']['n_movies']} movies, "
        f"hit rate={results['headline']['actual_hit_rate']:.2%}"
    )
    print(
        f"Classifier accuracy={classifier['accuracy']:.4f}, "
        f"ROC-AUC={classifier['roc_auc']:.4f}; "
        f"regressor log-RMSE={regressor['rmse_log']:.4f}, "
        f"Spearman={regressor['spearman']:.4f}"
    )
    print(
        f"Tuned threshold={tuned['threshold']:.4f}: "
        f"accuracy={tuned['accuracy']:.4f}, "
        f"precision={tuned['precision']:.4f}, recall={tuned['recall']:.4f}"
    )
    print(f"Notable release rows: {len(notable)}")


if __name__ == "__main__":
    main()
