# Kleos

Movie success predictor: hit/flop classification and profit-multiple regression from **pre-release** features, trained on the TMDB Movies Dataset (900k+ titles, not TMDB 5000) and backtested on the most recent calendar year of releases.

This repo is a **scaffold**. Modules expose function signatures and docstrings only. Implement them one at a time, starting with `src/data_loading.py` and `notebooks/eda_cast_order.ipynb`.

## Layout

```
kleos/
  data/raw/                 # untouched TMDB download
  data/processed/           # cleaned + feature-engineered output
  src/data_loading.py       # load raw data; keep budget>0 and revenue>0
  src/inflation.py          # CPI-adjust budget/revenue by release year
  src/features/             # one concern per file (see below)
  src/pipeline.py           # feature build + chronological sort + splits
  src/models/               # XGBoost classifier/regressor, metrics, SHAP
  src/backtest.py           # held-out most-recent-year evaluation
  app/streamlit_app.py      # form + dual predictions + SHAP
  notebooks/eda_cast_order.ipynb
  config.py
```

## Architectural constraints (already stubbed)

**Cast order.** Do not write the star-power formula until the EDA notebook checks TMDB `cast_order` against ~20 movies you know. `config.USE_WEIGHTED_STAR_POWER` switches:

- `True` — Path A: inverse-`cast_order` weights over the full cast
- `False` — Path B: unweighted mean of the top 5 billed names (default)

**Cold start.** `is_rookie_actor` / `is_rookie_director` are true when a person has fewer than 2 prior films in the dataset at this title's release. Missing historical scores impute to the **training-set median** profit multiple (`config.compute_median_profit_multiple`), not the mean and not zero.

**Leakage.** Actor/director history may only use films with `release_date` strictly before the current movie. `pipeline.sort_chronologically` + `assert_chronologically_sorted` must run before look-back features. `star_power.assert_no_future_films_in_history` is the per-lookup guard.

**Wide release.** Competition density counts titles in a +/- 2 week window that clear a **fitted inflation-adjusted budget percentile**, not a raw same-week title count. Budget is available before release; revenue would leak outcomes. Threshold and justification live on `config.WIDE_RELEASE_BUDGET_PERCENTILE`.

After CPI adjustment, modelling keeps movies with `budget_adj >= $1,000,000`
and `revenue_adj >= $10,000`. The current chronological model's train-fitted
75th-percentile budget cutoff is **$59,935,429.26** (2025 dollars). Runtime
code uses `CompetitionEncoder.fitted_threshold_`, not this README diagnostic.

**Splits.** The configured 2021 release year—the latest year with at least 150
qualifying movies—is held out entirely. Remaining
rows use the 75/25 `SPLIT_STRATEGY` (`chronological` or seeded `random`) via
`pipeline.split_train_test_backtest`. Randomly assigned split frames are still
date-sorted before historical feature calculation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the TMDB dump in `data/raw/` and point `config.RAW_TMDB_PATH` at the file.

## Credits fetch

asaniczka's dump has no cast/crew. After the CSV is in `data/raw/`:

```bash
cp .env.example .env   # set TMDB_API_KEY (v3 key or v4 JWT)
python -m src.fetch_credits          # resumable; writes data/raw/credits_cache.jsonl
python -m src.merge_credits          # writes data/processed/movies_with_credits.parquet
python -m src.fetch_details          # resumable; writes data/raw/details_cache.jsonl
python -m src.merge_details          # enriches the same parquet in place
```

The details pass fills release date, collection membership, and missing runtime
from TMDB. `pipeline.py` consumes the enriched parquet, not the raw CSV.
`--max-movies N` on either fetch command is a smoke test.

## Feature pipeline

```bash
python3 -m src.pipeline
```

This writes `X_{train,test,backtest}.parquet`, classification/regression targets,
movie-id metadata, and `feature_columns.json` under `data/processed/`, plus the
fitted inference encoders at `models/encoders.joblib`.

## Model training

```bash
python3 -m src.models.train
```

This tunes and saves the classifier/regressor, evaluates only the test split,
and writes evaluation and SHAP artifacts under `reports/`. The backtest split
remains untouched for `src/backtest.py`.

## Suggested implement order

1. `src/data_loading.py` (load + financial filter are in place)
2. TMDB enrichment: credits fetch/merge → details fetch/merge
3. `notebooks/eda_cast_order.ipynb` → set `USE_WEIGHTED_STAR_POWER`
4. `src/inflation.py`, then `src/features/*`
5. `src/pipeline.py` (sort, asserts, splits, orchestration)
6. `src/models/*`, `src/backtest.py`
7. `app/streamlit_app.py`
