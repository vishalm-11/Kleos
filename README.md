# Kleos — Movie Box Office Success Predictor

Kleos predicts a movie's financial outcome from **pre-release information
only**—budget, cast, director, genre, release timing, franchise status, and
launch-window competition—then backtests against a held-out year of releases.

Two models share one feature pipeline:

- **Classifier:** hit versus flop, where a hit has
  `profit_multiple >= 2.0`
- **Regressor:** predicts `profit_multiple` (`revenue / budget`)

Profit multiple normalizes out budget scale, preventing the model from merely
learning “large budget, large gross.”

---

## Headline results

Box office is genuinely difficult to predict before release. The goal is to
measure how much signal is extractable from public pre-release data and to
measure it without future-data leakage.

**Held-out test set, chronological split**

| Metric | Model | Baseline |
|---|---:|---:|
| Classifier ROC-AUC | **0.72** | 0.50 |
| Classifier accuracy | 0.65 | 0.55 majority class |
| Regressor log-RMSE | 0.70 | — |
| Regressor Spearman | 0.42 | 0 |

**Out-of-time backtest—2023 held out entirely (118 films)**

| Metric | Model | Baseline |
|---|---:|---:|
| ROC-AUC | **0.70** | 0.50 |
| PR-AUC | 0.59 | 0.37 hit rate |
| Accuracy at 0.5 | 0.60 | 0.63 majority class |
| Regressor log-RMSE | 0.68 | — |
| Regressor Spearman | 0.37 | 0 |

The classifier's **ranking** is more informative than its raw accuracy. Only
37% of 2023 films are hits, so always predicting “flop” scores 63% accuracy.
An ROC-AUC near 0.70 means a random hit is scored above a random flop about 70%
of the time, even though the default operating threshold does not beat the
majority baseline.

The F1-maximizing threshold fitted on the test set is persisted rather than
re-fitted on the backtest. On 2023 it raises recall from 0.68 to 0.86 and F1
from 0.56 to 0.58, while reducing accuracy and precision. This makes the
threshold trade-off explicit rather than hiding it.

---

## What it got right and wrong in 2023

The model trained only on earlier years and scored every 2023 title after all
training and threshold decisions were fixed.

**Correctly flagged as hits:** Barbie, The Super Mario Bros. Movie,
Spider-Man: Across the Spider-Verse, Guardians of the Galaxy Vol. 3,
John Wick: Chapter 4, Scream VI, Cocaine Bear, The Nun II, Talk to Me, and
Insidious: The Red Door.

**Correctly flagged as flops:** Blue Beetle, Air, The Creator, Hypnotic,
Ruby Gillman: Teenage Kraken, Knights of the Zodiac, Strays, and 65.

**Instructive misses:**

- **Sound of Freedom:** a roughly $15M film with a 14x multiple. The model
  called it a flop, illustrating how difficult sleeper phenomena are to infer
  from structured pre-release metadata.
- **Expensive underperformers:** Mission: Impossible—Dead Reckoning,
  Indiana Jones and the Dial of Destiny, Shazam! Fury of the Gods, Haunted
  Mansion, and The Flash carried budget/franchise signals associated with
  hits but failed to clear the 2x threshold.
- **Oppenheimer:** a genuine hit scored just below the default 0.5 threshold,
  another example of why ranking and threshold analysis matter.

See `reports/backtest_2023.md` and
`reports/backtest_2023_predictions.csv` for the complete results.

---

## Approach

### Leakage-safe cast and director star power

Each film receives cast and director scores based only on their earlier
commercial performance. A naive career average leaks future outcomes: a 2015
film could accidentally use the actor's 2020 hits.

Kleos instead performs a strict chronological forward pass. A film released on
date *D* can only see films released before *D*. Cast contributions use inverse
billing-order weights (`1 / (order + 1)`) across the full cast. Actors and
directors with fewer than two prior films use the training-set median, with
explicit rookie flags. Dedicated tests enforce the strict date boundary.

### Data integrity

The source TMDB dump contains roughly 1.5 million rows and is substantially
messier than its size suggests:

- Only about 18,000 rows initially report positive budget and revenue.
- Tiny placeholder values and fan-edit records can pass a simple `> 0` check.
- Recent financial fields in this snapshot are incomplete.
- Cast, crew, and collection membership require separate API enrichment.

The ingestion pipeline therefore:

- requires `vote_count >= 10`;
- applies inflation-adjusted floors of `$1M` budget and `$10K` revenue;
- validates cached TMDB credit/detail responses;
- fills release date, runtime, collection membership, cast, and director;
- prefers positive API budget/revenue values when available; and
- uses resumable, append-only JSONL caches with rate-limit handling.

The current model dataset contains **8,497 films** through the 2023 holdout.

### Feature pipeline

- **Budget:** CPI-U adjusted to 2025 dollars and log-transformed
- **Runtime and release year**
- **Genre:** multi-label one-hot encoding fitted on training data
- **Timing:** summer, holiday, awards, dump, spring, and late-summer windows
- **Franchise:** derived from `belongs_to_collection`
- **Studio:** train-fitted top production companies plus `other`
- **Star power:** leakage-safe historical cast/director performance
- **Competition:** other wide releases within ±1 and ±2 weeks, using a
  train-fitted budget percentile as a pre-release-safe proxy

All stateful encoders follow `fit(train) / transform(other)` semantics.

### SHAP findings

The classifier's top seven features by mean absolute SHAP value are
`is_franchise`, `runtime`, `log_budget`, `cast_size`, `release_year`,
`n_rookie_cast`, and `star_power`. The regressor's top seven are
`is_franchise`, `log_budget`, `runtime`, `release_year`, `n_rookie_cast`,
`cast_size`, and `star_power`. Rankings and plots are written to `reports/`.

---

## Limitations and next steps

- **Marketing spend is missing.** Trailer reach, awareness, theater count, and
  social interest may explain demand that production metadata cannot.
- **Recent-year data is incomplete.** The raw snapshot has zero budget,
  revenue, and votes for major 2024 titles, making 2023 the newest usable
  backtest year.
- **Sleeper hits and overpriced tentpoles remain difficult.**
- **Thresholds are objective-dependent.** F1 tuning favors recall and does not
  necessarily maximize accuracy or business value.
- Producer/writer history, recency weighting, and richer release-demand signals
  are promising extensions.
- The Streamlit file is currently an inference UI scaffold, not a deployed app.

---

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Streamlit, Matplotlib, and
the TMDB API.

## Running the pipeline

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env  # set TMDB_API_KEY

# Resumable TMDB enrichment
python3 -m src.fetch_credits
python3 -m src.merge_credits
python3 -m src.fetch_details
python3 -m src.merge_details

# Feature matrices and chronological splits
python3 -m src.pipeline

# Model search, test evaluation, persisted threshold, and SHAP
python3 -m src.models.train

# Permanently held-out 2023 evaluation
python3 -m src.backtest
```

Key outputs:

- `reports/eval_test.md`
- `reports/backtest_2023.md`
- `reports/backtest_2023_predictions.csv`
- `reports/shap_classifier.png`
- `reports/shap_regressor.png`

The test suite currently contains 23 passing tests:

```bash
pytest -q
```
