# Split-strategy comparison

Both runs use the post-inflation floors `budget_adj >= $1,000,000` and
`revenue_adj >= $10,000`, 25 randomized-search trials per model, and the same
2025 backtest exclusion.

| Strategy | Accuracy | ROC-AUC | Log RMSE | Spearman |
|---|---:|---:|---:|---:|
| Chronological | 0.6530 | 0.7184 | 0.6974 | 0.4217 |
| Random (seed 42) | 0.6863 | 0.7343 | 0.6888 | 0.4823 |

The random split is easier because train and test share the same era
distribution. The chronological result is the more realistic estimate for
predicting genuinely future releases.
