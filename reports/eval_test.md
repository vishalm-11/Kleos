# Kleos test-set evaluation

## Classifier

- Accuracy: 0.6627
- Majority-class baseline accuracy: 0.5297
- ROC-AUC: 0.7228
- PR-AUC: 0.7190
- Precision at 0.5: 0.6803
- Recall at 0.5: 0.5336
- F1 at 0.5: 0.5981

Confusion matrix (`[[TN, FP], [FN, TP]]`):

```text
[[873, 250], [465, 532]]
```

| Threshold | Precision | Recall | F1 | Predicted positive rate |
|---:|---:|---:|---:|---:|
| 0.3 | 0.5287 | 0.8957 | 0.6649 | 0.7967 |
| 0.4 | 0.6069 | 0.7232 | 0.6600 | 0.5604 |
| 0.5 | 0.6803 | 0.5336 | 0.5981 | 0.3689 |
| 0.6 | 0.7646 | 0.3942 | 0.5202 | 0.2425 |
| 0.7 | 0.7983 | 0.2778 | 0.4122 | 0.1637 |

## Regressor

- RMSE (original profit-multiple space): 6.3096
- R² (original space): 0.0201
- MAE (original space): 2.4414
- RMSE (log1p space, actual capped at 50): 0.7032
- R² (log1p space): 0.1751
- Spearman rank correlation: 0.4326
- Training-median baseline prediction: 1.7869
- Baseline RMSE: 6.5633
- Baseline MAE: 2.6693
- RMSE improvement over baseline: 0.2537
- MAE improvement over baseline: 0.2279
