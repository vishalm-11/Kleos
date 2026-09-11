# Kleos test-set evaluation

## Classifier

- Accuracy: 0.6549
- Majority-class baseline accuracy: 0.5480
- ROC-AUC: 0.7206
- PR-AUC: 0.6983
- Precision at 0.5: 0.6228
- Recall at 0.5: 0.5998
- F1 at 0.5: 0.6111

Confusion matrix (`[[TN, FP], [FN, TP]]`):

```text
[[804, 344], [379, 568]]
```

| Threshold | Precision | Recall | F1 | Predicted positive rate |
|---:|---:|---:|---:|---:|
| 0.3 | 0.4815 | 0.9609 | 0.6415 | 0.9021 |
| 0.4 | 0.5476 | 0.8268 | 0.6588 | 0.6826 |
| 0.5 | 0.6228 | 0.5998 | 0.6111 | 0.4353 |
| 0.6 | 0.7249 | 0.4340 | 0.5429 | 0.2706 |
| 0.7 | 0.7944 | 0.3020 | 0.4376 | 0.1718 |

## Regressor

- RMSE (original profit-multiple space): 5.8747
- R² (original space): 0.0405
- MAE (original space): 2.3685
- RMSE (log1p space, actual capped at 50): 0.7021
- R² (log1p space): 0.1697
- Spearman rank correlation: 0.4207
- Training-median baseline prediction: 1.8467
- Baseline RMSE: 6.1471
- Baseline MAE: 2.5791
- RMSE improvement over baseline: 0.2724
- MAE improvement over baseline: 0.2106
