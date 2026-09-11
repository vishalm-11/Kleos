# Kleos test-set evaluation

## Classifier

- Accuracy: 0.6530
- Majority-class baseline accuracy: 0.5489
- ROC-AUC: 0.7184
- PR-AUC: 0.6894
- Precision at 0.5: 0.6238
- Recall at 0.5: 0.5814
- F1 at 0.5: 0.6019

Confusion matrix (`[[TN, FP], [FN, TP]]`):

```text
[[867, 351], [419, 582]]
```

| Threshold | Precision | Recall | F1 | Predicted positive rate |
|---:|---:|---:|---:|---:|
| 0.3 | 0.5151 | 0.8881 | 0.6520 | 0.7778 |
| 0.4 | 0.5843 | 0.7582 | 0.6600 | 0.5854 |
| 0.5 | 0.6238 | 0.5814 | 0.6019 | 0.4205 |
| 0.6 | 0.7158 | 0.4605 | 0.5605 | 0.2902 |
| 0.7 | 0.7677 | 0.3367 | 0.4681 | 0.1978 |

## Regressor

- RMSE (original profit-multiple space): 5.7021
- R² (original space): 0.0452
- MAE (original space): 2.2959
- RMSE (log1p space, actual capped at 50): 0.6974
- R² (log1p space): 0.1665
- Spearman rank correlation: 0.4217
- Training-median baseline prediction: 1.7877
- Baseline RMSE: 5.9939
- Baseline MAE: 2.5288
- RMSE improvement over baseline: 0.2919
- MAE improvement over baseline: 0.2329
