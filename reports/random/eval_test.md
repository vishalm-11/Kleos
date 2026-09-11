# Kleos test-set evaluation

## Classifier

- Accuracy: 0.6863
- Majority-class baseline accuracy: 0.5354
- ROC-AUC: 0.7343
- PR-AUC: 0.7123
- Precision at 0.5: 0.6795
- Recall at 0.5: 0.6149
- F1 at 0.5: 0.6456

Confusion matrix (`[[TN, FP], [FN, TP]]`):

```text
[[889, 299], [397, 634]]
```

| Threshold | Precision | Recall | F1 | Predicted positive rate |
|---:|---:|---:|---:|---:|
| 0.3 | 0.5104 | 0.9282 | 0.6586 | 0.8450 |
| 0.4 | 0.5837 | 0.7983 | 0.6743 | 0.6354 |
| 0.5 | 0.6795 | 0.6149 | 0.6456 | 0.4205 |
| 0.6 | 0.7296 | 0.4345 | 0.5447 | 0.2767 |
| 0.7 | 0.7995 | 0.3055 | 0.4421 | 0.1776 |

## Regressor

- RMSE (original profit-multiple space): 9.0433
- R² (original space): 0.1294
- MAE (original space): 2.8557
- RMSE (log1p space, actual capped at 50): 0.6888
- R² (log1p space): 0.2706
- Spearman rank correlation: 0.4823
- Training-median baseline prediction: 1.7600
- Baseline RMSE: 9.9251
- Baseline MAE: 3.1837
- RMSE improvement over baseline: 0.8818
- MAE improvement over baseline: 0.3280
