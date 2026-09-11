# Kleos 2023 backtest

## Headline
- Movies: 118
- Actual hit rate: 37.29%
- Majority-class baseline accuracy: 0.6271

## Classifier
- Default threshold: 0.5000
- Accuracy: 0.6017
- ROC-AUC: 0.6953
- PR-AUC: 0.5868
- Precision / recall / F1 at 0.5: 0.4762 / 0.6818 / 0.5607
- Confusion matrix (`[[TN, FP], [FN, TP]]`): `[[41, 33], [14, 30]]`

Of the 44 actual hits, the model flagged 30 in advance. Of the
74 actual flops, it correctly called 41.

### Test-tuned operating threshold
- Threshold loaded from training artifact: 0.3913
- Accuracy: 0.5254
- Precision: 0.4318
- Recall: 0.8636
- F1: 0.5758
- Confusion matrix: `[[24, 50], [6, 38]]`

| threshold | precision | recall | f1 | predicted_positive_rate |
|---:|---:|---:|---:|---:|
| 0.3 | 0.3925 | 0.9545 | 0.5563 | 0.9068 |
| 0.4 | 0.4286 | 0.8182 | 0.5625 | 0.7119 |
| 0.5 | 0.4762 | 0.6818 | 0.5607 | 0.5339 |
| 0.6 | 0.6341 | 0.5909 | 0.6118 | 0.3475 |
| 0.7 | 0.7000 | 0.4773 | 0.5676 | 0.2542 |

## Regressor
- Log-RMSE: 0.6761
- Spearman: 0.3667
- MAE: 1.9754
- Training-median baseline: 1.8467
- Baseline RMSE / model RMSE: 3.3485 / 2.9932
- Baseline MAE / model MAE: 2.0785 / 1.9754

### Calibration by predicted quintile
| quintile | n_movies | mean_predicted | mean_actual |
|---:|---:|---:|---:|
| 1.0 | 24.0 | 1.165 | 1.036 |
| 2.0 | 23.0 | 1.532 | 2.216 |
| 3.0 | 24.0 | 1.992 | 2.268 |
| 4.0 | 23.0 | 2.800 | 2.055 |
| 5.0 | 24.0 | 4.394 | 5.072 |

## Notable releases
| title | budget_adj_m | actual_profit_multiple | predicted_profit_multiple | actual_hit | hit_probability | predicted_hit | correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lost in the Stars | $441.2M | 0.80 | 1.61 | 0 | 0.429 | 0 | yes |
| Fast X | $359.2M | 2.07 | 2.92 | 1 | 0.894 | 1 | yes |
| The Flash | $317.0M | 0.89 | 2.12 | 0 | 0.592 | 1 | no |
| Indiana Jones and the Dial of Destiny | $311.4M | 1.30 | 4.18 | 0 | 0.892 | 1 | no |
| Mission: Impossible - Dead Reckoning Part One | $307.5M | 1.94 | 3.99 | 0 | 0.926 | 1 | no |
| Guardians of the Galaxy Vol. 3 | $264.1M | 3.38 | 4.05 | 1 | 0.930 | 1 | yes |
| The Little Mermaid | $264.1M | 2.28 | 2.23 | 1 | 0.522 | 1 | yes |
| Ant-Man and the Wasp: Quantumania | $211.3M | 2.38 | 3.19 | 1 | 0.843 | 1 | yes |
| Elemental | $211.3M | 2.43 | 1.72 | 1 | 0.407 | 0 | no |
| Transformers: Rise of the Beasts | $206.0M | 2.20 | 1.58 | 1 | 0.442 | 0 | no |
| The Super Mario Bros. Movie | $105.7M | 13.56 | 3.93 | 1 | 0.863 | 1 | yes |
| Knights of the Zodiac | $63.4M | 0.11 | 1.10 | 0 | 0.326 | 0 | yes |
| Mafia Mamma | $43.3M | 0.10 | 1.20 | 0 | 0.282 | 0 | yes |
| Big George Foreman | $33.8M | 0.12 | 1.78 | 0 | 0.337 | 0 | yes |
| Insidious: The Red Door | $16.9M | 11.60 | 6.34 | 1 | 0.788 | 1 | yes |
| Sound of Freedom | $15.8M | 14.17 | 1.39 | 1 | 0.394 | 0 | no |
| Cheburashka | $8.1M | 12.60 | 4.18 | 1 | 0.780 | 1 | yes |
| Scarlet | $7.1M | 0.04 | 1.29 | 0 | 0.287 | 0 | yes |
| Talk to Me | $4.8M | 16.13 | 5.60 | 1 | 0.781 | 1 | yes |
| Detective Knight: Independence | $2.1M | 0.02 | 4.62 | 0 | 0.637 | 1 | no |

## Biggest classifier misses
| title | actual_hit | predicted_hit | hit_probability | actual_profit_multiple |
|---|---:|---:|---:|---:|
| Mission: Impossible - Dead Reckoning Part One | 0 | 1 | 0.926 | 1.94 |
| Indiana Jones and the Dial of Destiny | 0 | 1 | 0.892 | 1.30 |
| Shazam! Fury of the Gods | 0 | 1 | 0.890 | 1.07 |
| Magic Mike's Last Dance | 0 | 1 | 0.864 | 1.16 |
| Expend4bles | 0 | 1 | 0.808 | 0.15 |

## Biggest regressor misses
| title | actual_profit_multiple | predicted_profit_multiple | absolute_log_error |
|---|---:|---:|---:|
| Sound of Freedom | 14.17 | 1.39 | 1.849 |
| Detective Knight: Independence | 0.02 | 4.62 | 1.708 |
| Fallen Leaves | 8.38 | 1.37 | 1.374 |
| Beautiful Disaster | 0.27 | 4.03 | 1.373 |
| Barbie | 9.85 | 2.10 | 1.254 |
