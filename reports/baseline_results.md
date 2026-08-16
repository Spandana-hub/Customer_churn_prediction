# Baseline Model Results


## Test Set Metrics

| Model               |   Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |   Train Time (s) |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|-----------------:|
| Random Forest       |     0.7573 |      0.529  |   0.7608 | 0.624  |    0.8419 |             0.42 |
| Logistic Regression |     0.7409 |      0.5072 |   0.7608 | 0.6086 |    0.8418 |             0.07 |
| Decision Tree       |     0.7402 |      0.5063 |   0.7581 | 0.6071 |    0.8212 |             0.03 |
| Dummy (Stratified)  |     0.6157 |      0.2754 |   0.2769 | 0.2761 |    0.5073 |             0    |


## Cross-Validation Results (mean +/- std)

| Model               | Accuracy        | Precision       | Recall          | F1              | ROC-AUC         |
|:--------------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| Dummy (Stratified)  | 0.6115 ± 0.0096 | 0.2678 ± 0.0178 | 0.2707 ± 0.0178 | 0.2693 ± 0.0178 | 0.5023 ± 0.0122 |
| Logistic Regression | 0.7577 ± 0.0129 | 0.5275 ± 0.0158 | 0.8020 ± 0.0318 | 0.6363 ± 0.0202 | 0.8495 ± 0.0169 |
| Decision Tree       | 0.7340 ± 0.0159 | 0.4979 ± 0.0198 | 0.7582 ± 0.0395 | 0.6010 ± 0.0258 | 0.8112 ± 0.0241 |
| Random Forest       | 0.7650 ± 0.0124 | 0.5382 ± 0.0163 | 0.7838 ± 0.0234 | 0.6382 ± 0.0184 | 0.8460 ± 0.0176 |


## Notes

- All models trained with class_weight=balanced (handles 26.5% churn imbalance)

- CV: 5-fold StratifiedKFold

- Primary metric: ROC-AUC (robust to class imbalance)

- Random Forest is the strongest baseline

- Day 6 will add XGBoost, CatBoost, LightGBM with hyperparameter tuning
