# Advanced Model Results

## Best Hyperparameters

### XGBoost
```
{'subsample': 0.7, 'reg_lambda': 5, 'reg_alpha': 0.01, 'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.01, 'gamma': 0, 'colsample_bytree': 0.6}
```

### LightGBM
```
{'subsample': 0.8, 'reg_lambda': 0.1, 'reg_alpha': 1, 'num_leaves': 20, 'n_estimators': 200, 'min_child_samples': 50, 'max_depth': 5, 'learning_rate': 0.01, 'colsample_bytree': 0.7}
```

### CatBoost
```
{'learning_rate': 0.01, 'l2_leaf_reg': 7, 'iterations': 400, 'depth': 7, 'border_count': 64, 'bagging_temperature': 2}
```

## All Models Test Set Comparison

| Model               |   Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|
| CatBoost (Tuned)    |     0.753  |      0.5223 |   0.7876 | 0.6281 |    0.8441 |
| LightGBM (Tuned)    |     0.7658 |      0.5416 |   0.7527 | 0.6299 |    0.8439 |
| XGBoost (Tuned)     |     0.7616 |      0.5343 |   0.7742 | 0.6323 |    0.8428 |
| Random Forest       |     0.7573 |      0.529  |   0.7608 | 0.624  |    0.8419 |
| Logistic Regression |     0.7409 |      0.5072 |   0.7608 | 0.6086 |    0.8418 |
| Decision Tree       |     0.7402 |      0.5063 |   0.7581 | 0.6071 |    0.8212 |

## Advanced Models CV Results

| Model            | Accuracy        | Precision       | Recall          | F1              | ROC-AUC         |
|:-----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| XGBoost (Tuned)  | 0.7628 ± 0.0125 | 0.5352 ± 0.0165 | 0.7892 ± 0.0214 | 0.6377 ± 0.0165 | 0.8480 ± 0.0174 |
| LightGBM (Tuned) | 0.7685 ± 0.0123 | 0.5439 ± 0.0167 | 0.7751 ± 0.0231 | 0.6391 ± 0.0179 | 0.8460 ± 0.0179 |
| CatBoost (Tuned) | 0.7550 ± 0.0125 | 0.5241 ± 0.0154 | 0.8013 ± 0.0276 | 0.6336 ± 0.0185 | 0.8493 ± 0.0168 |

## Best Model

**CatBoost (Tuned)** selected by highest ROC-AUC.

Saved to `models/best_model.pkl`
