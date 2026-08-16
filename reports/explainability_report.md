# Model Explainability Report

**Best Model:** CatBoostClassifier

## Top 10 — Tree Feature Importance

|                                |       0 |
|:-------------------------------|--------:|
| Contract                       | 19.9842 |
| tenure                         | 10.506  |
| MonthlyCharges                 |  8.2517 |
| InternetService_Fiber optic    |  8.0298 |
| TenureGroup                    |  6.1669 |
| TotalCharges                   |  4.7525 |
| PaymentMethod_Electronic check |  4.6563 |
| PaperlessBilling               |  4.3232 |
| AvgMonthlySpend                |  4.1133 |
| ServiceCount                   |  3.0571 |

## Top 10 — Mean |SHAP| Value

|                                |      0 |
|:-------------------------------|-------:|
| Contract                       | 0.6674 |
| InternetService_Fiber optic    | 0.2773 |
| tenure                         | 0.259  |
| TenureGroup                    | 0.2308 |
| MonthlyCharges                 | 0.2034 |
| PaymentMethod_Electronic check | 0.194  |
| TotalCharges                   | 0.1216 |
| PaperlessBilling               | 0.109  |
| InternetService_No             | 0.099  |
| OnlineSecurity                 | 0.0775 |

## Threshold Tuning

| Metric | Default (0.50) | Optimal (0.47) |
|--------|----------------|---------------|
| F1     | 0.6281  | 0.6355 |
| AP     | —  | 0.6582 |

## Key Insights

- Top SHAP feature: **Contract**
- Second: **InternetService_Fiber optic**
- Third: **tenure**
- Optimal threshold (0.47) improves F1 over default (0.50)
- SHAP plots saved to images/shap_*.png
