# EDA Report — Customer Churn Prediction

## Dataset
- 7,043 customers, 20 features after cleaning (customerID dropped)
- Target: Churn (0=No 73.5%, 1=Yes 26.5%) — imbalanced

## Plots Generated
| File | Description |
|------|-------------|
| `churn_distribution.png` | Target class balance (bar + pie) |
| `numerical_distributions.png` | Histograms & boxplots per numerical feature |
| `categorical_churn_rates.png` | Churn rate per key categorical feature |
| `eda_tenure_by_churn.png` | Tenure KDE and violin by churn |
| `eda_charges_by_churn.png` | Monthly & Total Charges KDE and boxplot |
| `eda_senior_citizen_churn.png` | Senior citizen stacked bar + churn rate |
| `eda_contract_churn.png` | Contract type normalised stacked bar + count |
| `eda_internet_payment_churn.png` | Internet service & payment method churn rates |
| `eda_service_heatmap.png` | Churn rate heatmap across all 9 service features |
| `eda_pairplot.png` | Pairplot of numerical features coloured by churn |

## Key Findings

### Numerical Features
- **tenure**: Churners median ~10 months vs 38 months retained. Short-tenure = high risk.
- **MonthlyCharges**: Churners pay ~$74/mo vs ~$61/mo. Higher cost = higher churn.
- **TotalCharges**: Lower for churners due to short tenure. Highly correlated with tenure (r≈0.83).

### Categorical Features
- **Contract** (strongest): Month-to-month ~42% churn | One year ~11% | Two year ~3%.
- **InternetService**: Fiber optic ~42% | DSL ~19% | No internet ~7%.
- **PaymentMethod**: Electronic check ~45% | Others ~15-19%.
- **SeniorCitizen**: Seniors ~41% vs non-seniors ~24%.
- **OnlineSecurity / TechSupport**: Without service ~42% churn, with service ~15%.

## Modelling Implications
1. Top features: `Contract`, `tenure`, `MonthlyCharges`, `InternetService`, `PaymentMethod`
2. Engineer `AvgMonthlySpend = TotalCharges / (tenure+1)` to reduce multicollinearity
3. Handle class imbalance: use `class_weight='balanced'` or SMOTE
4. Use ROC-AUC and F1 (not accuracy) as primary metrics
