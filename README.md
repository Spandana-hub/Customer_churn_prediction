# Customer Churn Prediction

A complete, end-to-end Machine Learning project predicting customer churn for a telecom company.
Built task-by-task over 7 days, covering the full real-world ML pipeline.

---

## Project Structure

```
Customer-Churn-Prediction/
│
├── data/
│   ├── raw/                    # Original Telco Customer Churn CSV
│   └── processed/              # Cleaned, feature-engineered splits
│       ├── cleaned_data.csv
│       ├── features.csv
│       ├── train.csv
│       └── test.csv
│
├── notebooks/
│   └── customer_churn_analysis.ipynb   # Single notebook — all 7 sections
│
├── src/
│   ├── config.py               # Centralised paths & constants
│   ├── preprocessing.py        # Data cleaning pipeline
│   ├── feature_engineering.py  # Feature transformation pipeline
│   └── train.py                # Training, CV & evaluation utilities
│
├── models/
│   ├── scaler.pkl              # Fitted StandardScaler
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── best_model.pkl          # Best model selected by ROC-AUC
│
├── images/                     # All generated plots
├── reports/                    # Markdown + CSV result reports
├── requirements.txt
└── README.md
```

---

## Dataset

**IBM Telco Customer Churn** — 7,043 customers, 21 features.

- **Target:** `Churn` (Yes/No) — ~26.5% positive rate (imbalanced)
- **Features:** Demographics, account info, 9 telecom services, charges

---

## ML Pipeline — 7 Days

| Day | Task | Key Output |
|-----|------|-----------|
| 1 | Data Loading & Cleaning | `cleaned_data.csv`, null handling |
| 2 | Exploratory Data Analysis (EDA) — Part 1 | Churn distribution, numerical distributions |
| 3 | Exploratory Data Analysis (EDA) — Part 2 | Categorical plots, service heatmap, pairplot |
| 4 | Feature Engineering | `features.csv`, 3 new features, StandardScaler |
| 5 | Baseline Models | Logistic Regression, Decision Tree, Random Forest |
| 6 | Advanced Models + Tuning | XGBoost, LightGBM, CatBoost + RandomizedSearchCV |
| 7 | Model Explainability | SHAP beeswarm, waterfall, threshold tuning |

---

## Feature Engineering (Day 4)

| Step | Technique | Columns |
|------|-----------|---------|
| Simplify | `No internet/phone service` → `No` | 7 service cols |
| Binary encode | Yes/No → 1/0 | 11 cols |
| Ordinal encode | Contract → 0/1/2 | Contract |
| One-hot encode | Nominal categories | InternetService, PaymentMethod |
| **New: AvgMonthlySpend** | TotalCharges / (tenure+1) | — |
| **New: TenureGroup** | 4 tenure bins (0–72 months) | — |
| **New: ServiceCount** | Sum of add-on services | — |
| Scale | StandardScaler | tenure, MonthlyCharges, TotalCharges, AvgMonthlySpend |

---

## Results

### All Models — Test Set (sorted by ROC-AUC)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| XGBoost (Tuned) | — | — | — | — | — |
| LightGBM (Tuned) | — | — | — | — | — |
| CatBoost (Tuned) | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| Logistic Regression | — | — | — | — | — |
| Decision Tree | — | — | — | — | — |

> *Run the notebook to see actual scores — they depend on your machine's random seed results.*
> Full results in [`reports/all_models_comparison.csv`](reports/all_models_comparison.csv)

### Class Imbalance Strategy

| Model | Strategy |
|-------|---------|
| Logistic Regression, Decision Tree, Random Forest | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight = n_neg / n_pos` |
| LightGBM | `is_unbalance=True` |
| CatBoost | `auto_class_weights='Balanced'` |

---

## Model Explainability (Day 7)

SHAP (SHapley Additive exPlanations) was used to explain the best model:

| Plot | File | What it shows |
|------|------|--------------|
| Tree Feature Importance | `shap_feature_importance.png` | Split-based importance |
| Beeswarm | `shap_beeswarm.png` | Per-customer feature impact |
| Bar | `shap_bar.png` | Global mean \|SHAP\| ranking |
| Waterfall | `shap_waterfall.png` | Single prediction explained |
| Dependence | `shap_dependence.png` | Top-3 feature effects + interactions |
| PR + Threshold | `shap_pr_threshold.png` | Optimal classification threshold |

**Key churn drivers found:**
- Contract type (Month-to-month = highest risk)
- Tenure (new customers churn most)
- InternetService = Fiber optic
- PaymentMethod = Electronic check
- Monthly charges
- TenureGroup & AvgMonthlySpend

---

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/customer_churn_analysis.ipynb
```

---

## Tech Stack

| Category | Libraries |
|----------|----------|
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| ML | scikit-learn |
| Boosting | XGBoost 3.4, LightGBM 4.7, CatBoost 1.2 |
| Explainability | SHAP 0.52 |
| Serialisation | joblib |

---

## Key Learnings

- **Class imbalance** must be handled explicitly — `class_weight` or `scale_pos_weight`
- **ROC-AUC** is the primary metric for imbalanced classification, not accuracy
- **Threshold tuning** can significantly improve F1 over the default 0.5 threshold
- **SHAP** gives trust and actionability to model predictions
- **Feature engineering** (TenureGroup, ServiceCount, AvgMonthlySpend) adds predictive signal
- **RandomizedSearchCV** gives ~90% of GridSearchCV quality at a fraction of the compute time
