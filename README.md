# 📉 Customer Churn Prediction

A complete, end-to-end machine learning project to predict customer churn for a telecom company. Built as a structured **1-task-per-day learning project**, covering every major ML pipeline step.

---

## 📁 Project Structure

```
├── data/
│   ├── raw/                          # Original IBM Telco dataset
│   └── processed/                    # Cleaned & feature-engineered data
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Dataset overview & initial EDA
│   ├── 02_data_cleaning.ipynb        # Handling missing values & type fixes
│   ├── 03_eda_visualization.ipynb    # All EDA plots & business insights
│   ├── 04_feature_engineering.ipynb  # Encoding, scaling, new features
│   ├── 05_baseline_models.ipynb      # Baseline model comparisons
│   ├── 06_hyperparameter_tuning.ipynb# GridSearchCV & RandomizedSearchCV
│   ├── 07_model_comparison.ipynb     # Final model evaluation & selection
│   └── 08_model_explainability.ipynb # SHAP values & feature importance
├── src/
│   ├── config.py                     # Paths, column definitions, constants
│   ├── preprocessing.py              # Reusable data cleaning functions
│   ├── feature_engineering.py        # Feature transformation pipeline
│   ├── train.py                      # Model training utilities
│   ├── evaluate.py                   # Evaluation metrics & plots
│   ├── predict.py                    # Inference on new data
│   └── utils.py                      # General helper functions
├── models/                           # Saved model artifacts (.pkl)
├── reports/                          # Markdown reports with findings
├── images/                           # Exported plots (PNG)
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Source**: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows**: 7,043 customers
- **Columns**: 21 features
- **Target**: `Churn` — whether a customer left (Yes/No)
- **Class balance**: No: 73.5% / Yes: 26.5% (imbalanced)

---

## 🗓️ Daily Task Log

| Day | Task | Status |
|-----|------|--------|
| 1 | Project Setup & Data Exploration | ✅ Done |
| 2 | Data Cleaning & Preprocessing | 🔲 Pending |
| 3 | EDA Visualizations | 🔲 Pending |
| 4 | Feature Engineering | 🔲 Pending |
| 5 | Baseline Models | 🔲 Pending |
| 6 | Hyperparameter Tuning | 🔲 Pending |
| 7 | Model Comparison & Selection | 🔲 Pending |
| 8 | Model Explainability (SHAP) | 🔲 Pending |

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| ML | `scikit-learn` |
| Boosting | `xgboost`, `catboost`, `lightgbm` |
| Imbalance | `imbalanced-learn` (SMOTE) |
| Explainability | `shap` |
| Serialization | `joblib` |

---

## ⚙️ Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd customer-churn-prediction

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📈 Results

> Results will be updated as the project progresses.

---

## 🧠 Key Learnings

> Notes and insights will be added after each daily task.
