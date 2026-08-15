"""
train.py
========
Model training, cross-validation, and evaluation utilities
for the Customer Churn Prediction project.

Supports: Logistic Regression, Decision Tree, Random Forest,
          XGBoost, CatBoost, LightGBM (Days 5-6).
"""

import numpy as np
import pandas as pd
import joblib
import time
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, ConfusionMatrixDisplay
)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

SCORING = {
    "accuracy"  : "accuracy",
    "precision" : "precision",
    "recall"    : "recall",
    "f1"        : "f1",
    "roc_auc"   : "roc_auc",
}


def get_test_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Compute all classification metrics on the held-out test set.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc
    """
    y_pred  = model.predict(X_test)
    y_proba = (model.predict_proba(X_test)[:, 1]
               if hasattr(model, "predict_proba") else y_pred)

    return {
        "Accuracy" : round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "F1"       : round(f1_score(y_test, y_pred,        zero_division=0), 4),
        "ROC-AUC"  : round(roc_auc_score(y_test, y_proba), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training & CV
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    random_state: int = 42
) -> dict:
    """
    Run StratifiedKFold cross-validation and return mean ± std for each metric.

    StratifiedKFold preserves the 26.5% churn class ratio in every fold.

    Returns
    -------
    dict: {metric_name: {"mean": float, "std": float}}
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=skf, scoring=SCORING,
        return_train_score=False, n_jobs=-1
    )
    return {
        metric: {
            "mean": round(cv_results[f"test_{key}"].mean(), 4),
            "std" : round(cv_results[f"test_{key}"].std(),  4),
        }
        for metric, key in [
            ("Accuracy",  "accuracy"),
            ("Precision", "precision"),
            ("Recall",    "recall"),
            ("F1",        "f1"),
            ("ROC-AUC",   "roc_auc"),
        ]
    }


def train_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    cv: int = 5,
) -> dict:
    """
    Full training run: fit → CV → test evaluation.

    Parameters
    ----------
    model       : sklearn-compatible estimator
    X_train/y_train : training data
    X_test/y_test   : held-out test data
    model_name  : string label for reporting
    cv          : number of CV folds

    Returns
    -------
    dict with keys:
        name, model, cv_results, test_metrics, train_time_sec
    """
    print(f"\n  Training: {model_name}")
    print(f"  {'─' * 40}")

    # Fit on full training set
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 2)

    # Cross-validation (refit from scratch inside CV)
    print(f"  Running {cv}-fold Stratified CV ...", end=" ")
    cv_res = cross_validate_model(model, X_train, y_train, cv=cv)
    print("done")

    # Test set metrics
    test_metrics = get_test_metrics(model, X_test, y_test)

    print(f"  CV   F1={cv_res['F1']['mean']:.4f}  "
          f"ROC-AUC={cv_res['ROC-AUC']['mean']:.4f}")
    print(f"  Test F1={test_metrics['F1']:.4f}  "
          f"ROC-AUC={test_metrics['ROC-AUC']:.4f}  "
          f"({train_time}s)")

    return {
        "name"        : model_name,
        "model"       : model,
        "cv_results"  : cv_res,
        "test_metrics": test_metrics,
        "train_time"  : train_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Results formatting
# ─────────────────────────────────────────────────────────────────────────────

def build_results_table(results: list) -> pd.DataFrame:
    """
    Convert a list of train_model() result dicts into a comparison DataFrame.

    Columns: Model | Accuracy | Precision | Recall | F1 | ROC-AUC
    Rows: one per model, showing test-set metrics.
    """
    rows = []
    for r in results:
        row = {"Model": r["name"]}
        row.update(r["test_metrics"])
        row["Train Time (s)"] = r["train_time"]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    return df.sort_values("ROC-AUC", ascending=False)


def build_cv_table(results: list) -> pd.DataFrame:
    """CV mean ± std comparison across models."""
    rows = []
    for r in results:
        row = {"Model": r["name"]}
        for metric, vals in r["cv_results"].items():
            row[metric] = f"{vals['mean']:.4f} ± {vals['std']:.4f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, path: Path) -> None:
    """Save a fitted model to disk using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  Saved model → {path}")


def load_model(path: Path):
    """Load a model from disk."""
    return joblib.load(path)
