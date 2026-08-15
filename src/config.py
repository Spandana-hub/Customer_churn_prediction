"""
config.py
=========
Central configuration file for the Customer Churn Prediction project.
All paths, column definitions, and constants are defined here.
Import this module in notebooks and src/ scripts to avoid hardcoded paths.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Project Root & Directory Paths
# ─────────────────────────────────────────────

# Automatically resolve project root (one level above src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR       = PROJECT_ROOT / "data"
RAW_DATA_DIR   = DATA_DIR / "raw"
PROC_DATA_DIR  = DATA_DIR / "processed"
NOTEBOOKS_DIR  = PROJECT_ROOT / "notebooks"
MODELS_DIR     = PROJECT_ROOT / "models"
REPORTS_DIR    = PROJECT_ROOT / "reports"
IMAGES_DIR     = PROJECT_ROOT / "images"

# Create output dirs if they don't exist
for _dir in [PROC_DATA_DIR, MODELS_DIR, REPORTS_DIR, IMAGES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Data File Paths
# ─────────────────────────────────────────────

RAW_DATA_PATH     = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEANED_DATA_PATH = PROC_DATA_DIR / "cleaned_data.csv"
FEATURES_PATH     = PROC_DATA_DIR / "features.csv"
TRAIN_PATH        = PROC_DATA_DIR / "train.csv"
TEST_PATH         = PROC_DATA_DIR / "test.csv"
SCALER_PATH       = MODELS_DIR / "scaler.pkl"
BEST_MODEL_PATH   = MODELS_DIR / "best_model.pkl"


# ─────────────────────────────────────────────
# Column Definitions
# ─────────────────────────────────────────────

TARGET_COL = "Churn"

# Drop before modeling — not a feature
ID_COL = "customerID"

# Numerical features (continuous)
NUMERICAL_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# Binary Yes/No categorical columns → encode to 1/0
BINARY_COLS = [
    "gender",          # Female=0, Male=1
    "Partner",         # No=0, Yes=1
    "Dependents",      # No=0, Yes=1
    "PhoneService",    # No=0, Yes=1
    "PaperlessBilling",# No=0, Yes=1
]

# Ordinal categorical column → label encode with order
ORDINAL_COLS = {
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
}

# Nominal categorical columns → one-hot encode
NOMINAL_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaymentMethod",
]

# SeniorCitizen is already 0/1 integer — no encoding needed
ALREADY_ENCODED_COLS = ["SeniorCitizen"]


# ─────────────────────────────────────────────
# Model Training Constants
# ─────────────────────────────────────────────

RANDOM_STATE   = 42
TEST_SIZE      = 0.20       # 80/20 train-test split
CV_FOLDS       = 5          # StratifiedKFold folds
SCORING_METRIC = "roc_auc"  # Primary optimization metric


# ─────────────────────────────────────────────
# Visualization Settings
# ─────────────────────────────────────────────

PLOT_STYLE  = "seaborn-v0_8-darkgrid"
FIG_DPI     = 150
COLOR_CHURN = {"No": "#4CAF50", "Yes": "#F44336"}  # Green = stayed, Red = churned
PALETTE     = "husl"
