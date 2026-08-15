"""
feature_engineering.py
======================
Reusable feature engineering pipeline for the Customer Churn Prediction project.

Encoding strategy:
  - 3-class service cols (No internet/phone service) → simplify to Yes/No → 0/1
  - Binary Yes/No cols                              → Yes=1, No=0
  - Gender                                          → Male=1, Female=0
  - Contract (ordinal)                              → Month-to-month=0, One year=1, Two year=2
  - InternetService, PaymentMethod (nominal)        → one-hot encoding

New features engineered:
  - AvgMonthlySpend  = TotalCharges / (tenure + 1)   avoids div-by-zero for tenure=0
  - TenureGroup      = 0-12 / 13-24 / 25-48 / 49-72 months (label 0-3)
  - ServiceCount     = count of add-on services subscribed
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# Column group definitions
# ─────────────────────────────────────────────────────────────────────────────

# 3-value cols: Yes / No / "No internet service"  → simplify to Yes/No first
SERVICE_INTERNET_COLS = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
]

# 3-value col: Yes / No / "No phone service" → simplify to Yes/No
MULTI_LINES_COL = 'MultipleLines'

# Straight Yes/No binary cols
BINARY_COLS = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

# Ordinal: Month-to-month=0, One year=1, Two year=2
CONTRACT_MAP = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}

# One-hot encode (nominal, no natural order)
OHE_COLS = ['InternetService', 'PaymentMethod']

# Numerical columns to apply StandardScaler
SCALE_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend']


# ─────────────────────────────────────────────────────────────────────────────
# Step-by-step transformation functions
# ─────────────────────────────────────────────────────────────────────────────

def simplify_service_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace 'No internet service' / 'No phone service' with 'No'.

    Why: These values effectively mean the service is absent.
    Simplifying them to 'No' reduces cardinality from 3→2 for these columns,
    allowing clean binary encoding instead of one-hot encoding.
    """
    df = df.copy()
    for col in SERVICE_INTERNET_COLS:
        if col in df.columns:
            df[col] = df[col].replace('No internet service', 'No')
    if MULTI_LINES_COL in df.columns:
        df[MULTI_LINES_COL] = df[MULTI_LINES_COL].replace('No phone service', 'No')
    return df


def encode_binary_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all Yes/No columns to 1/0, and gender Male/Female to 1/0.
    Covers: BINARY_COLS + SERVICE_INTERNET_COLS + MULTI_LINES_COL + gender.
    """
    df = df.copy()
    yes_no_cols = BINARY_COLS + SERVICE_INTERNET_COLS + [MULTI_LINES_COL]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0}).astype('int8')

    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0}).astype('int8')

    return df


def encode_ordinal_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal-encode Contract: Month-to-month=0, One year=1, Two year=2.
    This preserves the natural ordering of contract commitment level.
    """
    df = df.copy()
    if 'Contract' in df.columns:
        df['Contract'] = df['Contract'].map(CONTRACT_MAP).astype('int8')
    return df


def encode_nominal_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode InternetService and PaymentMethod.
    drop_first=False keeps all categories for interpretability.
    Boolean dummies are cast to int8.
    """
    cols_present = [c for c in OHE_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cols_present, drop_first=False)
    # Cast bool columns produced by get_dummies → int8
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype('int8')
    return df


def engineer_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create three new derived features:

    AvgMonthlySpend: TotalCharges / (tenure + 1)
        Captures the average monthly spend rate.
        Dividing by (tenure+1) avoids ZeroDivisionError for new customers.

    TenureGroup: bins tenure into 4 segments (0-3 ordinal labels):
        0 → 0-12 months  (new customers, highest churn risk)
        1 → 13-24 months
        2 → 25-48 months
        3 → 49-72 months (loyal customers, lowest churn risk)

    ServiceCount: total number of add-on services subscribed.
        Customers with more services are more embedded → lower churn.
    """
    df = df.copy()

    # AvgMonthlySpend
    df['AvgMonthlySpend'] = (df['TotalCharges'] / (df['tenure'] + 1)).round(4)

    # TenureGroup
    df['TenureGroup'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype('int8')

    # ServiceCount — must be run AFTER binary encoding of service cols
    service_bin_cols = [c for c in SERVICE_INTERNET_COLS + [MULTI_LINES_COL]
                        if c in df.columns and df[c].dtype in ['int8', 'int64', 'int32']]
    df['ServiceCount'] = df[service_bin_cols].sum(axis=1).astype('int8')

    return df


def scale_features(df: pd.DataFrame,
                   scaler: StandardScaler = None,
                   fit: bool = True):
    """
    Apply StandardScaler to numerical columns.

    Parameters
    ----------
    df     : DataFrame
    scaler : existing StandardScaler (used when fit=False, i.e. on test set)
    fit    : if True, fit a new scaler on df (training set);
             if False, use provided scaler (test/inference set)

    Returns
    -------
    (df_scaled, scaler)
    """
    df = df.copy()
    cols = [c for c in SCALE_COLS if c in df.columns]
    if fit:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit=False")
        df[cols] = scaler.transform(df[cols])
    return df, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame,
                   fit_scaler: bool = True,
                   scaler: StandardScaler = None):
    """
    Run the full feature engineering pipeline in order:
      1. Simplify 3-class service columns
      2. Binary encode Yes/No & gender
      3. Ordinal encode Contract
      4. One-hot encode InternetService, PaymentMethod
      5. Engineer new features (AvgMonthlySpend, TenureGroup, ServiceCount)
      6. Scale numerical features

    Parameters
    ----------
    df          : cleaned DataFrame (output of preprocessing.clean_data)
    fit_scaler  : True for training set, False for test/inference
    scaler      : required if fit_scaler=False

    Returns
    -------
    (df_engineered, scaler)
    """
    print("=" * 55)
    print("  Feature Engineering Pipeline")
    print("=" * 55)

    df = simplify_service_cols(df)
    print("  [1] Simplified 3-class service cols → Yes/No")

    df = encode_binary_cols(df)
    print("  [2] Binary encoded (Yes/No, gender) → 0/1")

    df = encode_ordinal_cols(df)
    print("  [3] Ordinal encoded Contract → 0/1/2")

    df = encode_nominal_cols(df)
    print("  [4] One-hot encoded InternetService & PaymentMethod")

    df = engineer_new_features(df)
    print("  [5] Engineered AvgMonthlySpend, TenureGroup, ServiceCount")

    df, scaler = scale_features(df, scaler=scaler, fit=fit_scaler)
    print("  [6] StandardScaler applied to numerical cols")

    print("=" * 55)
    print(f"  Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("=" * 55)
    return df, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Train / Test split
# ─────────────────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame,
               target_col: str = 'Churn',
               test_size: float = 0.20,
               random_state: int = 42):
    """
    Stratified 80/20 train-test split.

    Stratification ensures both splits have the same 26.5% churn rate,
    which is critical for imbalanced classification.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"  Train : {X_train.shape[0]:,} rows  |  churn rate: {y_train.mean()*100:.1f}%")
    print(f"  Test  : {X_test.shape[0]:,} rows   |  churn rate: {y_test.mean()*100:.1f}%")
    print(f"  Features: {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test
