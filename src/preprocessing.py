"""
preprocessing.py
================
Reusable data cleaning and preprocessing functions for the
Customer Churn Prediction project.

Usage (in notebooks):
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
    from src.preprocessing import load_raw_data, clean_data, get_cleaning_report
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# 1. Data Loading
# ─────────────────────────────────────────────

def load_raw_data(path: Path) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn CSV.

    Parameters
    ----------
    path : Path
        Absolute path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe (no transformations applied yet).
    """
    df = pd.read_csv(path)
    print(f"✅ Loaded raw data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 2. Individual Cleaning Steps
# ─────────────────────────────────────────────

def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the TotalCharges column:
      - It is stored as object (string) in the raw CSV.
      - 11 rows have blank strings (' ') → these are customers with tenure=0
        who haven't been billed yet. We convert them to NaN first.
      - Impute NaN values with the column median.
      - Cast to float64.

    Why median and not mean?
      - TotalCharges has a right-skewed distribution (long tail of high-value customers).
      - Median is robust to skew and outliers, making it the safer choice here.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame (copy)
    """
    df = df.copy()

    # Step 1: Replace blank strings with NaN
    df['TotalCharges'] = df['TotalCharges'].str.strip().replace('', np.nan)

    # Step 2: Convert to numeric (any non-numeric → NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Step 3: Count and report imputed rows
    n_missing = df['TotalCharges'].isna().sum()
    if n_missing > 0:
        median_val = df['TotalCharges'].median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median_val)
        print(f"  ✔ TotalCharges: {n_missing} blank rows imputed with median = {median_val:.2f}")
    else:
        print("  ✔ TotalCharges: no missing values found")

    print(f"  ✔ TotalCharges dtype → {df['TotalCharges'].dtype}")
    return df


def drop_id_column(df: pd.DataFrame, id_col: str = 'customerID') -> pd.DataFrame:
    """
    Drop the customer identifier column — it carries no predictive signal.

    Parameters
    ----------
    df     : pd.DataFrame
    id_col : str, column name to drop (default: 'customerID')

    Returns
    -------
    pd.DataFrame (copy)
    """
    df = df.copy()
    if id_col in df.columns:
        df.drop(columns=[id_col], inplace=True)
        print(f"  ✔ Dropped column: '{id_col}'")
    else:
        print(f"  ⚠ Column '{id_col}' not found — skipping drop")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any fully duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame (copy)
    """
    df = df.copy()
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    removed = before - after
    if removed > 0:
        print(f"  ✔ Removed {removed} duplicate row(s)")
    else:
        print(f"  ✔ No duplicate rows found")
    return df


def fix_senior_citizen(df: pd.DataFrame) -> pd.DataFrame:
    """
    SeniorCitizen is already stored as int (0/1) in the dataset.
    No encoding needed, but we confirm dtype and rename for clarity.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame (copy)
    """
    df = df.copy()
    assert df['SeniorCitizen'].isin([0, 1]).all(), \
        "SeniorCitizen contains unexpected values!"
    print(f"  ✔ SeniorCitizen: confirmed as binary int (0=No, 1=Yes)")
    return df


def encode_target(df: pd.DataFrame, target_col: str = 'Churn') -> pd.DataFrame:
    """
    Encode the target variable from string ('Yes'/'No') → int (1/0).
    This makes it compatible with all sklearn estimators.

    Parameters
    ----------
    df         : pd.DataFrame
    target_col : str

    Returns
    -------
    pd.DataFrame (copy)
    """
    df = df.copy()
    mapping = {'Yes': 1, 'No': 0}
    df[target_col] = df[target_col].map(mapping)
    print(f"  ✔ '{target_col}' encoded: Yes→1, No→0")
    print(f"    Distribution: {df[target_col].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────
# 3. Master Cleaning Pipeline
# ─────────────────────────────────────────────

def clean_data(
    df: pd.DataFrame,
    id_col: str = 'customerID',
    target_col: str = 'Churn',
    encode_target_flag: bool = True
) -> pd.DataFrame:
    """
    Run the full cleaning pipeline in sequence:
      1. Fix TotalCharges (blank → NaN → impute median → float)
      2. Drop customerID
      3. Remove duplicate rows
      4. Validate SeniorCitizen
      5. Encode target variable (optional)

    Parameters
    ----------
    df                 : pd.DataFrame  — raw dataframe
    id_col             : str           — identifier column to drop
    target_col         : str           — target column name
    encode_target_flag : bool          — if True, encode Churn as 0/1

    Returns
    -------
    pd.DataFrame — cleaned dataframe
    """
    print("=" * 55)
    print("  🧹 Running Cleaning Pipeline")
    print("=" * 55)

    df = fix_total_charges(df)
    df = drop_id_column(df, id_col=id_col)
    df = remove_duplicates(df)
    df = fix_senior_citizen(df)

    if encode_target_flag:
        df = encode_target(df, target_col=target_col)

    print("=" * 55)
    print(f"  ✅ Cleaning complete! Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print("=" * 55)
    return df


# ─────────────────────────────────────────────
# 4. Cleaning Report
# ─────────────────────────────────────────────

def get_cleaning_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a before/after cleaning comparison report showing:
    - Rows and columns counts
    - Null counts per column
    - Dtype changes

    Parameters
    ----------
    df_raw   : pd.DataFrame — original raw dataframe
    df_clean : pd.DataFrame — cleaned dataframe

    Returns
    -------
    pd.DataFrame — comparison report
    """
    rows_removed = len(df_raw) - len(df_clean)
    cols_removed = df_raw.shape[1] - df_clean.shape[1]

    print("📋 Cleaning Report")
    print("-" * 40)
    print(f"  Rows : {len(df_raw):,} → {len(df_clean):,}  (removed: {rows_removed})")
    print(f"  Cols : {df_raw.shape[1]:,} → {df_clean.shape[1]:,}  (removed: {cols_removed})")
    print()

    # Dtype changes
    common_cols = [c for c in df_raw.columns if c in df_clean.columns]
    dtype_changes = []
    for col in common_cols:
        before = str(df_raw[col].dtype)
        after  = str(df_clean[col].dtype)
        if before != after:
            dtype_changes.append({'Column': col, 'Before': before, 'After': after})

    if dtype_changes:
        print("  Dtype changes:")
        report_df = pd.DataFrame(dtype_changes)
        print(report_df.to_string(index=False))
    else:
        print("  No dtype changes.")

    # Remaining nulls
    null_counts = df_clean.isnull().sum()
    remaining_nulls = null_counts[null_counts > 0]
    print()
    if len(remaining_nulls) > 0:
        print("  ⚠ Remaining nulls after cleaning:")
        print(remaining_nulls)
    else:
        print("  ✅ No null values remain in cleaned dataset")

    return df_clean.dtypes.rename("dtype_after").to_frame()
