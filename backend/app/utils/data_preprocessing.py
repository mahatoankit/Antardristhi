import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Automatically detect and categorize column types in a dataframe.

    Returns:
        Dictionary with categorized columns (numeric, categorical, datetime, text)
    """
    column_types = {"numeric": [], "categorical": [], "datetime": [], "text": []}

    for col in df.columns:
        # Check if column can be parsed as datetime
        try:
            if df[col].dtype == "object":
                pd.to_datetime(df[col], errors="raise")
                column_types["datetime"].append(col)
                continue
        except (ValueError, TypeError):
            pass

        # Check for numeric columns
        if np.issubdtype(df[col].dtype, np.number):
            column_types["numeric"].append(col)

        # Check for categorical columns (including boolean)
        elif df[col].dtype == "bool" or (
            df[col].nunique() / len(df) < 0.1 and df[col].nunique() < 20
        ):
            column_types["categorical"].append(col)

        # Assume text for remaining object columns
        elif df[col].dtype == "object":
            # If most values are less than 100 chars, treat as categorical
            if df[col].astype(str).str.len().mean() < 100:
                column_types["categorical"].append(col)
            else:
                column_types["text"].append(col)

    return column_types


def handle_missing_values(
    df: pd.DataFrame, strategy: str = "auto"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle missing values in dataframe using specified strategy

    Args:
        df: Input dataframe
        strategy: Strategy for handling missing values ('auto', 'drop', 'mean', 'median', 'mode')

    Returns:
        Processed dataframe and dictionary of imputation methods used
    """
    imputation_info = {}
    processed_df = df.copy()

    # Get column types
    col_types = detect_column_types(df)

    # Count missing values
    missing_counts = df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0].index.tolist()

    if not cols_with_missing:
        logger.info("No missing values detected")
        return processed_df, imputation_info

    logger.info(f"Handling missing values in columns: {cols_with_missing}")

    # Determine strategy if auto
    if strategy == "auto":
        # If too many missing values (>50%), drop columns
        drop_cols = [
            col for col in cols_with_missing if missing_counts[col] / len(df) > 0.5
        ]

        if drop_cols:
            processed_df = processed_df.drop(columns=drop_cols)
            imputation_info["dropped_columns"] = drop_cols
            logger.info(f"Dropped columns with >50% missing values: {drop_cols}")

            # Update cols_with_missing
            cols_with_missing = [
                col for col in cols_with_missing if col not in drop_cols
            ]

        # For remaining columns, impute based on data type
        for col in cols_with_missing:
            if col in col_types["numeric"]:
                # For numeric, use median
                median_val = df[col].median()
                processed_df[col] = processed_df[col].fillna(median_val)
                imputation_info[col] = {"method": "median", "value": float(median_val)}

            elif col in col_types["categorical"]:
                # For categorical, use mode
                mode_val = df[col].mode()[0]
                processed_df[col] = processed_df[col].fillna(mode_val)
                imputation_info[col] = {"method": "mode", "value": mode_val}

            elif col in col_types["datetime"]:
                # For datetime, forward fill then backward fill
                processed_df[col] = processed_df[col].ffill().bfill()
                imputation_info[col] = {"method": "ffill+bfill"}

    # For other strategies
    elif strategy == "drop":
        processed_df = processed_df.dropna()
        imputation_info["method"] = "drop rows"
        imputation_info["rows_before"] = len(df)
        imputation_info["rows_after"] = len(processed_df)

    elif strategy in ["mean", "median", "mode"]:
        for col in cols_with_missing:
            if col in col_types["numeric"]:
                if strategy == "mean":
                    val = df[col].mean()
                elif strategy == "median":
                    val = df[col].median()
                else:  # mode
                    val = df[col].mode()[0]

                processed_df[col] = processed_df[col].fillna(val)
                imputation_info[col] = {"method": strategy, "value": float(val)}

            elif col in col_types["categorical"] and strategy == "mode":
                mode_val = df[col].mode()[0]
                processed_df[col] = processed_df[col].fillna(mode_val)
                imputation_info[col] = {"method": "mode", "value": mode_val}

    return processed_df, imputation_info


def normalize_data(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Normalize numeric columns to 0-1 range

    Returns:
        Normalized dataframe and normalization parameters
    """
    normalized_df = df.copy()
    normalization_params = {}

    # Get numeric columns if not specified
    if columns is None:
        col_types = detect_column_types(df)
        columns = col_types["numeric"]

    # Filter to only include existing numeric columns
    numeric_cols = [
        col
        for col in columns
        if col in df.columns and np.issubdtype(df[col].dtype, np.number)
    ]

    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()

        # Skip if min and max are the same (no variability)
        if min_val == max_val:
            continue

        normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
        normalization_params[col] = {"min": float(min_val), "max": float(max_val)}

    return normalized_df, normalization_params


def detect_time_columns(df: pd.DataFrame) -> List[str]:
    """Identify potential time series columns"""
    time_cols = []

    for col in df.columns:
        # Check if column name suggests time
        if any(
            time_word in col.lower()
            for time_word in ["date", "time", "day", "month", "year", "quarter"]
        ):
            time_cols.append(col)
            continue

        # Check if column can be parsed as datetime
        if df[col].dtype == "object":
            try:
                pd.to_datetime(df[col])
                time_cols.append(col)
            except (ValueError, TypeError):
                pass

    return time_cols


def detect_id_columns(df: pd.DataFrame) -> List[str]:
    """Identify potential ID columns"""
    id_cols = []

    for col in df.columns:
        # Check if column name suggests ID
        if "id" in col.lower() or "_key" in col.lower():
            id_cols.append(col)
            continue

        # Check if column has unique values equal to row count (perfect key)
        if df[col].nunique() == len(df) and not np.issubdtype(df[col].dtype, np.number):
            id_cols.append(col)

    return id_cols


def prepare_time_series_data(
    df: pd.DataFrame, date_col: str, value_col: str, freq: str = None
) -> pd.DataFrame:
    """
    Prepare time series data for analysis/forecasting

    Args:
        df: Input dataframe
        date_col: Column containing dates
        value_col: Column containing values to analyze/forecast
        freq: Frequency for resampling (e.g., 'D', 'M', 'Y')

    Returns:
        DataFrame prepared for time series analysis
    """
    # Ensure date column is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Sort by date
    df = df.sort_values(by=date_col)

    # Resample if frequency specified
    if freq:
        df = df.set_index(date_col)
        df = df.resample(freq)[value_col].mean().reset_index()

    return df
