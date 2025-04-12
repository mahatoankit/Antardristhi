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


def prepare_data_for_visualization(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: Optional[str] = None,
    category_col: Optional[str] = None,
    limit: int = 50,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Prepare data for visualization

    Args:
        df: Input dataframe
        chart_type: Type of chart ('bar', 'line', 'scatter', 'pie', 'histogram')
        x_col: Column to use for x-axis
        y_col: Column to use for y-axis (optional)
        category_col: Column to use for categories/grouping (optional)
        limit: Maximum number of data points

    Returns:
        Prepared dataframe and metadata for the visualization
    """
    metadata = {
        "chart_type": chart_type,
        "x_col": x_col,
        "y_col": y_col,
        "category_col": category_col,
        "original_size": len(df),
    }

    prepared_df = df.copy()

    # Drop nulls in relevant columns
    cols_to_check = [col for col in [x_col, y_col, category_col] if col is not None]
    if cols_to_check:
        prepared_df = prepared_df.dropna(subset=cols_to_check)
        metadata["size_after_dropna"] = len(prepared_df)

    # For categorical x-axis, limit to top categories
    if chart_type in ["bar", "pie"] and y_col is None:
        # Count values and sort
        value_counts = prepared_df[x_col].value_counts().reset_index()
        value_counts.columns = [x_col, "count"]
        prepared_df = value_counts.head(limit)
        metadata["aggregation"] = "count"

    # For categorical x-axis with numeric y-axis, aggregate by x
    elif chart_type in ["bar"] and y_col is not None:
        if category_col:
            # Group by both x and category, then pivot
            grouped = (
                prepared_df.groupby([x_col, category_col])[y_col].mean().reset_index()
            )
            prepared_df = grouped.pivot(
                index=x_col, columns=category_col, values=y_col
            ).reset_index()
            metadata["aggregation"] = "mean by category"
        else:
            # Just group by x
            grouped = prepared_df.groupby(x_col)[y_col].mean().reset_index()
            prepared_df = grouped.head(limit)
            metadata["aggregation"] = "mean"

    # For time series data, ensure sorted and limit points if needed
    elif chart_type == "line":
        # Try to convert to datetime if it's not already
        if prepared_df[x_col].dtype != "datetime64[ns]":
            try:
                prepared_df[x_col] = pd.to_datetime(prepared_df[x_col])
            except:
                pass

        prepared_df = prepared_df.sort_values(by=x_col)

        # If too many points, resample to reduce
        if len(prepared_df) > limit:
            if prepared_df[x_col].dtype == "datetime64[ns]":
                prepared_df = prepared_df.set_index(x_col)
                # Determine appropriate frequency based on date range
                date_range = (prepared_df.index.max() - prepared_df.index.min()).days
                if date_range > 365 * 2:  # More than 2 years
                    freq = "M"  # Monthly
                elif date_range > 60:  # More than 2 months
                    freq = "W"  # Weekly
                else:
                    freq = "D"  # Daily

                if category_col:
                    # This is more complex - group by time and category
                    prepared_df = (
                        prepared_df.groupby([pd.Grouper(freq=freq), category_col])[
                            y_col
                        ]
                        .mean()
                        .reset_index()
                    )
                else:
                    prepared_df = prepared_df.resample(freq)[y_col].mean().reset_index()
                metadata["resampled"] = True
                metadata["frequency"] = freq
            else:
                # Just take every nth row
                n = max(1, len(prepared_df) // limit)
                prepared_df = prepared_df.iloc[::n, :].head(limit)
                metadata["sampled"] = True
                metadata["sampling_rate"] = n

    # For scatter plots, sample if too many points
    elif chart_type == "scatter" and len(prepared_df) > limit:
        prepared_df = prepared_df.sample(limit, random_state=42)
        metadata["sampled"] = True
        metadata["sample_size"] = limit

    # For histograms, just return the data (the binning will be done during visualization)
    elif chart_type == "histogram":
        if len(prepared_df) > 1000:
            prepared_df = prepared_df.sample(1000, random_state=42)
            metadata["sampled"] = True

    # Final limit to ensure we don't return too much data
    if len(prepared_df) > limit and chart_type not in ["histogram"]:
        prepared_df = prepared_df.head(limit)
        metadata["limited"] = True
        metadata["limit"] = limit

    metadata["final_size"] = len(prepared_df)
    return prepared_df, metadata


def prepare_data_for_table(
    df: pd.DataFrame,
    max_rows: int = 100,
    sort_by: Optional[str] = None,
    ascending: bool = True,
    filter_expr: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Prepare data for table display in the chatbot

    Args:
        df: Input dataframe
        max_rows: Maximum number of rows to include
        sort_by: Column to sort by (optional)
        ascending: Sort direction
        filter_expr: Filter expression (optional)

    Returns:
        Prepared dataframe and metadata for the table
    """
    metadata = {
        "original_size": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }

    prepared_df = df.copy()

    # Apply filter if provided
    if filter_expr:
        try:
            prepared_df = prepared_df.query(filter_expr)
            metadata["filtered"] = True
            metadata["filter_expr"] = filter_expr
            metadata["size_after_filter"] = len(prepared_df)
        except Exception as e:
            metadata["filter_error"] = str(e)

    # Apply sorting if provided
    if sort_by and sort_by in prepared_df.columns:
        prepared_df = prepared_df.sort_values(by=sort_by, ascending=ascending)
        metadata["sorted"] = True
        metadata["sort_by"] = sort_by
        metadata["ascending"] = ascending

    # Limit rows
    if len(prepared_df) > max_rows:
        prepared_df = prepared_df.head(max_rows)
        metadata["limited"] = True
        metadata["limit"] = max_rows

    # Handle data types for JSON serialization
    for col in prepared_df.columns:
        if pd.api.types.is_datetime64_any_dtype(prepared_df[col]):
            prepared_df[col] = prepared_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        elif pd.api.types.is_numeric_dtype(prepared_df[col]):
            # Convert numpy int64/float64 to Python native types for JSON serialization
            if pd.api.types.is_integer_dtype(prepared_df[col]):
                prepared_df[col] = prepared_df[col].astype("int").astype("object")
            else:
                # Round floats to 4 decimal places for display
                prepared_df[col] = (
                    prepared_df[col].round(4).astype("float").astype("object")
                )

    metadata["final_size"] = len(prepared_df)
    return prepared_df, metadata


def detect_dataset_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect various features and patterns in a dataset to assist with analysis

    Args:
        df: Input dataframe

    Returns:
        Dictionary with detected features and dataset characteristics
    """
    features = {}

    # Basic info
    features["row_count"] = len(df)
    features["column_count"] = df.shape[1]

    # Column types
    column_types = detect_column_types(df)
    features["column_types"] = column_types

    # Time and ID columns
    features["time_columns"] = detect_time_columns(df)
    features["id_columns"] = detect_id_columns(df)

    # Missing values
    missing_counts = df.isnull().sum()
    features["missing_counts"] = missing_counts.to_dict()
    features["missing_percentage"] = (missing_counts / len(df) * 100).to_dict()

    # Detect potential target variables for prediction
    potential_targets = []

    # Numeric columns with few unique values might be targets
    for col in column_types["numeric"]:
        if 2 <= df[col].nunique() <= 15:
            potential_targets.append(
                {
                    "column": col,
                    "type": "numeric_categorical",
                    "unique_values": df[col].nunique(),
                }
            )

    # Boolean columns are usually targets
    for col in df.columns:
        if df[col].dtype == "bool" or (
            df[col].isin([0, 1]).all() and df[col].nunique() == 2
        ):
            potential_targets.append(
                {"column": col, "type": "boolean", "balance": df[col].mean()}
            )

    # Categorical columns with few classes
    for col in column_types["categorical"]:
        if 2 <= df[col].nunique() <= 10:
            potential_targets.append(
                {
                    "column": col,
                    "type": "categorical",
                    "unique_values": df[col].nunique(),
                    "classes": df[col].value_counts().to_dict(),
                }
            )

    features["potential_targets"] = potential_targets

    # Detect potential data patterns
    features["patterns"] = {}

    # Check if could be time series
    if features["time_columns"] and column_types["numeric"]:
        features["patterns"]["time_series"] = True
        features["patterns"]["time_series_columns"] = column_types["numeric"]

    # Check if could be transaction/event data
    if len(features["time_columns"]) >= 1 and len(column_types["categorical"]) >= 1:
        features["patterns"]["event_data"] = True

    # Check if could be customer/user data
    user_patterns = ["user", "customer", "client", "person", "employee", "student"]
    if any(
        any(pattern in col.lower() for pattern in user_patterns) for col in df.columns
    ):
        features["patterns"]["user_data"] = True

    return features
