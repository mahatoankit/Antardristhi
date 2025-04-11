import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
import os
from datetime import datetime
import io
import base64

# Import utility modules
from ..utils.data_preprocessing import (
    detect_column_types,
    handle_missing_values,
    normalize_data,
    detect_time_columns,
    detect_id_columns,
    prepare_time_series_data,
)
from ..utils.time_series import (
    auto_forecast,
    forecast_with_prophet,
    forecast_with_arima,
    simple_trend_analysis,
)
from ..utils.clustering import (
    auto_segment,
    segment_with_kmeans,
    segment_with_dbscan,
    simple_segmentation,
)
from ..utils.anomaly_detection import (
    auto_detect_outliers,
    detect_outliers_isolation_forest,
    detect_outliers_lof,
    detect_outliers_statistical,
)
from ..utils.nlp_processing import (
    process_natural_language_query,
    generate_llm_explanation,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory for models
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "cache"
)
os.makedirs(CACHE_DIR, exist_ok=True)


class AnalysisEngine:
    """
    Coordinating service for ML-powered data analysis
    """

    def __init__(self):
        """Initialize the analysis engine"""
        self.preprocessed_data = {}
        self.analysis_results = {}
        self.active_models = {}

    def preprocess_data(
        self, df: pd.DataFrame, missing_strategy: str = "auto"
    ) -> Dict[str, Any]:
        """
        Preprocess data before analysis

        Args:
            df: Input dataframe
            missing_strategy: Strategy for handling missing values

        Returns:
            Dictionary with preprocessing results
        """
        try:
            # Generate a unique ID for this dataset
            data_id = f"data_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Make a copy of the dataframe to avoid modifying the original
            df_copy = df.copy()

            # Detect column types
            column_types = detect_column_types(df_copy)

            # Handle missing values
            df_processed, imputation_info = handle_missing_values(
                df_copy, strategy=missing_strategy
            )

            # Detect time and ID columns
            time_columns = detect_time_columns(df_processed)
            id_columns = detect_id_columns(df_processed)

            # Store the preprocessed data
            self.preprocessed_data[data_id] = {
                "df": df_processed,
                "column_types": column_types,
                "imputation_info": imputation_info,
                "time_columns": time_columns,
                "id_columns": id_columns,
                "shape": df_processed.shape,
                "timestamp": datetime.now(),
            }

            # Return basic info without the dataframe itself
            result = {
                "data_id": data_id,
                "rows": df_processed.shape[0],
                "columns": df_processed.shape[1],
                "column_types": column_types,
                "missing_values_handled": imputation_info,
                "detected_time_columns": time_columns,
                "detected_id_columns": id_columns,
            }

            return result

        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            return {"error": str(e)}

    def analyze_data_with_prompt(self, data_id: str, prompt: str) -> Dict[str, Any]:
        """
        Analyze data based on natural language prompt

        Args:
            data_id: ID of preprocessed dataset
            prompt: Natural language prompt describing the analysis

        Returns:
            Dictionary with analysis results
        """
        try:
            # Check if data exists
            if data_id not in self.preprocessed_data:
                return {"error": f"Dataset with ID {data_id} not found"}

            # Get preprocessed data
            data_info = self.preprocessed_data[data_id]
            df = data_info["df"]

            # Process the natural language query
            query_analysis = process_natural_language_query(prompt, df.columns.tolist())

            # Extract analysis plan
            plan = query_analysis["analysis_plan"]

            # Execute the appropriate analysis based on the intent
            result = self._execute_analysis_plan(df, plan)

            # Generate explanation
            if "error" not in result:
                explanation = generate_llm_explanation(result, plan)
                result["explanation"] = explanation

            # Store result
            analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.analysis_results[analysis_id] = {
                "query_analysis": query_analysis,
                "result": result,
                "data_id": data_id,
                "timestamp": datetime.now(),
            }

            # Return results with analysis ID
            return {
                "analysis_id": analysis_id,
                "query_analysis": query_analysis,
                "result": result,
            }

        except Exception as e:
            logger.error(f"Error in prompt-based analysis: {str(e)}")
            return {"error": str(e)}

    def analyze_with_time_series(
        self,
        data_id: str,
        date_col: Optional[str] = None,
        value_col: Optional[str] = None,
        periods: int = 30,
    ) -> Dict[str, Any]:
        """
        Perform time series analysis and forecasting

        Args:
            data_id: ID of preprocessed dataset
            date_col: Date column name (auto-detected if None)
            value_col: Value column to forecast (auto-detected if None)
            periods: Number of periods to forecast

        Returns:
            Dictionary with time series analysis results
        """
        try:
            # Check if data exists
            if data_id not in self.preprocessed_data:
                return {"error": f"Dataset with ID {data_id} not found"}

            # Get preprocessed data
            data_info = self.preprocessed_data[data_id]
            df = data_info["df"]

            # Auto-detect date column if not provided
            if date_col is None and data_info["time_columns"]:
                date_col = data_info["time_columns"][0]

            # Auto-detect value column if not provided
            if value_col is None:
                numeric_cols = data_info["column_types"]["numeric"]
                if numeric_cols:
                    # Select first numeric column that's not the date
                    value_col = next(
                        (col for col in numeric_cols if col != date_col), None
                    )

            if date_col is None:
                return {"error": "No date column found. Please specify a date column."}

            if value_col is None:
                return {
                    "error": "No numeric column found for forecasting. Please specify a value column."
                }

            # Prepare time series data
            try:
                ts_df = prepare_time_series_data(df, date_col, value_col)
            except Exception as e:
                return {"error": f"Error preparing time series data: {str(e)}"}

            # Perform forecasting
            result = auto_forecast(ts_df, date_col, value_col, periods)

            # Store result
            analysis_id = f"timeseries_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.analysis_results[analysis_id] = {
                "type": "time_series",
                "result": result,
                "data_id": data_id,
                "params": {
                    "date_col": date_col,
                    "value_col": value_col,
                    "periods": periods,
                },
                "timestamp": datetime.now(),
            }

            return {
                "analysis_id": analysis_id,
                "type": "time_series",
                "parameters": {
                    "date_col": date_col,
                    "value_col": value_col,
                    "periods": periods,
                },
                "result": result,
            }

        except Exception as e:
            logger.error(f"Error in time series analysis: {str(e)}")
            return {"error": str(e)}

    def analyze_with_clustering(
        self,
        data_id: str,
        features: Optional[List[str]] = None,
        n_clusters: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform clustering/segmentation analysis

        Args:
            data_id: ID of preprocessed dataset
            features: Features to use for clustering (auto-selected if None)
            n_clusters: Number of clusters (auto-determined if None)

        Returns:
            Dictionary with clustering results
        """
        try:
            # Check if data exists
            if data_id not in self.preprocessed_data:
                return {"error": f"Dataset with ID {data_id} not found"}

            # Get preprocessed data
            data_info = self.preprocessed_data[data_id]
            df = data_info["df"]

            # Auto-select features if not provided
            if features is None:
                numeric_cols = data_info["column_types"]["numeric"]
                if numeric_cols:
                    features = numeric_cols[:5]  # Use up to 5 numeric features

            if not features:
                return {"error": "No suitable features found for clustering"}

            # Perform clustering
            result = auto_segment(df, features, n_clusters)

            # Store result
            analysis_id = f"clustering_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.analysis_results[analysis_id] = {
                "type": "clustering",
                "result": result,
                "data_id": data_id,
                "params": {
                    "features": features,
                    "n_clusters": (
                        n_clusters if n_clusters else result.get("n_clusters")
                    ),
                },
                "timestamp": datetime.now(),
            }

            return {
                "analysis_id": analysis_id,
                "type": "clustering",
                "parameters": {
                    "features": features,
                    "n_clusters": (
                        n_clusters if n_clusters else result.get("n_clusters")
                    ),
                },
                "result": result,
            }

        except Exception as e:
            logger.error(f"Error in clustering analysis: {str(e)}")
            return {"error": str(e)}

    def analyze_with_anomaly_detection(
        self,
        data_id: str,
        features: Optional[List[str]] = None,
        contamination: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Perform anomaly/outlier detection

        Args:
            data_id: ID of preprocessed dataset
            features: Features to use for outlier detection (auto-selected if None)
            contamination: Expected proportion of outliers

        Returns:
            Dictionary with outlier detection results
        """
        try:
            # Check if data exists
            if data_id not in self.preprocessed_data:
                return {"error": f"Dataset with ID {data_id} not found"}

            # Get preprocessed data
            data_info = self.preprocessed_data[data_id]
            df = data_info["df"]

            # Auto-select features if not provided
            if features is None:
                numeric_cols = data_info["column_types"]["numeric"]
                if numeric_cols:
                    features = numeric_cols  # Use all numeric features

            if not features:
                return {"error": "No suitable features found for anomaly detection"}

            # Perform anomaly detection
            result = auto_detect_outliers(df, features, contamination)

            # Store result
            analysis_id = f"anomaly_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.analysis_results[analysis_id] = {
                "type": "anomaly_detection",
                "result": result,
                "data_id": data_id,
                "params": {"features": features, "contamination": contamination},
                "timestamp": datetime.now(),
            }

            return {
                "analysis_id": analysis_id,
                "type": "anomaly_detection",
                "parameters": {"features": features, "contamination": contamination},
                "result": result,
            }

        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return {"error": str(e)}

    def _execute_analysis_plan(
        self, df: pd.DataFrame, plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute an analysis plan generated from NLP processing

        Args:
            df: Input dataframe
            plan: Analysis plan from NLP processing

        Returns:
            Dictionary with analysis results
        """
        analysis_type = plan.get("analysis_type")
        parameters = plan.get("parameters", {})
        columns = plan.get("columns_to_use", [])

        # Execute based on analysis type
        if analysis_type == "time_series":
            date_col = parameters.get("date_col")
            value_col = parameters.get("value_col")
            periods = parameters.get("periods", 30)

            if date_col and value_col:
                return auto_forecast(df, date_col, value_col, periods)
            else:
                return {"error": "Missing required time series parameters"}

        elif analysis_type == "clustering":
            n_clusters = parameters.get("n_clusters")

            if columns:
                return auto_segment(df, columns, n_clusters)
            else:
                return {"error": "No columns specified for clustering"}

        elif analysis_type == "anomaly_detection":
            contamination = parameters.get("contamination", 0.05)

            if columns:
                return auto_detect_outliers(df, columns, contamination)
            else:
                return {"error": "No columns specified for anomaly detection"}

        elif analysis_type == "summary":
            # Basic statistics
            result = {
                "statistics": (
                    df[columns].describe().to_dict()
                    if columns
                    else df.describe().to_dict()
                ),
                "shape": df.shape,
                "column_types": detect_column_types(df),
                "missing_values": (
                    df[columns].isnull().sum().to_dict()
                    if columns
                    else df.isnull().sum().to_dict()
                ),
            }

            # Add correlation matrix if multiple numeric columns
            numeric_cols = [
                col for col in columns if pd.api.types.is_numeric_dtype(df[col])
            ]
            if len(numeric_cols) > 1:
                result["correlations"] = df[numeric_cols].corr().to_dict()

            return result

        elif analysis_type == "correlation":
            # Correlation analysis
            numeric_cols = [
                col for col in columns if pd.api.types.is_numeric_dtype(df[col])
            ]
            if len(numeric_cols) > 1:
                return {
                    "correlations": df[numeric_cols].corr().to_dict(),
                    "columns_analyzed": numeric_cols,
                }
            else:
                return {"error": "Not enough numeric columns for correlation analysis"}

        elif analysis_type == "distribution":
            # Distribution analysis
            if not columns:
                return {"error": "No columns specified for distribution analysis"}

            target_col = columns[0]
            group_by = parameters.get("group_by")

            result = {
                "statistics": df[target_col].describe().to_dict(),
                "column": target_col,
            }

            # If grouping is specified
            if group_by and group_by in df.columns:
                result["group_statistics"] = (
                    df.groupby(group_by)[target_col].describe().to_dict()
                )

            return result

        elif analysis_type == "ranking":
            # Ranking analysis
            rank_by = parameters.get("rank_by")
            group_by = parameters.get("group_by")
            top_n = parameters.get("top_n", 10)
            ascending = parameters.get("ascending", False)

            if not rank_by or not group_by:
                return {"error": "Missing required ranking parameters"}

            if rank_by not in df.columns or group_by not in df.columns:
                return {"error": "Specified columns not found in dataset"}

            # Group and aggregate
            grouped = (
                df.groupby(group_by)[rank_by]
                .agg(["sum", "mean", "count"])
                .reset_index()
            )

            # Sort by sum (most common ranking metric)
            sorted_data = grouped.sort_values("sum", ascending=ascending).head(top_n)

            return {
                "ranking": sorted_data.to_dict("records"),
                "rank_by": rank_by,
                "group_by": group_by,
                "direction": "bottom" if ascending else "top",
                "limit": top_n,
            }

        elif analysis_type == "comparison":
            # Comparison analysis
            compare_by = parameters.get("compare_by")

            if not compare_by or compare_by not in df.columns:
                return {"error": "Invalid comparison column"}

            metrics = [col for col in columns if col != compare_by]
            if not metrics:
                return {"error": "No metrics specified for comparison"}

            # Group and aggregate by comparison column
            result = {
                "comparison_by": compare_by,
                "metrics": metrics,
                "comparison_data": df.groupby(compare_by)[metrics]
                .agg(["mean", "sum", "count"])
                .to_dict(),
            }

            return result

        else:
            return {"error": f"Unsupported analysis type: {analysis_type}"}

    def get_visualization_data(self, analysis_id: str, viz_type: str) -> Dict[str, Any]:
        """
        Generate visualization data for a specific analysis

        Args:
            analysis_id: ID of analysis result
            viz_type: Type of visualization (e.g., 'line', 'bar', 'scatter')

        Returns:
            Dictionary with visualization data
        """
        try:
            # Check if analysis exists
            if analysis_id not in self.analysis_results:
                return {"error": f"Analysis with ID {analysis_id} not found"}

            analysis_info = self.analysis_results[analysis_id]
            result = analysis_info["result"]

            # Extract visualization data based on analysis type
            if "plots" in result and viz_type in result["plots"]:
                # If plot data already exists, return it
                return {"plot_data": result["plots"][viz_type]}

            # If plots don't exist, analysis type specific fallback
            analysis_type = analysis_info.get("type")

            if analysis_type == "time_series":
                # For time series, generate line chart
                if viz_type == "line" and "forecast" in result:
                    # Visualization data already exists in result
                    return {"plot_data": result.get("plot", "")}

            # Add more visualization types as needed

            return {
                "error": f"Visualization type {viz_type} not available for this analysis"
            }

        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return {"error": str(e)}

    def get_analysis_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of all analyses with basic info

        Returns:
            List of analysis summaries
        """
        analyses = []
        for analysis_id, analysis_info in self.analysis_results.items():
            analyses.append(
                {
                    "analysis_id": analysis_id,
                    "type": analysis_info.get("type", "unknown"),
                    "data_id": analysis_info.get("data_id"),
                    "timestamp": (
                        analysis_info.get("timestamp").isoformat()
                        if analysis_info.get("timestamp")
                        else None
                    ),
                    "parameters": analysis_info.get("params", {}),
                }
            )

        return analyses

    def get_data_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of all datasets with basic info

        Returns:
            List of dataset summaries
        """
        datasets = []
        for data_id, data_info in self.preprocessed_data.items():
            datasets.append(
                {
                    "data_id": data_id,
                    "rows": data_info["shape"][0],
                    "columns": data_info["shape"][1],
                    "timestamp": (
                        data_info.get("timestamp").isoformat()
                        if data_info.get("timestamp")
                        else None
                    ),
                }
            )

        return datasets

    def clear_cache(self, older_than_days: int = 1) -> Dict[str, int]:
        """
        Clear cached data and analysis results older than specified days

        Args:
            older_than_days: Clear items older than this many days

        Returns:
            Dictionary with counts of items cleared
        """
        cutoff_time = datetime.now() - pd.Timedelta(days=older_than_days)

        # Clear old data
        data_cleared = 0
        data_to_remove = []
        for data_id, data_info in self.preprocessed_data.items():
            if data_info.get("timestamp", datetime.now()) < cutoff_time:
                data_to_remove.append(data_id)
                data_cleared += 1

        for data_id in data_to_remove:
            del self.preprocessed_data[data_id]

        # Clear old analyses
        analysis_cleared = 0
        analysis_to_remove = []
        for analysis_id, analysis_info in self.analysis_results.items():
            if analysis_info.get("timestamp", datetime.now()) < cutoff_time:
                analysis_to_remove.append(analysis_id)
                analysis_cleared += 1

        for analysis_id in analysis_to_remove:
            del self.analysis_results[analysis_id]

        return {"data_cleared": data_cleared, "analysis_cleared": analysis_cleared}
