import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
import os
from datetime import datetime
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import json
import traceback

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

    # Message type constants for consistent chatbot response format
    MESSAGE_TYPE_TEXT = "text"
    MESSAGE_TYPE_CHART = "chart"
    MESSAGE_TYPE_TABLE = "table"
    MESSAGE_TYPE_ERROR = "error"

    def __init__(self):
        """Initialize the analysis engine"""
        self.preprocessed_data = {}
        self.analysis_results = {}
        self.active_models = {}
        # Response Message Types
        self.MESSAGE_TYPE_TEXT = "text"
        self.MESSAGE_TYPE_CHART = "chart"
        self.MESSAGE_TYPE_TABLE = "table"
        self.MESSAGE_TYPE_ERROR = "error"

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
        Analyze data based on natural language prompt and return detailed answers.

        Args:
            data_id: ID of preprocessed dataset
            prompt: Natural language prompt describing the analysis

        Returns:
            Dictionary with analysis results and chat response
        """
        try:
            # Check if data exists
            if data_id not in self.preprocessed_data:
                return {"error": f"Dataset with ID {data_id} not found"}

            # Get preprocessed data
            data_info = self.preprocessed_data[data_id]
            df = data_info["df"]

            # Detect dataset type
            detection = self.detect_dataset_type(df)
            dataset_type = detection["dataset_type"]
            column_types = detection["column_types"]

            # Perform analysis based on dataset type
            if dataset_type == "time_series":
                if not column_types["time"]:
                    return {"error": "Time series analysis requires a time column."}
                result = self._analyze_time_series(df, column_types["time"], prompt)
            elif dataset_type == "numeric":
                result = self._analyze_numeric_data(df, column_types["numeric"], prompt)
            elif dataset_type == "categorical":
                result = self._analyze_categorical_data(
                    df, column_types["categorical"], prompt
                )
            else:
                result = self._analyze_mixed_data(df, column_types, prompt)

            # Generate explanation
            if "error" not in result:
                explanation = generate_llm_explanation(result, prompt)
                result["explanation"] = explanation

            # Generate chat response
            chat_response = self.generate_chat_response(prompt, result)

            return {
                "chat_response": chat_response,
                "result": result,
            }

        except Exception as e:
            logger.error(f"Error in prompt-based analysis: {str(e)}")
            return {"error": str(e)}

    def generate_chat_response(self, prompt: str, result: Dict[str, Any]) -> str:
        """
        Generate a conversational response for chat based on the analysis result.

        Args:
            prompt: User's natural language prompt
            result: Analysis result dictionary

        Returns:
            String response for chat
        """
        if "error" in result:
            return f"Sorry, I couldn't process your request. Error: {result['error']}"

        response = ""

        # Include explanation if available
        if "explanation" in result:
            response += (
                f"Here is the analysis based on your request: {result['explanation']}\n"
            )

        # Include statistics if available
        if "statistics" in result:
            response += "Here are some key statistics from your data:\n"
            stats = result["statistics"]
            for col, col_stats in stats.items():
                response += f"- {col}: Mean = {col_stats.get('mean')}, Std = {col_stats.get('std')}\n"

        # Include correlations if available
        if "correlations" in result:
            response += "Here are the correlations between numeric columns:\n"
            for col, corr_values in result["correlations"].items():
                response += f"- {col}: {corr_values}\n"

        # Default response if no specific details are available
        if not response:
            response = "Here is the analysis of your data. Let me know if you need further details."

        return response

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

    def analyze_cleanliness(self, data_id: str) -> Dict[str, Any]:
        """
        Analyze the cleanliness of the dataset.

        Args:
            data_id: ID of preprocessed dataset

        Returns:
            Dictionary with cleanliness metrics
        """
        try:
            if data_id not in self.preprocessed_data:
                return {"error": f"Dataset with ID {data_id} not found"}

            data_info = self.preprocessed_data[data_id]
            df = data_info["df"]

            # Calculate missing values
            missing_values = df.isnull().sum().to_dict()

            # Detect outliers (example using Z-score)
            outliers = {}
            for col in df.select_dtypes(include=["float64", "int64"]).columns:
                z_scores = (df[col] - df[col].mean()) / df[col].std()
                outliers[col] = (z_scores.abs() > 3).sum()

            return {
                "missing_values": missing_values,
                "outliers": outliers,
            }

        except Exception as e:
            logger.error(f"Error analyzing cleanliness: {str(e)}")
            return {"error": str(e)}

    def analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze missing data in the dataset.

        Args:
            df: DataFrame containing the dataset

        Returns:
            Dictionary with missing data insights
        """
        missing_values = df.isnull().sum().to_dict()
        total_missing = sum(missing_values.values())
        missing_percentage = {
            col: (count / len(df)) * 100 for col, count in missing_values.items()
        }

        return {
            "total_missing": total_missing,
            "missing_values": missing_values,
            "missing_percentage": missing_percentage,
        }

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

    def describe_dataset(self, data_id: str) -> Dict[str, Any]:
        """
        Generate a brief description of the dataset with chat-friendly output.

        Args:
            data_id: ID of preprocessed dataset

        Returns:
            Dictionary with dataset description and chat response
        """
        try:
            if data_id not in self.preprocessed_data:
                return {"error": f"Dataset with ID {data_id} not found"}

            data_info = self.preprocessed_data[data_id]
            df = data_info["df"]

            description = {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_types": data_info["column_types"],
                "missing_values": data_info["imputation_info"],
                "time_columns": data_info["time_columns"],
                "id_columns": data_info["id_columns"],
                "summary_statistics": df.describe(include="all").to_dict(),
            }

            # Generate chat-friendly response
            chat_response = (
                f"The dataset contains {description['rows']} rows and {description['columns']} columns. "
                f"It includes the following column types: {description['column_types']}. "
                f"Missing values have been handled as follows: {description['missing_values']}. "
                f"Time-related columns detected: {description['time_columns']}. "
                f"ID columns detected: {description['id_columns']}."
            )

            return {"description": description, "chat_response": chat_response}

        except Exception as e:
            logger.error(f"Error describing dataset: {str(e)}")
            return {"error": str(e)}

    def suggest_questions(self, data_id: str) -> Dict[str, Any]:
        """
        Suggest questions based on the dataset with chat-friendly output.

        Args:
            data_id: ID of preprocessed dataset

        Returns:
            Dictionary with suggested questions and chat response
        """
        try:
            if data_id not in self.preprocessed_data:
                return {"error": f"Dataset with ID {data_id} not found"}

            data_info = self.preprocessed_data[data_id]
            column_types = data_info["column_types"]

            questions = []
            for col in column_types["numeric"]:
                questions.append(f"What is the distribution of {col}?")
                questions.append(f"Are there any correlations involving {col}?")

            for col in column_types["categorical"]:
                questions.append(f"What are the most common categories in {col}?")
                questions.append(f"How does {col} relate to other columns?")

            if data_info["time_columns"]:
                time_col = data_info["time_columns"][0]
                questions.append(f"What trends can be observed over {time_col}?")
                questions.append(f"Are there any seasonal patterns in {time_col}?")

            # Generate chat-friendly response
            chat_response = (
                "Here are some questions you can ask about your dataset:\n"
                + "\n".join(f"- {q}" for q in questions)
            )

            return {"questions": questions, "chat_response": chat_response}

        except Exception as e:
            logger.error(f"Error suggesting questions: {str(e)}")
            return {"error": str(e)}

    def generate_visualizations(
        self, df: pd.DataFrame, column_types: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Generate visualizations based on the dataset.

        Args:
            df: DataFrame containing the dataset
            column_types: Dictionary with column classifications

        Returns:
            Dictionary with visualization data
        """
        visualizations = {}

        # Generate histograms for numeric columns
        for col in column_types["numeric"]:
            visualizations[f"histogram_{col}"] = df[col].plot(kind="hist").get_figure()

        # Generate bar charts for categorical columns
        for col in column_types["categorical"]:
            visualizations[f"bar_chart_{col}"] = (
                df[col].value_counts().plot(kind="bar").get_figure()
            )

        return visualizations

    def generate_custom_visualization(
        self, data_id: str, chart_type: str, x_col: str, y_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a custom visualization based on user request.

        Args:
            data_id: ID of preprocessed dataset
            chart_type: Type of chart (e.g., 'scatter', 'line', 'bar')
            x_col: X-axis column
            y_col: Y-axis column (optional)

        Returns:
            Dictionary with base64-encoded visualization
        """
        try:
            if data_id not in self.preprocessed_data:
                return {"error": f"Dataset with ID {data_id} not found"}

            data_info = self.preprocessed_data[data_id]
            df = data_info["df"]

            plt.figure(figsize=(8, 6))
            if chart_type == "scatter" and y_col:
                sns.scatterplot(data=df, x=x_col, y=y_col)
                plt.title(f"Scatter Plot of {x_col} vs {y_col}")
            elif chart_type == "line" and y_col:
                sns.lineplot(data=df, x=x_col, y=y_col)
                plt.title(f"Line Plot of {x_col} vs {y_col}")
            elif chart_type == "bar":
                sns.barplot(
                    x=df[x_col].value_counts().index, y=df[x_col].value_counts().values
                )
                plt.title(f"Bar Chart of {x_col}")
            else:
                return {"error": f"Unsupported chart type: {chart_type}"}

            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            visualization = base64.b64encode(buffer.read()).decode("utf-8")
            buffer.close()
            plt.close()

            return {"visualization": visualization}

        except Exception as e:
            logger.error(f"Error generating custom visualization: {str(e)}")
            return {"error": str(e)}

    def _analyze_time_series(
        self, df: pd.DataFrame, time_columns: List[str], prompt: str
    ) -> Dict[str, Any]:
        """
        Perform time series analysis on the dataset.

        Args:
            df: DataFrame containing the dataset
            time_columns: List of time-related columns
            prompt: User's natural language prompt

        Returns:
            Dictionary with time series analysis results
        """
        try:
            # Example: Perform trend analysis or forecasting
            time_col = time_columns[0]
            result = simple_trend_analysis(df, time_col)
            return result
        except Exception as e:
            logger.error(f"Error in time series analysis: {str(e)}")
            return {"error": str(e)}

    def _analyze_categorical_data(
        self, df: pd.DataFrame, categorical_columns: List[str], prompt: str
    ) -> Dict[str, Any]:
        """
        Perform analysis on categorical data.

        Args:
            df: DataFrame containing the dataset
            categorical_columns: List of categorical columns
            prompt: User's natural language prompt

        Returns:
            Dictionary with categorical analysis results
        """
        try:
            category_counts = {
                col: df[col].value_counts().to_dict() for col in categorical_columns
            }
            return {"category_counts": category_counts}
        except Exception as e:
            logger.error(f"Error in categorical data analysis: {str(e)}")
            return {"error": str(e)}

    def detect_dataset_type(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect the type of dataset based on its columns.

        Args:
            df: DataFrame containing the dataset

        Returns:
            Dictionary with detected dataset type and column classifications
        """
        column_types = {
            "numeric": df.select_dtypes(include=["float64", "int64"]).columns.tolist(),
            "categorical": df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist(),
            "time": [
                col
                for col in df.columns
                if pd.api.types.is_datetime64_any_dtype(df[col])
            ],
        }

        dataset_type = "mixed"
        if column_types["time"]:
            dataset_type = "time_series"
        elif column_types["numeric"] and not column_types["categorical"]:
            dataset_type = "numeric"
        elif column_types["categorical"] and not column_types["numeric"]:
            dataset_type = "categorical"

        return {"dataset_type": dataset_type, "column_types": column_types}

    def _analyze_numeric_data(
        self, df: pd.DataFrame, numeric_columns: List[str], prompt: str
    ) -> Dict[str, Any]:
        """
        Perform analysis on numeric data.

        Args:
            df: DataFrame containing the dataset
            numeric_columns: List of numeric columns
            prompt: User's natural language prompt

        Returns:
            Dictionary with numeric analysis results
        """
        try:
            # Basic statistics
            stats = df[numeric_columns].describe().to_dict()

            # Calculate correlations if we have multiple numeric columns
            correlations = {}
            if len(numeric_columns) > 1:
                correlations = df[numeric_columns].corr().to_dict()

            # Identify potential outliers using z-score
            outliers = {}
            for col in numeric_columns:
                if df[col].std() > 0:  # Avoid division by zero
                    z_scores = (df[col] - df[col].mean()) / df[col].std()
                    outliers[col] = (z_scores.abs() > 3).sum()
                else:
                    outliers[col] = 0

            return {
                "statistics": stats,
                "correlations": correlations,
                "outliers": outliers,
                "analysis_type": "numeric",
            }
        except Exception as e:
            logging.error(f"Error in numeric data analysis: {str(e)}")
            return {"error": str(e)}

    def _analyze_mixed_data(
        self, df: pd.DataFrame, column_types: Dict[str, List[str]], prompt: str
    ) -> Dict[str, Any]:
        """
        Perform analysis on mixed data types.

        Args:
            df: DataFrame containing the dataset
            column_types: Dictionary of column types
            prompt: User's natural language prompt

        Returns:
            Dictionary with mixed analysis results
        """
        try:
            result = {"analysis_type": "mixed", "statistics": {}}

            # Analyze numeric columns
            if column_types.get("numeric"):
                numeric_analysis = self._analyze_numeric_data(
                    df, column_types["numeric"], prompt
                )
                if "error" not in numeric_analysis:
                    result["numeric_analysis"] = numeric_analysis

            # Analyze categorical columns
            if column_types.get("categorical"):
                categorical_analysis = self._analyze_categorical_data(
                    df, column_types["categorical"], prompt
                )
                if "error" not in categorical_analysis:
                    result["categorical_analysis"] = categorical_analysis

            # Basic dataset statistics
            result["statistics"] = {
                "row_count": df.shape[0],
                "column_count": df.shape[1],
                "missing_values": df.isnull().sum().to_dict(),
            }

            # Generate a text explanation of the data
            explanation = self._generate_data_explanation(df, column_types, prompt)
            result["text"] = explanation
            result["explanation"] = explanation

            return result
        except Exception as e:
            logging.error(f"Error in mixed data analysis: {str(e)}")
            return {"error": str(e)}

    def _generate_data_explanation(
        self, df: pd.DataFrame, column_types: Dict[str, List[str]], prompt: str
    ) -> str:
        """
        Generate a human-readable explanation of the data analysis.

        Args:
            df: DataFrame containing the dataset
            column_types: Dictionary of column types
            prompt: User's natural language prompt

        Returns:
            A string explanation of the data
        """
        try:
            # Create a basic explanation
            explanation = (
                f"This dataset contains {df.shape[0]} rows and {df.shape[1]} columns. "
            )

            # Add information about numeric columns
            if column_types.get("numeric"):
                explanation += (
                    f"There are {len(column_types['numeric'])} numeric columns. "
                )

                # Add some basic statistics for key numeric columns (up to 3)
                for i, col in enumerate(column_types["numeric"][:3]):
                    explanation += (
                        f"The column '{col}' has a mean of {df[col].mean():.2f} "
                    )
                    explanation += (
                        f"and ranges from {df[col].min():.2f} to {df[col].max():.2f}. "
                    )

            # Add information about categorical columns
            if column_types.get("categorical"):
                explanation += f"There are {len(column_types['categorical'])} categorical columns. "

                # Add some information about categories for key categorical columns (up to 2)
                for i, col in enumerate(column_types["categorical"][:2]):
                    unique_vals = df[col].nunique()
                    explanation += (
                        f"The column '{col}' has {unique_vals} unique values. "
                    )

                    # If there are a reasonable number of categories, list the most common ones
                    if unique_vals <= 10:
                        top_cats = df[col].value_counts().head(3)
                        explanation += f"The most common values are {', '.join([f'{cat} ({count})' for cat, count in top_cats.items()])}. "

            # Add information about missing values
            missing = df.isnull().sum().sum()
            if missing > 0:
                explanation += f"There are {missing} missing values in the dataset. "
                cols_with_missing = df.columns[df.isnull().any()].tolist()
                explanation += (
                    f"Columns with missing values: {', '.join(cols_with_missing)}. "
                )
            else:
                explanation += "The dataset has no missing values. "

            # Tailor the explanation based on the prompt
            prompt_lower = prompt.lower()
            if "distribution" in prompt_lower or "histogram" in prompt_lower:
                explanation += self._add_distribution_explanation(df, column_types)
            elif "correlation" in prompt_lower or "relationship" in prompt_lower:
                explanation += self._add_correlation_explanation(df, column_types)
            elif "trend" in prompt_lower or "time" in prompt_lower:
                explanation += self._add_trend_explanation(df, column_types)

            return explanation
        except Exception as e:
            logging.error(f"Error generating explanation: {str(e)}")
            return "I was unable to generate a detailed explanation due to an error in the analysis."

    def _add_distribution_explanation(
        self, df: pd.DataFrame, column_types: Dict[str, List[str]]
    ) -> str:
        """Add explanation about data distributions"""
        explanation = "\n\nRegarding distributions in the data: "

        # Only analyze numeric columns for distribution
        if column_types.get("numeric"):
            for col in column_types["numeric"][:2]:  # Limit to 2 columns
                skew = df[col].skew()
                if abs(skew) < 0.5:
                    explanation += (
                        f"The '{col}' column has a fairly symmetric distribution. "
                    )
                elif skew > 0:
                    explanation += (
                        f"The '{col}' column is right-skewed (positively skewed). "
                    )
                else:
                    explanation += (
                        f"The '{col}' column is left-skewed (negatively skewed). "
                    )

        return explanation

    def _add_correlation_explanation(
        self, df: pd.DataFrame, column_types: Dict[str, List[str]]
    ) -> str:
        """Add explanation about correlations in the data"""
        explanation = "\n\nRegarding correlations in the data: "

        # Check if we have enough numeric columns for correlation analysis
        if column_types.get("numeric") and len(column_types["numeric"]) > 1:
            corr_matrix = df[column_types["numeric"]].corr()

            # Find strongest correlation (excluding self-correlations)
            strongest_corr = 0
            col1, col2 = "", ""

            for i, c1 in enumerate(corr_matrix.columns):
                for c2 in corr_matrix.columns[i + 1 :]:
                    if abs(corr_matrix.loc[c1, c2]) > abs(strongest_corr):
                        strongest_corr = corr_matrix.loc[c1, c2]
                        col1, col2 = c1, c2

            if col1 and col2:
                if strongest_corr > 0.7:
                    explanation += f"There is a strong positive correlation ({strongest_corr:.2f}) between '{col1}' and '{col2}'. "
                elif strongest_corr > 0.4:
                    explanation += f"There is a moderate positive correlation ({strongest_corr:.2f}) between '{col1}' and '{col2}'. "
                elif strongest_corr > 0.1:
                    explanation += f"There is a weak positive correlation ({strongest_corr:.2f}) between '{col1}' and '{col2}'. "
                elif strongest_corr < -0.7:
                    explanation += f"There is a strong negative correlation ({strongest_corr:.2f}) between '{col1}' and '{col2}'. "
                elif strongest_corr < -0.4:
                    explanation += f"There is a moderate negative correlation ({strongest_corr:.2f}) between '{col1}' and '{col2}'. "
                elif strongest_corr < -0.1:
                    explanation += f"There is a weak negative correlation ({strongest_corr:.2f}) between '{col1}' and '{col2}'. "
                else:
                    explanation += f"There are no strong correlations between numeric variables in this dataset. "
            else:
                explanation += f"There are not enough numeric columns to perform correlation analysis. "
        else:
            explanation += f"There are not enough numeric columns to perform correlation analysis. "

        return explanation

    def _add_trend_explanation(
        self, df: pd.DataFrame, column_types: Dict[str, List[str]]
    ) -> str:
        """Add explanation about trends in the data"""
        explanation = "\n\nRegarding trends in the data: "

        # Check if we have time columns
        if column_types.get("time") and column_types.get("numeric"):
            time_col = column_types["time"][0]
            numeric_col = column_types["numeric"][0]

            explanation += f"To analyze trends, you can examine how '{numeric_col}' changes over '{time_col}'. "
            explanation += f"Consider plotting a line chart with '{time_col}' on the x-axis and '{numeric_col}' on the y-axis. "
        else:
            explanation += f"No time-related columns were detected, so trend analysis may not be applicable. "

        return explanation

    def generate_auto_visualizations(self, file_id: str) -> List[Dict[str, Any]]:
        """
        Generate automatic visualizations for a dataset.

        Args:
            file_id: ID of preprocessed dataset

        Returns:
            List of dictionaries with visualization data
        """
        try:
            if file_id not in self.preprocessed_data:
                return [{"error": f"Dataset with ID {file_id} not found"}]

            data_info = self.preprocessed_data[file_id]
            df = data_info["df"]
            column_types = data_info["column_types"]

            visualizations = []

            # Generate distribution plots for numeric columns (up to 2)
            for i, col in enumerate(column_types.get("numeric", [])[:2]):
                try:
                    plt.figure(figsize=(8, 6))
                    sns.histplot(df[col], kde=True)
                    plt.title(f"Distribution of {col}")
                    plt.tight_layout()

                    # Convert plot to base64 image
                    buffer = BytesIO()
                    plt.savefig(buffer, format="png")
                    buffer.seek(0)
                    image_data = base64.b64encode(buffer.read()).decode("utf-8")
                    buffer.close()
                    plt.close()

                    visualizations.append(
                        {
                            "type": "histogram",
                            "title": f"Distribution of {col}",
                            "image": image_data,
                        }
                    )
                except Exception as e:
                    logging.error(f"Error generating histogram for {col}: {str(e)}")

            # Generate bar charts for categorical columns (up to 2)
            for i, col in enumerate(column_types.get("categorical", [])[:2]):
                try:
                    # Only create charts for columns with reasonable number of categories
                    if df[col].nunique() <= 10:
                        plt.figure(figsize=(8, 6))
                        value_counts = df[col].value_counts().head(10)
                        sns.barplot(x=value_counts.index, y=value_counts.values)
                        plt.xticks(rotation=45)
                        plt.title(f"Counts by {col}")
                        plt.tight_layout()

                        # Convert plot to base64 image
                        buffer = BytesIO()
                        plt.savefig(buffer, format="png")
                        buffer.seek(0)
                        image_data = base64.b64encode(buffer.read()).decode("utf-8")
                        buffer.close()
                        plt.close()

                        visualizations.append(
                            {
                                "type": "bar",
                                "title": f"Counts by {col}",
                                "image": image_data,
                            }
                        )
                except Exception as e:
                    logging.error(f"Error generating bar chart for {col}: {str(e)}")

            # Generate scatter plot if we have at least 2 numeric columns
            if len(column_types.get("numeric", [])) >= 2:
                try:
                    col1 = column_types["numeric"][0]
                    col2 = column_types["numeric"][1]

                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(data=df, x=col1, y=col2)
                    plt.title(f"Relationship between {col1} and {col2}")
                    plt.tight_layout()

                    # Convert plot to base64 image
                    buffer = BytesIO()
                    plt.savefig(buffer, format="png")
                    buffer.seek(0)
                    image_data = base64.b64encode(buffer.read()).decode("utf-8")
                    buffer.close()
                    plt.close()

                    visualizations.append(
                        {
                            "type": "scatter",
                            "title": f"Relationship between {col1} and {col2}",
                            "image": image_data,
                        }
                    )
                except Exception as e:
                    logging.error(f"Error generating scatter plot: {str(e)}")

            # If we have a date/time column and at least one numeric column
            if column_types.get("time") and column_types.get("numeric"):
                try:
                    time_col = column_types["time"][0]
                    numeric_col = column_types["numeric"][0]

                    plt.figure(figsize=(10, 6))
                    plt.plot(df[time_col], df[numeric_col])
                    plt.title(f"Trend of {numeric_col} over time")
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    # Convert plot to base64 image
                    buffer = BytesIO()
                    plt.savefig(buffer, format="png")
                    buffer.seek(0)
                    image_data = base64.b64encode(buffer.read()).decode("utf-8")
                    buffer.close()
                    plt.close()

                    visualizations.append(
                        {
                            "type": "line",
                            "title": f"Trend of {numeric_col} over time",
                            "image": image_data,
                        }
                    )
                except Exception as e:
                    logging.error(f"Error generating time series plot: {str(e)}")

            return visualizations
        except Exception as e:
            logging.error(f"Error generating auto visualizations: {str(e)}")
            return [{"error": str(e)}]

    def format_table_data(self, df: pd.DataFrame, max_rows: int = 10) -> Dict[str, Any]:
        """
        Format dataframe as table data for chatbot display.

        Args:
            df: DataFrame to format
            max_rows: Maximum number of rows to include

        Returns:
            Dictionary with table data in a format suitable for frontend display
        """
        try:
            # Limit number of rows
            if len(df) > max_rows:
                display_df = df.head(max_rows)
            else:
                display_df = df

            # Convert to table data format
            columns = display_df.columns.tolist()
            rows = display_df.to_dict("records")

            # Format any nan values as empty strings
            for row in rows:
                for key, value in row.items():
                    if pd.isna(value):
                        row[key] = ""
                    elif isinstance(value, (np.float64, float)):
                        row[key] = round(value, 4)  # Round floats for better display

            return {
                "type": self.MESSAGE_TYPE_TABLE,
                "data": {
                    "columns": columns,
                    "rows": rows,
                    "totalRows": len(df),
                    "displayedRows": len(display_df),
                },
            }
        except Exception as e:
            logger.error(f"Error formatting table data: {str(e)}")
            return {
                "type": self.MESSAGE_TYPE_ERROR,
                "data": {"error": f"Error formatting table: {str(e)}"},
            }

    def format_chart_data(self, visualization: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format chart data for chatbot display.

        Args:
            visualization: Dictionary with visualization data (must include 'image' field)

        Returns:
            Dictionary with chart data in a format suitable for frontend display
        """
        try:
            if "image" not in visualization and "imageData" not in visualization:
                return {
                    "type": self.MESSAGE_TYPE_ERROR,
                    "data": {"error": "Invalid visualization: no image data"},
                }

            # If imageData is already present, use it directly
            if "imageData" in visualization:
                return {
                    "type": self.MESSAGE_TYPE_CHART,
                    "data": {
                        "chartType": visualization.get("type", "unknown"),
                        "title": visualization.get("title", "Chart"),
                        "imageData": visualization["imageData"],
                    },
                }

            # Otherwise, convert the image to base64
            return {
                "type": self.MESSAGE_TYPE_CHART,
                "data": {
                    "chartType": visualization.get("type", "unknown"),
                    "title": visualization.get("title", "Chart"),
                    "imageData": f"data:image/png;base64,{visualization['image']}",
                },
            }
        except Exception as e:
            logger.error(f"Error formatting chart data: {str(e)}")
            return {
                "type": self.MESSAGE_TYPE_ERROR,
                "data": {"error": f"Error formatting chart: {str(e)}"},
            }

    def format_text_response(self, text: str) -> Dict[str, Any]:
        """
        Format text response for chatbot display.

        Args:
            text: Text content

        Returns:
            Dictionary with text data in a format suitable for frontend display
        """
        return {"type": self.MESSAGE_TYPE_TEXT, "data": {"text": text}}

    def format_error_response(self, error: str) -> Dict[str, Any]:
        """
        Format error response for chatbot display.

        Args:
            error: Error message

        Returns:
            Dictionary with error data in a format suitable for frontend display
        """
        return {"type": self.MESSAGE_TYPE_ERROR, "data": {"error": error}}

    def format_complex_response(
        self, result: Dict[str, Any], query: str
    ) -> List[Dict[str, Any]]:
        """
        Create a complex chat response with multiple message types (text, tables, charts).

        Args:
            result: Analysis result dictionary
            query: Original user query

        Returns:
            List of message objects for the chatbot
        """
        try:
            messages = []

            # Add text explanation if available
            if "explanation" in result or "text" in result:
                text = result.get("explanation", result.get("text", ""))
                messages.append(self.format_text_response(text))

            # Add chart messages if available
            if "charts" in result and isinstance(result["charts"], list):
                for chart in result["charts"]:
                    if "imageData" in chart:
                        # Chart already in the right format
                        messages.append(
                            {"type": self.MESSAGE_TYPE_CHART, "data": chart}
                        )
                    elif "image" in chart:
                        # Convert to the right format
                        messages.append(self.format_chart_data(chart))

            # Support for visualization in different formats
            if "visualization" in result and result["visualization"]:
                if isinstance(result["visualization"], dict):
                    messages.append(self.format_chart_data(result["visualization"]))
                elif isinstance(result["visualization"], str) and result[
                    "visualization"
                ].startswith("data:image"):
                    messages.append(
                        {
                            "type": self.MESSAGE_TYPE_CHART,
                            "data": {
                                "chartType": "generic",
                                "title": "Visualization",
                                "imageData": result["visualization"],
                            },
                        }
                    )

            # Add table data if available
            if "tableData" in result and isinstance(result["tableData"], dict):
                # Data already formatted for table
                messages.append(
                    {"type": self.MESSAGE_TYPE_TABLE, "data": result["tableData"]}
                )
            elif "dataframe" in result:
                # Convert DataFrame to table format
                try:
                    df = pd.DataFrame(result["dataframe"])
                    messages.append(self.format_table_data(df))
                except Exception as e:
                    logger.error(f"Error converting dataframe to table: {str(e)}")

            # Add statistics as a table if available and no other tables
            if (
                not any(m["type"] == self.MESSAGE_TYPE_TABLE for m in messages)
                and "statistics" in result
            ):
                try:
                    # Convert statistics to a table
                    stats_df = pd.DataFrame(result["statistics"])
                    messages.append(self.format_table_data(stats_df))
                except Exception as e:
                    logger.error(f"Error converting statistics to table: {str(e)}")

            # If no messages were added, add a default message
            if not messages:
                messages.append(
                    self.format_text_response(
                        "I've analyzed your data but couldn't generate a specific response. "
                        "Try asking more specific questions about your dataset."
                    )
                )

            return messages
        except Exception as e:
            logger.error(f"Error formatting complex response: {str(e)}")
            return [self.format_error_response(f"Error processing result: {str(e)}")]

    def analyze_time_series(self, df: pd.DataFrame, prompt: str) -> Dict[str, Any]:
        """
        Perform time series analysis on the dataset.

        Args:
            df: DataFrame containing the dataset
            prompt: User's natural language prompt

        Returns:
            Dictionary with time series analysis results
        """
        try:
            # Detect time columns
            time_columns = detect_time_columns(df)
            if not time_columns:
                return {
                    "error": "No time columns detected in the dataset. Time series analysis requires date or time data."
                }

            time_col = time_columns[0]  # Use first time column

            # Identify target columns - use numeric columns that aren't time
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            target_cols = [col for col in numeric_cols if col not in time_columns]

            if not target_cols:
                return {
                    "error": "No numeric columns found to forecast. Time series analysis requires numeric data."
                }

            target_col = target_cols[0]  # Use first numeric column by default

            # Look for target column in prompt
            for col in target_cols:
                if col.lower() in prompt.lower():
                    target_col = col
                    break

            # Prepare time series data
            ts_data = prepare_time_series_data(df, time_col, target_col)

            # Determine forecast periods based on prompt
            forecast_periods = 10  # Default
            period_words = re.findall(r"next\s+(\d+)", prompt.lower())
            if period_words:
                try:
                    forecast_periods = int(period_words[0])
                except ValueError:
                    pass

            # Perform forecast
            forecast_result = auto_forecast(ts_data, forecast_periods)

            # Generate visualization
            plt.figure(figsize=(10, 6))
            plt.plot(ts_data.index, ts_data.values, label="Historical")
            plt.plot(
                forecast_result["forecast_index"],
                forecast_result["forecast_values"],
                label="Forecast",
                linestyle="--",
            )
            plt.title(f"Time Series Forecast of {target_col}")
            plt.xlabel(time_col)
            plt.ylabel(target_col)
            plt.legend()
            plt.grid(True)

            # Save plot to bytes
            img_bytes = BytesIO()
            plt.savefig(img_bytes, format="png")
            plt.close()
            img_bytes.seek(0)
            img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

            # Prepare result
            result = {
                "analysis_type": "time_series",
                "target_column": target_col,
                "time_column": time_col,
                "forecast_periods": forecast_periods,
                "forecast": forecast_result["forecast_values"].tolist(),
                "text": f"I've analyzed the time series data for {target_col} over {time_col} and generated a forecast for the next {forecast_periods} periods.",
                "charts": [
                    {
                        "type": "line",
                        "title": f"Time Series Forecast of {target_col}",
                        "image": img_base64,
                    }
                ],
            }

            return result

        except Exception as e:
            logger.error(f"Error in time series analysis: {str(e)}")
            return {"error": f"Error in time series analysis: {str(e)}"}

    def analyze_correlations(self, df: pd.DataFrame, prompt: str) -> Dict[str, Any]:
        """
        Analyze correlations between variables in the dataset.

        Args:
            df: DataFrame containing the dataset
            prompt: User's natural language prompt

        Returns:
            Dictionary with correlation analysis results
        """
        try:
            # Filter for numeric columns only
            numeric_df = df.select_dtypes(include=["number"])

            if numeric_df.empty or numeric_df.shape[1] < 2:
                return {
                    "error": "Not enough numeric columns for correlation analysis. Need at least 2 numeric columns."
                }

            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()

            # Find strongest correlations (positive and negative)
            strongest_pos = 0
            strongest_neg = 0
            pos_cols = ("", "")
            neg_cols = ("", "")

            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # Upper triangle only (avoid duplicates)
                        corr_val = corr_matrix.loc[col1, col2]
                        if corr_val > strongest_pos:
                            strongest_pos = corr_val
                            pos_cols = (col1, col2)
                        elif corr_val < strongest_neg:
                            strongest_neg = corr_val
                            neg_cols = (col1, col2)

            # Generate scatter plots for the strongest correlations
            charts = []

            # Positive correlation plot
            if strongest_pos > 0.1:
                plt.figure(figsize=(8, 6))
                plt.scatter(df[pos_cols[0]], df[pos_cols[1]], alpha=0.6)
                plt.title(
                    f"Correlation: {pos_cols[0]} vs {pos_cols[1]} (r={strongest_pos:.2f})"
                )
                plt.xlabel(pos_cols[0])
                plt.ylabel(pos_cols[1])
                plt.grid(True)

                # Save plot to bytes
                img_bytes = BytesIO()
                plt.savefig(img_bytes, format="png")
                plt.close()
                img_bytes.seek(0)
                img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

                charts.append(
                    {
                        "type": "scatter",
                        "title": f"Positive Correlation: {pos_cols[0]} vs {pos_cols[1]}",
                        "image": img_base64,
                    }
                )

            # Negative correlation plot
            if strongest_neg < -0.1:
                plt.figure(figsize=(8, 6))
                plt.scatter(df[neg_cols[0]], df[neg_cols[1]], alpha=0.6, color="red")
                plt.title(
                    f"Correlation: {neg_cols[0]} vs {neg_cols[1]} (r={strongest_neg:.2f})"
                )
                plt.xlabel(neg_cols[0])
                plt.ylabel(neg_cols[1])
                plt.grid(True)

                # Save plot to bytes
                img_bytes = BytesIO()
                plt.savefig(img_bytes, format="png")
                plt.close()
                img_bytes.seek(0)
                img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

                charts.append(
                    {
                        "type": "scatter",
                        "title": f"Negative Correlation: {neg_cols[0]} vs {neg_cols[1]}",
                        "image": img_base64,
                    }
                )

            # Heatmap of correlation matrix
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            sns.heatmap(
                corr_matrix,
                mask=mask,
                cmap=cmap,
                vmax=1,
                vmin=-1,
                center=0,
                square=True,
                linewidths=0.5,
                annot=True,
                fmt=".2f",
            )
            plt.title("Correlation Heatmap")

            # Save plot to bytes
            img_bytes = BytesIO()
            plt.savefig(img_bytes, format="png")
            plt.close()
            img_bytes.seek(0)
            img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

            charts.append(
                {"type": "heatmap", "title": "Correlation Heatmap", "image": img_base64}
            )

            # Create table of top correlations
            top_corrs = []
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # Upper triangle only
                        corr_val = corr_matrix.loc[col1, col2]
                        if abs(corr_val) > 0.3:  # Only include meaningful correlations
                            top_corrs.append(
                                {
                                    "Variable 1": col1,
                                    "Variable 2": col2,
                                    "Correlation": corr_val,
                                }
                            )

            top_corrs_df = pd.DataFrame(top_corrs).sort_values(
                "Correlation", key=abs, ascending=False
            )

            # Create explanation text
            text = "I've analyzed the correlations between numeric variables in your dataset.\n\n"

            if strongest_pos > 0.7:
                text += f"There is a strong positive correlation ({strongest_pos:.2f}) between {pos_cols[0]} and {pos_cols[1]}.\n"
            elif strongest_pos > 0.4:
                text += f"There is a moderate positive correlation ({strongest_pos:.2f}) between {pos_cols[0]} and {pos_cols[1]}.\n"
            elif strongest_pos > 0.1:
                text += f"There is a weak positive correlation ({strongest_pos:.2f}) between {pos_cols[0]} and {pos_cols[1]}.\n"

            if strongest_neg < -0.7:
                text += f"There is a strong negative correlation ({strongest_neg:.2f}) between {neg_cols[0]} and {neg_cols[1]}.\n"
            elif strongest_neg < -0.4:
                text += f"There is a moderate negative correlation ({strongest_neg:.2f}) between {neg_cols[0]} and {neg_cols[1]}.\n"
            elif strongest_neg < -0.1:
                text += f"There is a weak negative correlation ({strongest_neg:.2f}) between {neg_cols[0]} and {neg_cols[1]}.\n"

            text += "\nThe correlation heatmap shows relationships between all numeric variables."

            return {
                "analysis_type": "correlation",
                "text": text,
                "charts": charts,
                "dataframe": (
                    top_corrs_df.to_dict("records") if not top_corrs_df.empty else None
                ),
                "statistics": {"correlation_matrix": corr_matrix.to_dict()},
            }

        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return {"error": f"Error in correlation analysis: {str(e)}"}

    def analyze_clustering(self, df: pd.DataFrame, prompt: str) -> Dict[str, Any]:
        """
        Perform clustering analysis on the dataset.

        Args:
            df: DataFrame containing the dataset
            prompt: User's natural language prompt

        Returns:
            Dictionary with clustering analysis results
        """
        try:
            # Filter for numeric columns only for clustering
            numeric_df = df.select_dtypes(include=["number"])

            if numeric_df.empty or numeric_df.shape[1] < 2:
                return {
                    "error": "Not enough numeric columns for clustering analysis. Need at least 2 numeric features."
                }

            # Normalize data for clustering
            normalized_data, scaler = normalize_data(numeric_df)

            # Determine number of clusters based on prompt or auto-detect
            n_clusters = 3  # Default
            cluster_words = re.findall(r"(\d+)\s+clusters", prompt.lower())
            if cluster_words:
                try:
                    n_clusters = int(cluster_words[0])
                except ValueError:
                    pass

            # Perform clustering
            clustering_result = segment_with_kmeans(
                normalized_data, n_clusters=n_clusters
            )

            # Add cluster labels to original dataframe
            df_with_clusters = df.copy()
            df_with_clusters["Cluster"] = clustering_result["labels"]

            # Calculate cluster statistics
            cluster_stats = []
            for i in range(n_clusters):
                cluster_df = df_with_clusters[df_with_clusters["Cluster"] == i]
                cluster_stat = {
                    "Cluster": i,
                    "Size": len(cluster_df),
                    "Percentage": len(cluster_df) / len(df) * 100,
                }

                # Add statistics for each numeric column
                for col in numeric_df.columns:
                    cluster_stat[f"{col}_mean"] = cluster_df[col].mean()

                cluster_stats.append(cluster_stat)

            cluster_stats_df = pd.DataFrame(cluster_stats)

            # Generate PCA visualization for clusters
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(normalized_data)

            # Plot clusters
            plt.figure(figsize=(10, 8))
            colors = [
                "royalblue",
                "forestgreen",
                "firebrick",
                "darkorange",
                "purple",
                "teal",
                "coral",
                "gold",
                "slategray",
                "deeppink",
            ]

            for i in range(n_clusters):
                plt.scatter(
                    pca_result[clustering_result["labels"] == i, 0],
                    pca_result[clustering_result["labels"] == i, 1],
                    s=50,
                    c=colors[i % len(colors)],
                    label=f"Cluster {i}",
                )

            plt.title("Cluster Visualization (PCA)")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.legend()
            plt.grid(True)

            # Save plot to bytes
            img_bytes = BytesIO()
            plt.savefig(img_bytes, format="png")
            plt.close()
            img_bytes.seek(0)
            img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

            # Create explanation text
            text = f"I've segmented your data into {n_clusters} clusters. Here's what I found:\n\n"

            for i, stats in enumerate(cluster_stats):
                text += f"Cluster {i} contains {stats['Size']} records ({stats['Percentage']:.1f}% of the data).\n"

            text += "\nThe visualization shows the clusters projected onto the first two principal components."

            return {
                "analysis_type": "clustering",
                "n_clusters": n_clusters,
                "text": text,
                "charts": [
                    {
                        "type": "scatter",
                        "title": "Cluster Visualization (PCA)",
                        "image": img_base64,
                    }
                ],
                "dataframe": cluster_stats_df.to_dict("records"),
                "cluster_labels": clustering_result["labels"].tolist(),
            }

        except Exception as e:
            logger.error(f"Error in clustering analysis: {str(e)}")
            return {"error": f"Error in clustering analysis: {str(e)}"}

    def analyze_anomalies(self, df: pd.DataFrame, prompt: str) -> Dict[str, Any]:
        """
        Perform anomaly detection on the dataset.

        Args:
            df: DataFrame containing the dataset
            prompt: User's natural language prompt

        Returns:
            Dictionary with anomaly detection results
        """
        try:
            # Filter for numeric columns only for anomaly detection
            numeric_df = df.select_dtypes(include=["number"])

            if numeric_df.empty:
                return {"error": "No numeric columns found for anomaly detection."}

            # Auto detect outliers
            outlier_result = auto_detect_outliers(numeric_df)

            # Add outlier flags to original dataframe
            df_with_outliers = df.copy()
            df_with_outliers["is_outlier"] = outlier_result["outlier_indices"]

            # Calculate outlier stats
            outlier_count = sum(outlier_result["outlier_indices"])
            outlier_percent = outlier_count / len(df) * 100

            # Get outlier records
            outliers_df = df_with_outliers[df_with_outliers["is_outlier"] == 1]

            # Generate box plots for key numeric columns
            charts = []
            for col in numeric_df.columns[:5]:  # Limit to top 5 columns
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=df[col])
                plt.title(f"Box Plot of {col} showing Outliers")
                plt.grid(True)

                # Save plot to bytes
                img_bytes = BytesIO()
                plt.savefig(img_bytes, format="png")
                plt.close()
                img_bytes.seek(0)
                img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

                charts.append(
                    {
                        "type": "boxplot",
                        "title": f"Box Plot of {col}",
                        "image": img_base64,
                    }
                )

            # Create explanation text
            text = f"I've analyzed your data for anomalies and found {outlier_count} outliers ({outlier_percent:.1f}% of the dataset).\n\n"

            if outlier_count > 0:
                text += "The boxplot visualizations show the distribution of key variables with outliers."
                text += "\n\nOutliers are data points that significantly differ from other observations and may represent:"
                text += "\n- Data entry or measurement errors"
                text += "\n- Rare events or special cases that should be investigated"
                text += (
                    "\n- Potential fraud or unusual behavior depending on your domain"
                )
            else:
                text += "No significant outliers were detected in the dataset."

            return {
                "analysis_type": "anomaly_detection",
                "outlier_count": outlier_count,
                "outlier_percentage": outlier_percent,
                "text": text,
                "charts": charts,
                "dataframe": (
                    outliers_df.head(10).to_dict("records")
                    if not outliers_df.empty
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return {"error": f"Error in anomaly detection: {str(e)}"}

    def generate_visualization(self, df: pd.DataFrame, prompt: str) -> Dict[str, Any]:
        """
        Generate visualizations based on the prompt.

        Args:
            df: DataFrame containing the dataset
            prompt: User's natural language prompt

        Returns:
            Dictionary with visualization results
        """
        try:
            # Determine chart type based on prompt
            chart_type = "bar"  # Default

            if any(
                word in prompt.lower() for word in ["line", "trend", "time", "series"]
            ):
                chart_type = "line"
            elif any(
                word in prompt.lower()
                for word in ["scatter", "correlation", "relationship"]
            ):
                chart_type = "scatter"
            elif any(
                word in prompt.lower() for word in ["pie", "percentage", "proportion"]
            ):
                chart_type = "pie"
            elif any(word in prompt.lower() for word in ["histogram", "distribution"]):
                chart_type = "histogram"
            elif any(word in prompt.lower() for word in ["box", "boxplot", "outlier"]):
                chart_type = "boxplot"
            elif any(word in prompt.lower() for word in ["heatmap", "heat", "matrix"]):
                chart_type = "heatmap"

            # Extract column names from prompt
            all_columns = df.columns.tolist()
            mentioned_columns = [
                col for col in all_columns if col.lower() in prompt.lower()
            ]

            # Prioritize columns by type
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            time_cols = detect_time_columns(df)

            # Generate the specified chart
            if chart_type == "bar":
                # Bar chart needs categorical (x) and numeric (y) columns
                x_col = None
                y_col = None

                if mentioned_columns:
                    for col in mentioned_columns:
                        if col in categorical_cols and x_col is None:
                            x_col = col
                        elif col in numeric_cols and y_col is None:
                            y_col = col

                # If no suitable columns mentioned, use defaults
                if x_col is None and categorical_cols:
                    x_col = categorical_cols[0]
                if y_col is None and numeric_cols:
                    y_col = numeric_cols[0]

                if x_col and y_col:
                    plt.figure(figsize=(10, 6))
                    df.groupby(x_col)[y_col].mean().sort_values(ascending=False).plot(
                        kind="bar"
                    )
                    plt.title(f"Average {y_col} by {x_col}")
                    plt.xlabel(x_col)
                    plt.ylabel(f"Average {y_col}")
                    plt.grid(True)
                else:
                    return {"error": "Could not find suitable columns for a bar chart."}

            elif chart_type == "line":
                # Line chart works best with time series data
                x_col = None
                y_col = None

                if mentioned_columns:
                    for col in mentioned_columns:
                        if col in time_cols and x_col is None:
                            x_col = col
                        elif col in numeric_cols and y_col is None:
                            y_col = col

                # If no suitable columns mentioned, use defaults
                if x_col is None:
                    if time_cols:
                        x_col = time_cols[0]
                    elif categorical_cols:
                        x_col = categorical_cols[0]
                    elif len(numeric_cols) > 1:
                        x_col = numeric_cols[0]

                if y_col is None and numeric_cols:
                    y_col = (
                        numeric_cols[0]
                        if x_col != numeric_cols[0] and numeric_cols
                        else numeric_cols[1] if len(numeric_cols) > 1 else None
                    )

                if x_col and y_col:
                    plt.figure(figsize=(10, 6))
                    if x_col in time_cols:
                        # Make sure time column is properly formatted
                        df_sorted = df.sort_values(by=x_col)
                        plt.plot(df_sorted[x_col], df_sorted[y_col])
                    else:
                        df.groupby(x_col)[y_col].mean().plot(kind="line")
                    plt.title(f"{y_col} over {x_col}")
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.grid(True)
                else:
                    return {
                        "error": "Could not find suitable columns for a line chart."
                    }

            elif chart_type == "scatter":
                # Scatter plot needs two numeric columns
                x_col = None
                y_col = None

                if mentioned_columns:
                    numeric_mentioned = [
                        col for col in mentioned_columns if col in numeric_cols
                    ]
                    if len(numeric_mentioned) >= 2:
                        x_col = numeric_mentioned[0]
                        y_col = numeric_mentioned[1]

                # If no suitable columns mentioned, use defaults
                if x_col is None or y_col is None:
                    if len(numeric_cols) >= 2:
                        x_col = numeric_cols[0]
                        y_col = numeric_cols[1]
                    else:
                        return {
                            "error": "Not enough numeric columns for a scatter plot."
                        }

                plt.figure(figsize=(10, 6))
                plt.scatter(df[x_col], df[y_col], alpha=0.6)
                plt.title(f"Relationship between {x_col} and {y_col}")
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.grid(True)

            elif chart_type == "pie":
                # Pie chart works with categorical data
                col = None

                if mentioned_columns:
                    for c in mentioned_columns:
                        if c in categorical_cols:
                            col = c
                            break

                # If no suitable column mentioned, use default
                if col is None and categorical_cols:
                    col = categorical_cols[0]

                if col:
                    plt.figure(figsize=(10, 6))
                    df[col].value_counts().head(10).plot(kind="pie", autopct="%1.1f%%")
                    plt.title(f"Distribution of {col}")
                    plt.ylabel("")  # Hide ylabel
                else:
                    return {
                        "error": "Could not find a suitable categorical column for a pie chart."
                    }

            elif chart_type == "histogram":
                # Histogram works with numeric data
                col = None

                if mentioned_columns:
                    for c in mentioned_columns:
                        if c in numeric_cols:
                            col = c
                            break

                # If no suitable column mentioned, use default
                if col is None and numeric_cols:
                    col = numeric_cols[0]

                if col:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(df[col], kde=True)
                    plt.title(f"Distribution of {col}")
                    plt.xlabel(col)
                    plt.grid(True)
                else:
                    return {
                        "error": "Could not find a suitable numeric column for a histogram."
                    }

            elif chart_type == "boxplot":
                # Boxplot works with numeric data
                col = None

                if mentioned_columns:
                    for c in mentioned_columns:
                        if c in numeric_cols:
                            col = c
                            break

                # If no suitable column mentioned, use default
                if col is None and numeric_cols:
                    col = numeric_cols[0]

                if col:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x=df[col])
                    plt.title(f"Box Plot of {col}")
                    plt.grid(True)
                else:
                    return {
                        "error": "Could not find a suitable numeric column for a box plot."
                    }

            elif chart_type == "heatmap":
                # Heatmap works with correlation matrix
                numeric_df = df.select_dtypes(include=["number"])

                if numeric_df.shape[1] < 2:
                    return {"error": "Not enough numeric columns for a heatmap."}

                plt.figure(figsize=(12, 10))
                corr_matrix = numeric_df.corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                cmap = sns.diverging_palette(230, 20, as_cmap=True)

                sns.heatmap(
                    corr_matrix,
                    mask=mask,
                    cmap=cmap,
                    vmax=1,
                    vmin=-1,
                    center=0,
                    square=True,
                    linewidths=0.5,
                    annot=True,
                    fmt=".2f",
                )
                plt.title("Correlation Heatmap")

            # Save plot to bytes
            img_bytes = BytesIO()
            plt.savefig(img_bytes, format="png")
            plt.close()
            img_bytes.seek(0)
            img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

            # Create explanation text
            text = f"I've created a {chart_type} chart based on your request. "

            if chart_type == "bar":
                text += f"This shows the average {y_col} for each {x_col} category."
            elif chart_type == "line":
                text += f"This shows how {y_col} changes over {x_col}."
            elif chart_type == "scatter":
                text += f"This shows the relationship between {x_col} and {y_col}."
            elif chart_type == "pie":
                text += f"This shows the distribution of {col} categories."
            elif chart_type == "histogram":
                text += f"This shows the distribution of {col} values."
            elif chart_type == "boxplot":
                text += f"This shows the distribution and potential outliers in {col}."
            elif chart_type == "heatmap":
                text += "This shows the correlation between all numeric variables in your dataset."

            return {
                "analysis_type": "visualization",
                "chart_type": chart_type,
                "text": text,
                "charts": [
                    {
                        "type": chart_type,
                        "title": f"{chart_type.capitalize()} Chart",
                        "image": img_base64,
                    }
                ],
            }

        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return {"error": f"Error generating visualization: {str(e)}"}

    def general_analysis(self, df: pd.DataFrame, prompt: str) -> Dict[str, Any]:
        """
        Perform general analysis on the dataset based on the prompt.

        Args:
            df: DataFrame containing the dataset
            prompt: User's natural language prompt

        Returns:
            Dictionary with general analysis results
        """
        try:
            # Detect column types
            column_types = {}
            column_types["numeric"] = df.select_dtypes(
                include=["number"]
            ).columns.tolist()
            column_types["categorical"] = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            column_types["time"] = detect_time_columns(df)

            # Generate overall statistics
            stats = {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "missing_values": df.isnull().sum().sum(),
                "column_types": {
                    "numeric": len(column_types["numeric"]),
                    "categorical": len(column_types["categorical"]),
                    "time": len(column_types["time"]),
                },
            }

            # Generate dataset summary
            summary = {}

            # Summary for numeric columns
            for col in column_types["numeric"][:5]:  # Limit to 5 columns
                summary[col] = {
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "std": df[col].std(),
                }

            # Summary for categorical columns
            for col in column_types["categorical"][:5]:  # Limit to 5 columns
                value_counts = df[col].value_counts().head(5).to_dict()
                summary[col] = {
                    "unique_values": df[col].nunique(),
                    "top_values": value_counts,
                }

            # Generate appropriate visualizations
            charts = []

            # If we have numeric columns, show distribution of first numeric column
            if column_types["numeric"]:
                col = column_types["numeric"][0]
                plt.figure(figsize=(10, 6))
                sns.histplot(df[col], kde=True)
                plt.title(f"Distribution of {col}")
                plt.xlabel(col)
                plt.grid(True)

                # Save plot to bytes
                img_bytes = BytesIO()
                plt.savefig(img_bytes, format="png")
                plt.close()
                img_bytes.seek(0)
                img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

                charts.append(
                    {
                        "type": "histogram",
                        "title": f"Distribution of {col}",
                        "image": img_base64,
                    }
                )

            # If we have categorical columns, show top categories
            if column_types["categorical"]:
                col = column_types["categorical"][0]
                plt.figure(figsize=(10, 6))
                df[col].value_counts().head(10).plot(kind="bar")
                plt.title(f"Top Categories of {col}")
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.grid(True)

                # Save plot to bytes
                img_bytes = BytesIO()
                plt.savefig(img_bytes, format="png")
                plt.close()
                img_bytes.seek(0)
                img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

                charts.append(
                    {
                        "type": "bar",
                        "title": f"Top Categories of {col}",
                        "image": img_base64,
                    }
                )

            # If we have time and numeric columns, show time trend
            if column_types["time"] and column_types["numeric"]:
                time_col = column_types["time"][0]
                numeric_col = column_types["numeric"][0]

                plt.figure(figsize=(10, 6))
                df_sorted = df.sort_values(by=time_col)
                plt.plot(df_sorted[time_col], df_sorted[numeric_col])
                plt.title(f"{numeric_col} over Time")
                plt.xlabel(time_col)
                plt.ylabel(numeric_col)
                plt.grid(True)

                # Save plot to bytes
                img_bytes = BytesIO()
                plt.savefig(img_bytes, format="png")
                plt.close()
                img_bytes.seek(0)
                img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

                charts.append(
                    {
                        "type": "line",
                        "title": f"{numeric_col} over Time",
                        "image": img_base64,
                    }
                )

            # Generate text explanation
            text = self._generate_data_explanation(df, column_types, prompt)

            return {
                "analysis_type": "general",
                "text": text,
                "statistics": stats,
                "summary": summary,
                "charts": charts,
                "dataframe": df.head(10).to_dict("records"),
            }

        except Exception as e:
            logger.error(f"Error in general analysis: {str(e)}")
            return {"error": f"Error in general analysis: {str(e)}"}
