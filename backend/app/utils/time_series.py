import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory for models
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "cache", "models"
)
os.makedirs(CACHE_DIR, exist_ok=True)


def forecast_with_prophet(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    periods: int = 30,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
) -> Dict[str, Any]:
    """
    Perform time series forecasting using Facebook Prophet

    Args:
        df: DataFrame with time series data
        date_col: Name of the date column
        value_col: Name of the value column to forecast
        periods: Number of periods to forecast
        yearly_seasonality: Whether to include yearly seasonality
        weekly_seasonality: Whether to include weekly seasonality
        daily_seasonality: Whether to include daily seasonality

    Returns:
        Dictionary with forecast results and plot
    """
    try:
        # Prepare data for Prophet (needs 'ds' and 'y' columns)
        prophet_df = df[[date_col, value_col]].copy()
        prophet_df.columns = ["ds", "y"]

        # Check for missing or infinite values
        if prophet_df["y"].isnull().any() or np.isinf(prophet_df["y"]).any():
            prophet_df["y"] = (
                prophet_df["y"]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(prophet_df["y"].median())
            )
            logger.warning("Replaced missing or infinite values in data")

        # Create and fit model
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
        )
        model.fit(prophet_df)

        # Create future dataframe and predict
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        # Save model for caching
        model_path = os.path.join(
            CACHE_DIR,
            f"prophet_{date_col}_{value_col}_{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib",
        )
        joblib.dump(model, model_path)

        # Generate plot
        fig = model.plot(forecast)
        plt.title(f"Forecast of {value_col} for the next {periods} periods")

        # Convert plot to base64 for easy transmission
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close(fig)

        # Prepare result
        result = {
            "forecast": forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            .tail(periods)
            .to_dict("records"),
            "plot": plot_data,
            "model_info": {
                "type": "prophet",
                "path": model_path,
                "date_col": date_col,
                "value_col": value_col,
                "periods": periods,
            },
            "components": {
                "trend": forecast["trend"].to_list(),
                "has_yearly_seasonality": yearly_seasonality,
                "has_weekly_seasonality": weekly_seasonality,
                "has_daily_seasonality": daily_seasonality,
            },
        }

        return result

    except Exception as e:
        logger.error(f"Error in Prophet forecasting: {str(e)}")
        return {"error": str(e), "fallback_to": "arima"}


def forecast_with_arima(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    periods: int = 30,
    order: Tuple[int, int, int] = (5, 1, 0),
) -> Dict[str, Any]:
    """
    Perform time series forecasting using ARIMA model

    Args:
        df: DataFrame with time series data
        date_col: Name of the date column
        value_col: Name of the value column to forecast
        periods: Number of periods to forecast
        order: ARIMA order (p, d, q)

    Returns:
        Dictionary with forecast results and plot
    """
    try:
        # Prepare data
        ts_data = df.set_index(date_col)[value_col]

        # Fit ARIMA model
        model = ARIMA(ts_data, order=order)
        model_fit = model.fit()

        # Forecast
        forecast = model_fit.forecast(steps=periods)
        forecast_index = pd.date_range(
            start=ts_data.index[-1] + pd.Timedelta(days=1),
            periods=periods,
            freq=pd.infer_freq(ts_data.index),
        )
        forecast_series = pd.Series(forecast, index=forecast_index)

        # Create confidence intervals
        conf_int = model_fit.get_forecast(steps=periods).conf_int()

        # Generate plot
        plt.figure(figsize=(10, 6))
        plt.plot(ts_data.index, ts_data, label="Historical")
        plt.plot(forecast_series.index, forecast_series, color="red", label="Forecast")
        plt.fill_between(
            conf_int.index,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            color="pink",
            alpha=0.3,
        )
        plt.title(f"ARIMA Forecast ({order}) of {value_col}")
        plt.legend()

        # Convert plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close()

        # Save model
        model_path = os.path.join(
            CACHE_DIR,
            f"arima_{date_col}_{value_col}_{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib",
        )
        joblib.dump(model_fit, model_path)

        # Prepare results
        result = {
            "forecast": [
                {"date": idx.strftime("%Y-%m-%d"), "value": val}
                for idx, val in forecast_series.items()
            ],
            "plot": plot_data,
            "model_info": {
                "type": "arima",
                "path": model_path,
                "date_col": date_col,
                "value_col": value_col,
                "periods": periods,
                "order": order,
            },
            "metrics": {"aic": model_fit.aic, "bic": model_fit.bic},
        }

        return result

    except Exception as e:
        logger.error(f"Error in ARIMA forecasting: {str(e)}")
        return {"error": str(e), "fallback_to": "simple_trend"}


def simple_trend_analysis(
    df: pd.DataFrame, date_col: str, value_col: str
) -> Dict[str, Any]:
    """
    Perform simple trend analysis as fallback when advanced models fail
    """
    try:
        # Ensure date column is datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Sort and prepare data
        df = df.sort_values(by=date_col)
        ts_data = df.set_index(date_col)[value_col]

        # Calculate rolling metrics
        window_size = max(3, len(ts_data) // 10)  # Dynamic window size
        rolling_mean = ts_data.rolling(window=window_size).mean()

        # Linear regression for trend
        x = np.arange(len(ts_data))
        y = ts_data.values
        valid_mask = ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        if len(x_valid) > 1:
            slope, intercept = np.polyfit(x_valid, y_valid, 1)
            trend_line = intercept + slope * x
            trend_direction = (
                "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            )

            # Calculate percent change
            first_valid = ts_data.iloc[valid_mask.argmax()]
            last_valid = ts_data.iloc[valid_mask[::-1].argmax()]
            percent_change = (
                ((last_valid - first_valid) / first_valid * 100)
                if first_valid != 0
                else 0
            )
        else:
            trend_line = np.full_like(x, np.nan)
            trend_direction = "unknown"
            percent_change = 0

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(ts_data.index, ts_data, label="Actual")
        plt.plot(
            ts_data.index,
            rolling_mean,
            label=f"{window_size}-period Rolling Average",
            color="orange",
        )

        if len(x_valid) > 1:
            plt.plot(
                ts_data.index,
                trend_line,
                label=f"Trend Line ({trend_direction})",
                color="red",
                linestyle="--",
            )

        plt.title(f"Trend Analysis for {value_col}")
        plt.legend()

        # Convert plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close()

        # Return results
        return {
            "trend_direction": trend_direction,
            "percent_change": round(percent_change, 2),
            "rolling_average": rolling_mean.dropna().to_dict(),
            "plot": plot_data,
            "analysis_type": "simple_trend",
        }

    except Exception as e:
        logger.error(f"Error in simple trend analysis: {str(e)}")
        return {"error": str(e), "fallback_to": None}


def auto_forecast(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    periods: int = 30,
) -> Dict[str, Any]:
    """
    Automatically select and apply the appropriate forecasting model

    Args:
        df: DataFrame with time series data
        date_col: Name of the date column (auto-detected if None)
        value_col: Name of the value column to forecast (auto-detected if None)
        periods: Number of periods to forecast

    Returns:
        Dictionary with forecast results
    """
    from ..utils.data_preprocessing import detect_time_columns, detect_column_types

    # Auto-detect date column if not provided
    if date_col is None:
        time_cols = detect_time_columns(df)
        if time_cols:
            date_col = time_cols[0]
        else:
            return {"error": "No date column detected. Please specify the date column."}

    # Auto-detect value column if not provided
    if value_col is None:
        col_types = detect_column_types(df)
        numeric_cols = [col for col in col_types["numeric"] if col != date_col]

        if numeric_cols:
            # Select column with highest variance as default
            variances = {
                col: df[col].var() for col in numeric_cols if not df[col].isnull().all()
            }
            if variances:
                value_col = max(variances, key=variances.get)
            else:
                value_col = numeric_cols[0]
        else:
            return {"error": "No numeric columns detected for forecasting."}

    # Try Prophet first
    prophet_result = forecast_with_prophet(df, date_col, value_col, periods)

    # If Prophet fails, try ARIMA
    if "error" in prophet_result:
        logger.info(
            f"Prophet forecasting failed, trying ARIMA: {prophet_result['error']}"
        )
        arima_result = forecast_with_arima(df, date_col, value_col, periods)

        # If ARIMA fails, fall back to simple trend analysis
        if "error" in arima_result:
            logger.info(
                f"ARIMA forecasting failed, falling back to simple trend: {arima_result['error']}"
            )
            result = simple_trend_analysis(df, date_col, value_col)
            result["attempts"] = ["prophet", "arima", "simple_trend"]
            return result

        arima_result["attempts"] = ["prophet", "arima"]
        return arima_result

    prophet_result["attempts"] = ["prophet"]
    return prophet_result
