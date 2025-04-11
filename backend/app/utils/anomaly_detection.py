import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import joblib
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory for models
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "cache", "models"
)
os.makedirs(CACHE_DIR, exist_ok=True)


def detect_outliers_isolation_forest(
    df: pd.DataFrame, features: List[str], contamination: float = 0.05
) -> Dict[str, Any]:
    """
    Detect outliers using Isolation Forest algorithm

    Args:
        df: Input dataframe
        features: Features to use for outlier detection
        contamination: Expected proportion of outliers

    Returns:
        Dictionary with outlier detection results
    """
    try:
        # Extract features
        X = df[features].copy()

        # Handle missing values
        X = X.fillna(X.mean())

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit Isolation Forest
        isolation_forest = IsolationForest(
            n_estimators=100, contamination=contamination, random_state=42
        )

        # Predict outliers (-1 for outliers, 1 for inliers)
        df["outlier"] = isolation_forest.fit_predict(X_scaled)
        df["outlier_score"] = -isolation_forest.decision_function(X_scaled)

        # Convert to boolean for easier filtering
        df["is_outlier"] = df["outlier"] == -1

        # Identify outliers
        outliers = df[df["is_outlier"]]

        # Save model
        model_path = os.path.join(
            CACHE_DIR,
            f"isolation_forest_{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib",
        )
        joblib.dump(
            {"model": isolation_forest, "scaler": scaler, "features": features},
            model_path,
        )

        # Visualize outliers
        # If more than 2 features, use PCA for visualization
        if len(features) > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            plt.figure(figsize=(10, 8))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["outlier_score"], cmap="YlOrRd")
            plt.colorbar(label="Outlier Score")
            plt.title("Outlier Detection with Isolation Forest")
            plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")

            # Mark top outliers
            top_outliers = df.nlargest(10, "outlier_score")
            top_outliers_pca = pca.transform(scaler.transform(top_outliers[features]))
            plt.scatter(
                top_outliers_pca[:, 0],
                top_outliers_pca[:, 1],
                s=100,
                edgecolors="k",
                facecolors="none",
                label="Top Outliers",
            )
            plt.legend()

        else:  # Direct 2D plot for 2 features
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                X[features[0]], X[features[1]], c=df["outlier_score"], cmap="YlOrRd"
            )
            plt.colorbar(scatter, label="Outlier Score")
            plt.title("Outlier Detection with Isolation Forest")
            plt.xlabel(features[0])
            plt.ylabel(features[1])

            # Mark top outliers
            top_outliers = df.nlargest(10, "outlier_score")
            plt.scatter(
                top_outliers[features[0]],
                top_outliers[features[1]],
                s=100,
                edgecolors="k",
                facecolors="none",
                label="Top Outliers",
            )
            plt.legend()

        # Convert plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        outlier_plot = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close()

        # Prepare outlier statistics
        feature_stats_outliers = outliers[features].describe().to_dict()
        feature_stats_normal = df[~df["is_outlier"]][features].describe().to_dict()

        # Create outlier histogram for each feature
        hist_plots = {}
        for feature in features:
            plt.figure(figsize=(10, 6))
            plt.hist(df[~df["is_outlier"]][feature], bins=30, alpha=0.5, label="Normal")
            plt.hist(outliers[feature], bins=30, alpha=0.5, label="Outliers")
            plt.title(f"Distribution of {feature} - Normal vs Outliers")
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.legend()

            # Convert plot to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            hist_plots[feature] = base64.b64encode(buffer.getvalue()).decode("utf-8")
            plt.close()

        # Prepare result
        result = {
            "outlier_count": len(outliers),
            "outlier_percentage": len(outliers) / len(df) * 100,
            "top_outliers": outliers.sort_values("outlier_score", ascending=False)
            .head(10)
            .to_dict("records"),
            "outlier_stats": feature_stats_outliers,
            "normal_stats": feature_stats_normal,
            "model_info": {
                "type": "isolation_forest",
                "path": model_path,
                "features": features,
                "contamination": contamination,
            },
            "plots": {"outliers": outlier_plot, "histograms": hist_plots},
        }

        return result

    except Exception as e:
        logger.error(f"Error in Isolation Forest outlier detection: {str(e)}")
        return {"error": str(e), "fallback_to": "lof"}


def detect_outliers_lof(
    df: pd.DataFrame,
    features: List[str],
    contamination: float = 0.05,
    n_neighbors: int = 20,
) -> Dict[str, Any]:
    """
    Detect outliers using Local Outlier Factor algorithm

    Args:
        df: Input dataframe
        features: Features to use for outlier detection
        contamination: Expected proportion of outliers
        n_neighbors: Number of neighbors to consider

    Returns:
        Dictionary with outlier detection results
    """
    try:
        # Extract features
        X = df[features].copy()

        # Handle missing values
        X = X.fillna(X.mean())

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit LOF
        lof = LocalOutlierFactor(
            n_neighbors=min(n_neighbors, len(df) - 1), contamination=contamination
        )

        # Predict outliers (-1 for outliers, 1 for inliers)
        df["outlier"] = lof.fit_predict(X_scaled)
        df["outlier_score"] = -lof.negative_outlier_factor_

        # Convert to boolean for easier filtering
        df["is_outlier"] = df["outlier"] == -1

        # Identify outliers
        outliers = df[df["is_outlier"]]

        # Visualize outliers
        # If more than 2 features, use PCA for visualization
        if len(features) > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            plt.figure(figsize=(10, 8))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["outlier_score"], cmap="YlOrRd")
            plt.colorbar(label="Outlier Score")
            plt.title("Outlier Detection with Local Outlier Factor")
            plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")

            # Mark top outliers
            top_outliers = df.nlargest(10, "outlier_score")
            top_outliers_pca = pca.transform(scaler.transform(top_outliers[features]))
            plt.scatter(
                top_outliers_pca[:, 0],
                top_outliers_pca[:, 1],
                s=100,
                edgecolors="k",
                facecolors="none",
                label="Top Outliers",
            )
            plt.legend()

        else:  # Direct 2D plot for 2 features
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                X[features[0]], X[features[1]], c=df["outlier_score"], cmap="YlOrRd"
            )
            plt.colorbar(scatter, label="Outlier Score")
            plt.title("Outlier Detection with Local Outlier Factor")
            plt.xlabel(features[0])
            plt.ylabel(features[1])

            # Mark top outliers
            top_outliers = df.nlargest(10, "outlier_score")
            plt.scatter(
                top_outliers[features[0]],
                top_outliers[features[1]],
                s=100,
                edgecolors="k",
                facecolors="none",
                label="Top Outliers",
            )
            plt.legend()

        # Convert plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        outlier_plot = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close()

        # Prepare result
        result = {
            "outlier_count": len(outliers),
            "outlier_percentage": len(outliers) / len(df) * 100,
            "top_outliers": outliers.sort_values("outlier_score", ascending=False)
            .head(10)
            .to_dict("records"),
            "model_info": {
                "type": "lof",
                "features": features,
                "contamination": contamination,
                "n_neighbors": n_neighbors,
            },
            "plots": {"outliers": outlier_plot},
        }

        return result

    except Exception as e:
        logger.error(f"Error in LOF outlier detection: {str(e)}")
        return {"error": str(e), "fallback_to": "statistical"}


def detect_outliers_statistical(
    df: pd.DataFrame, features: List[str], threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Detect outliers using statistical methods (Z-score)

    Args:
        df: Input dataframe
        features: Features to use for outlier detection
        threshold: Z-score threshold (typically 2.5-3.5)

    Returns:
        Dictionary with outlier detection results
    """
    try:
        # Create a copy of dataframe
        df_result = df.copy()

        # Calculate Z-scores for each feature
        z_scores = pd.DataFrame()
        for feature in features:
            z_scores[f"{feature}_zscore"] = np.abs(
                (df[feature] - df[feature].mean()) / df[feature].std()
            )

        # Mark outliers where any feature exceeds threshold
        df_result["outlier_score"] = z_scores.max(axis=1)
        df_result["is_outlier"] = df_result["outlier_score"] > threshold

        # Identify outliers
        outliers = df_result[df_result["is_outlier"]]

        # Create boxplots for each feature
        boxplot_data = {}
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")

            # Convert plot to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            boxplot_data[feature] = base64.b64encode(buffer.getvalue()).decode("utf-8")
            plt.close()

        # Prepare result
        result = {
            "outlier_count": len(outliers),
            "outlier_percentage": len(outliers) / len(df) * 100,
            "top_outliers": outliers.sort_values("outlier_score", ascending=False)
            .head(10)
            .to_dict("records"),
            "outlier_stats": {
                "method": "z_score",
                "threshold": threshold,
                "features": features,
            },
            "plots": {"boxplots": boxplot_data},
        }

        return result

    except Exception as e:
        logger.error(f"Error in statistical outlier detection: {str(e)}")
        return {"error": str(e)}


def auto_detect_outliers(
    df: pd.DataFrame, features: Optional[List[str]] = None, contamination: float = 0.05
) -> Dict[str, Any]:
    """
    Automatically select and apply the appropriate outlier detection technique

    Args:
        df: Input dataframe
        features: Features to use for outlier detection (auto-selected if None)
        contamination: Expected proportion of outliers

    Returns:
        Dictionary with outlier detection results
    """
    from ..utils.data_preprocessing import detect_column_types

    # Auto-select features if not provided
    if features is None or len(features) == 0:
        col_types = detect_column_types(df)
        features = col_types.get("numeric", [])[:8]  # Limit to 8 numeric features

        if len(features) == 0:
            return {
                "error": "No suitable numeric features found for outlier detection."
            }

    # Try Isolation Forest first (works well with higher dimensional data)
    if_result = detect_outliers_isolation_forest(df, features, contamination)

    # If Isolation Forest fails, try LOF
    if "error" in if_result:
        logger.info(f"Isolation Forest failed, trying LOF: {if_result['error']}")
        lof_result = detect_outliers_lof(df, features, contamination)

        # If LOF fails, fall back to statistical method
        if "error" in lof_result:
            logger.info(
                f"LOF failed, falling back to statistical method: {lof_result['error']}"
            )
            result = detect_outliers_statistical(df, features)
            result["attempts"] = ["isolation_forest", "lof", "statistical"]
            return result

        lof_result["attempts"] = ["isolation_forest", "lof"]
        return lof_result

    if_result["attempts"] = ["isolation_forest"]
    return if_result
