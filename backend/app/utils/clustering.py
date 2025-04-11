import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.cluster import KMeans, DBSCAN
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


def segment_with_kmeans(
    df: pd.DataFrame,
    features: List[str],
    n_clusters: Optional[int] = None,
    max_clusters: int = 10,
) -> Dict[str, Any]:
    """
    Segment data using K-Means clustering

    Args:
        df: Input dataframe
        features: List of feature columns for segmentation
        n_clusters: Number of clusters (auto-determined if None)
        max_clusters: Maximum number of clusters to try when auto-determining

    Returns:
        Dictionary with segmentation results and visualizations
    """
    try:
        # Extract features
        X = df[features].copy()

        # Handle missing values
        X = X.fillna(X.mean())

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Auto-determine number of clusters if not provided
        if n_clusters is None:
            # Use elbow method to find optimal k
            inertia = []
            k_range = range(
                2, min(max_clusters + 1, len(df) // 5, 15)
            )  # Reasonable range

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)

            # Find elbow point (simple method)
            inertia_diff = np.diff(inertia)
            inertia_diff2 = np.diff(inertia_diff)
            elbow_index = np.argmax(inertia_diff2) + 1 if len(inertia_diff2) > 0 else 0
            n_clusters = k_range[min(elbow_index + 1, len(k_range) - 1)]

            # Plot elbow curve
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, inertia, "bo-")
            plt.plot(
                k_range[elbow_index + 1], inertia[elbow_index + 1], "ro", markersize=12
            )
            plt.xlabel("Number of Clusters")
            plt.ylabel("Inertia")
            plt.title("Elbow Method for Optimal k")
            plt.grid(True)

            # Convert elbow plot to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            elbow_plot = base64.b64encode(buffer.getvalue()).decode("utf-8")
            plt.close()
        else:
            elbow_plot = None

        # Perform clustering with optimal/provided k
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(X_scaled)

        # Save model
        model_path = os.path.join(
            CACHE_DIR,
            f"kmeans_{n_clusters}_{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib",
        )
        joblib.dump(
            {"kmeans": kmeans, "scaler": scaler, "features": features}, model_path
        )

        # Generate cluster profiles
        cluster_profiles = df.groupby("cluster")[features].mean().reset_index()

        # Get cluster sizes
        cluster_sizes = df["cluster"].value_counts().sort_index().to_dict()

        # If more than 2 features, use PCA for visualization
        if len(features) > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # Plot clusters in PCA space
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                X_pca[:, 0], X_pca[:, 1], c=df["cluster"], cmap="viridis", alpha=0.7
            )
            plt.colorbar(scatter, label="Cluster")
            plt.title(f"Data Segments ({n_clusters} clusters)")
            plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")

            # Mark cluster centers in PCA space
            centers_pca = pca.transform(kmeans.cluster_centers_)
            plt.scatter(
                centers_pca[:, 0],
                centers_pca[:, 1],
                s=200,
                marker="X",
                c="red",
                label="Cluster Centers",
            )
            plt.legend()

        else:  # Direct 2D plot for 2 features
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                X[features[0]],
                X[features[1]],
                c=df["cluster"],
                cmap="viridis",
                alpha=0.7,
            )
            plt.colorbar(scatter, label="Cluster")
            plt.title(f"Data Segments ({n_clusters} clusters)")
            plt.xlabel(features[0])
            plt.ylabel(features[1])

            # Mark cluster centers
            centers = kmeans.cluster_centers_
            plt.scatter(
                scaler.inverse_transform(centers)[:, 0],
                scaler.inverse_transform(centers)[:, 1],
                s=200,
                marker="X",
                c="red",
                label="Cluster Centers",
            )
            plt.legend()

        # Convert cluster plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        cluster_plot = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close()

        # Generate radar chart for feature importance per cluster
        if len(features) >= 3:
            # Normalize data for radar chart
            cluster_profiles_norm = pd.DataFrame()
            for feature in features:
                min_val = df[feature].min()
                max_val = df[feature].max()
                if max_val > min_val:
                    cluster_profiles_norm[feature] = (
                        cluster_profiles[feature] - min_val
                    ) / (max_val - min_val)
                else:
                    cluster_profiles_norm[feature] = 0.5  # Default when no variation

            # Create radar chart
            fig = plt.figure(figsize=(12, 10))

            # Set number of angles for radar chart (one per feature)
            angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
            angles += angles[:1]  # Close the loop

            # Create subplot with polar projection
            ax = fig.add_subplot(111, polar=True)

            # Add feature labels
            plt.xticks(angles[:-1], features, size=12)

            # Plot each cluster
            for cluster_id in range(n_clusters):
                values = cluster_profiles_norm.iloc[cluster_id].values.tolist()
                values += values[:1]  # Close the loop
                ax.plot(
                    angles,
                    values,
                    linewidth=2,
                    linestyle="solid",
                    label=f"Cluster {cluster_id}",
                )
                ax.fill(angles, values, alpha=0.1)

            plt.title("Cluster Profiles", size=16)
            plt.legend(loc="upper right")

            # Convert radar chart to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            radar_plot = base64.b64encode(buffer.getvalue()).decode("utf-8")
            plt.close()
        else:
            radar_plot = None

        # Generate segment descriptions
        segment_descriptions = []
        for cluster_id in range(n_clusters):
            profile = cluster_profiles.loc[
                cluster_profiles["cluster"] == cluster_id, features
            ].iloc[0]

            # Get top 3 highest and lowest features for this cluster
            highest = profile.nlargest(min(3, len(features)))
            lowest = profile.nsmallest(min(3, len(features)))

            # Create description
            description = f"Segment {cluster_id}: "
            description += f"High in {', '.join(highest.index.tolist())}. "
            description += f"Low in {', '.join(lowest.index.tolist())}. "
            description += f"({cluster_sizes.get(cluster_id, 0)} items, {cluster_sizes.get(cluster_id, 0)/len(df):.1%} of total)"

            segment_descriptions.append(description)

        # Prepare results
        result = {
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes,
            "cluster_profiles": cluster_profiles.to_dict("records"),
            "segment_descriptions": segment_descriptions,
            "model_info": {"type": "kmeans", "path": model_path, "features": features},
            "plots": {
                "clusters": cluster_plot,
                "radar": radar_plot,
                "elbow": elbow_plot,
            },
        }

        return result

    except Exception as e:
        logger.error(f"Error in K-Means segmentation: {str(e)}")
        return {"error": str(e), "fallback_to": "simple_segmentation"}


def segment_with_dbscan(
    df: pd.DataFrame, features: List[str], eps: float = 0.5, min_samples: int = 5
) -> Dict[str, Any]:
    """
    Segment data using DBSCAN clustering (good for detecting outliers)

    Args:
        df: Input dataframe
        features: List of feature columns for segmentation
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter

    Returns:
        Dictionary with segmentation results
    """
    try:
        # Extract features
        X = df[features].copy()

        # Handle missing values
        X = X.fillna(X.mean())

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df["cluster"] = dbscan.fit_predict(X_scaled)

        # Count cluster sizes
        cluster_sizes = df["cluster"].value_counts().sort_index().to_dict()

        # Find outliers (cluster -1)
        outliers = df[df["cluster"] == -1]

        # Generate cluster profiles
        cluster_profiles = df.groupby("cluster")[features].mean().reset_index()

        # If more than 2 features, use PCA for visualization
        if len(features) > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # Plot clusters in PCA space
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                X_pca[:, 0], X_pca[:, 1], c=df["cluster"], cmap="viridis", alpha=0.7
            )
            plt.colorbar(scatter, label="Cluster")
            plt.title("DBSCAN Clustering Results")
            plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")

        else:  # Direct 2D plot for 2 features
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                X[features[0]],
                X[features[1]],
                c=df["cluster"],
                cmap="viridis",
                alpha=0.7,
            )
            plt.colorbar(scatter, label="Cluster")
            plt.title("DBSCAN Clustering Results")
            plt.xlabel(features[0])
            plt.ylabel(features[1])

        # Convert cluster plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        cluster_plot = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close()

        # Save model
        model_path = os.path.join(
            CACHE_DIR,
            f"dbscan_{eps}_{min_samples}_{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib",
        )
        joblib.dump(
            {"dbscan": dbscan, "scaler": scaler, "features": features}, model_path
        )

        # Prepare results
        result = {
            "n_clusters": len(set(df["cluster"]))
            - (1 if -1 in df["cluster"].values else 0),
            "cluster_sizes": cluster_sizes,
            "outliers_count": len(outliers),
            "cluster_profiles": cluster_profiles.to_dict("records"),
            "model_info": {
                "type": "dbscan",
                "path": model_path,
                "features": features,
                "eps": eps,
                "min_samples": min_samples,
            },
            "plots": {"clusters": cluster_plot},
        }

        return result

    except Exception as e:
        logger.error(f"Error in DBSCAN segmentation: {str(e)}")
        return {"error": str(e), "fallback_to": "kmeans"}


def simple_segmentation(
    df: pd.DataFrame, feature: str, n_bins: int = 5
) -> Dict[str, Any]:
    """
    Perform simple segmentation by binning a single feature

    Args:
        df: Input dataframe
        feature: Feature to segment by
        n_bins: Number of bins

    Returns:
        Dictionary with segmentation results
    """
    try:
        # Create a copy of the dataframe
        df_result = df.copy()

        # Create bins
        df_result["segment"] = pd.qcut(
            df[feature], q=n_bins, labels=False, duplicates="drop"
        )

        # Get bin edges for labels
        bins = pd.qcut(df[feature], q=n_bins, duplicates="drop")
        bin_labels = [
            f"{interval.left:.2f} - {interval.right:.2f}"
            for interval in bins.cat.categories
        ]

        # Count segments
        segment_counts = df_result["segment"].value_counts().sort_index().to_dict()

        # Generate segment profiles
        segment_profiles = (
            df_result.groupby("segment")[feature]
            .agg(["min", "max", "mean", "count"])
            .reset_index()
        )
        segment_profiles["label"] = bin_labels

        # Plot distribution
        plt.figure(figsize=(10, 6))
        counts = [segment_counts.get(i, 0) for i in range(len(bin_labels))]
        plt.bar(bin_labels, counts)
        plt.title(f"Segments by {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)

        # Convert plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        segment_plot = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close()

        # Prepare results
        result = {
            "n_segments": len(bin_labels),
            "segment_counts": segment_counts,
            "segment_profiles": segment_profiles.to_dict("records"),
            "segment_labels": bin_labels,
            "feature": feature,
            "analysis_type": "simple_binning",
            "plots": {"segments": segment_plot},
        }

        return result

    except Exception as e:
        logger.error(f"Error in simple segmentation: {str(e)}")
        return {"error": str(e)}


def auto_segment(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_clusters: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Automatically select and apply the appropriate segmentation technique

    Args:
        df: Input dataframe
        features: List of features for segmentation (auto-selected if None)
        n_clusters: Number of clusters (auto-determined if None)

    Returns:
        Dictionary with segmentation results
    """
    from ..utils.data_preprocessing import detect_column_types

    # Auto-select features if not provided
    if features is None or len(features) == 0:
        col_types = detect_column_types(df)
        # Use numeric features by default
        features = col_types.get("numeric", [])[:5]  # Limit to 5 features

        if len(features) == 0:
            return {"error": "No suitable numeric features found for segmentation."}

    # If only one feature, use simple binning
    if len(features) == 1:
        logger.info(f"Only one feature selected, using simple segmentation")
        return simple_segmentation(df, features[0])

    # Try K-Means clustering first
    kmeans_result = segment_with_kmeans(df, features, n_clusters)

    # If K-Means fails, try DBSCAN
    if "error" in kmeans_result:
        logger.info(
            f"K-Means clustering failed, trying DBSCAN: {kmeans_result['error']}"
        )
        dbscan_result = segment_with_dbscan(df, features)

        # If DBSCAN fails, fall back to simple segmentation on first feature
        if "error" in dbscan_result:
            logger.info(
                f"DBSCAN clustering failed, falling back to simple segmentation: {dbscan_result['error']}"
            )
            result = simple_segmentation(df, features[0])
            result["attempts"] = ["kmeans", "dbscan", "simple_segmentation"]
            return result

        dbscan_result["attempts"] = ["kmeans", "dbscan"]
        return dbscan_result

    kmeans_result["attempts"] = ["kmeans"]
    return kmeans_result
