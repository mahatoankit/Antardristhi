"""
Titanic Dataset Analysis Script

This script demonstrates using the AnalysisEngine with the Titanic dataset
to perform various types of analysis.
"""

import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add the app directory to the path
backend_dir = Path(__file__).resolve().parent
sys.path.append(str(backend_dir))

# Import the AnalysisEngine
from app.services.analysis_service import AnalysisEngine


def analyze_titanic():
    """Analyze the Titanic dataset using the AnalysisEngine"""

    print("Loading Titanic dataset...")
    titanic_path = os.path.join(backend_dir, "titanic.csv")
    df = pd.read_csv(titanic_path)

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {', '.join(df.columns)}")

    # Clean up the dataset - drop zero columns and fix column names
    drop_cols = [col for col in df.columns if col == "zero"]
    df = df.drop(columns=drop_cols)

    # Fix the column names
    df = df.rename(columns={"2urvived": "Survived"})

    # Convert categorical columns to appropriate types
    df["Sex"] = df["Sex"].map({0: "male", 1: "female"})
    df["Embarked"] = df["Embarked"].map({0: "C", 1: "Q", 2: "S"})
    df["Survived"] = df["Survived"].astype(int)

    print(f"\nAfter cleanup: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns after cleanup: {', '.join(df.columns)}")

    # Basic statistics
    print("\nBasic statistics:")
    print(f"Survival rate: {df['Survived'].mean():.2%}")
    print(f"Average age: {df['Age'].mean():.1f} years")
    print(f"Average fare: ${df['Fare'].mean():.2f}")

    # Save a cleaned version of the dataset
    cleaned_path = os.path.join(backend_dir, "titanic_cleaned.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"\nCleaned dataset saved to {cleaned_path}")

    # Initialize the analysis engine
    print("\nInitializing AnalysisEngine...")
    engine = AnalysisEngine()

    # Step 1: Preprocess the data
    print("\nPreprocessing data...")
    preprocess_result = engine.preprocess_data(df)

    if "error" in preprocess_result:
        print(f"Error in preprocessing: {preprocess_result['error']}")
        return

    data_id = preprocess_result["data_id"]
    print(f"Preprocessing complete. Data ID: {data_id}")
    print(f"Column types detected: {preprocess_result['column_types']}")
    print(f"Missing values handled: {preprocess_result['missing_values_handled']}")

    # Step 2: Perform clustering analysis
    print("\nPerforming clustering analysis...")
    features = ["Age", "Fare"]
    clustering_result = engine.analyze_with_clustering(
        data_id=data_id, features=features, n_clusters=3
    )

    if "error" in clustering_result:
        print(f"Error in clustering: {clustering_result['error']}")
    else:
        print(f"Clustering complete. Analysis ID: {clustering_result['analysis_id']}")
        print(f"Number of clusters: {clustering_result['result']['n_clusters']}")
        print(f"Cluster sizes: {clustering_result['result']['cluster_sizes']}")

        # Visualize clusters if available
        if "clusters" in clustering_result["result"]:
            clusters = clustering_result["result"]["clusters"]

            # Create a scatter plot of Age vs Fare, colored by cluster
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(
                df["Age"], df["Fare"], c=clusters, cmap="viridis", alpha=0.7
            )
            plt.colorbar(scatter, label="Cluster")
            plt.xlabel("Age")
            plt.ylabel("Fare")
            plt.title("Passenger Clusters by Age and Fare")
            plt.savefig(os.path.join(backend_dir, "titanic_clusters.png"))
            print(
                f"Cluster visualization saved to {os.path.join(backend_dir, 'titanic_clusters.png')}"
            )

    # Step 3: Perform anomaly detection
    print("\nPerforming anomaly detection...")
    anomaly_result = engine.analyze_with_anomaly_detection(
        data_id=data_id, features=["Fare"], contamination=0.05
    )

    if "error" in anomaly_result:
        print(f"Error in anomaly detection: {anomaly_result['error']}")
    else:
        print(
            f"Anomaly detection complete. Analysis ID: {anomaly_result['analysis_id']}"
        )
        outlier_count = sum(anomaly_result["result"]["outliers"])
        print(
            f"Number of outliers detected: {outlier_count} ({outlier_count/df.shape[0]:.2%} of data)"
        )

        # Visualize outliers
        if "outliers" in anomaly_result["result"]:
            outliers = anomaly_result["result"]["outliers"]

            plt.figure(figsize=(10, 6))
            plt.scatter(
                range(len(df)),
                df["Fare"],
                c=["red" if x else "blue" for x in outliers],
                alpha=0.7,
            )
            plt.xlabel("Passenger Index")
            plt.ylabel("Fare")
            plt.title("Passenger Fare Outliers")
            plt.savefig(os.path.join(backend_dir, "titanic_outliers.png"))
            print(
                f"Outlier visualization saved to {os.path.join(backend_dir, 'titanic_outliers.png')}"
            )

    # Step 4: Create survival analysis by passenger class
    print("\nAnalyzing survival rates by passenger class...")
    class_survival = df.groupby("Pclass")["Survived"].mean()

    plt.figure(figsize=(8, 6))
    class_survival.plot(kind="bar")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate")
    plt.title("Survival Rate by Passenger Class")
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(backend_dir, "titanic_survival_by_class.png"))
    print(
        f"Survival analysis saved to {os.path.join(backend_dir, 'titanic_survival_by_class.png')}"
    )

    # Step 5: Analyze with natural language prompt
    print("\nPerforming natural language analysis...")
    nl_result = engine.analyze_data_with_prompt(
        data_id=data_id,
        prompt="What is the survival rate by passenger class and gender?",
    )

    if "error" in nl_result:
        print(f"Error in NL analysis: {nl_result['error']}")
    else:
        print(f"NL analysis complete. Analysis ID: {nl_result['analysis_id']}")
        print(f"Analysis plan: {nl_result['query_analysis']['analysis_plan']}")
        if "explanation" in nl_result["result"]:
            print(f"Explanation: {nl_result['result']['explanation']}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    analyze_titanic()
