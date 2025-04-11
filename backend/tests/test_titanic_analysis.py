import os
import sys
import pandas as pd
import pytest
from pathlib import Path

# Add the app module to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

# Import AnalysisEngine from app
from app.services.analysis_service import AnalysisEngine


class TestTitanicAnalysis:
    """Test suite for analyzing the Titanic dataset with AnalysisEngine"""

    @pytest.fixture
    def titanic_df(self):
        """Load the Titanic dataset as a fixture"""
        titanic_path = os.path.join(backend_dir, "titanic.csv")
        return pd.read_csv(titanic_path)

    @pytest.fixture
    def analysis_engine(self):
        """Create an instance of AnalysisEngine as a fixture"""
        return AnalysisEngine()

    def test_preprocessing(self, titanic_df, analysis_engine):
        """Test data preprocessing with Titanic dataset"""
        # Preprocess the data
        preprocess_result = analysis_engine.preprocess_data(titanic_df)

        # Verify preprocessing succeeded
        assert "error" not in preprocess_result
        assert "data_id" in preprocess_result

        # Check if the column types were detected correctly
        assert "column_types" in preprocess_result
        assert "numeric" in preprocess_result["column_types"]
        assert "categorical" in preprocess_result["column_types"]

        # Verify missing values were handled
        assert "missing_values_handled" in preprocess_result

        return preprocess_result["data_id"]

    def test_clustering(self, titanic_df, analysis_engine):
        """Test clustering analysis with Titanic dataset"""
        # Preprocess the data first
        preprocess_result = analysis_engine.preprocess_data(titanic_df)
        data_id = preprocess_result["data_id"]

        # Perform clustering analysis
        # We'll use age and fare as features for clustering passengers
        features = ["Age", "Fare"]
        clustering_result = analysis_engine.analyze_with_clustering(
            data_id=data_id,
            features=features,
            n_clusters=3,  # Try clustering passengers into 3 groups
        )

        # Verify clustering succeeded
        assert "error" not in clustering_result
        assert "result" in clustering_result
        assert "clusters" in clustering_result["result"]

        # Check if we have the expected number of clusters
        assert clustering_result["result"]["n_clusters"] == 3

    def test_anomaly_detection(self, titanic_df, analysis_engine):
        """Test anomaly detection with Titanic dataset"""
        # Preprocess the data first
        preprocess_result = analysis_engine.preprocess_data(titanic_df)
        data_id = preprocess_result["data_id"]

        # Perform anomaly detection
        # Looking for outliers in passenger fares
        features = ["Fare"]
        anomaly_result = analysis_engine.analyze_with_anomaly_detection(
            data_id=data_id,
            features=features,
            contamination=0.05,  # Assume about 5% of data points are outliers
        )

        # Verify anomaly detection succeeded
        assert "error" not in anomaly_result
        assert "result" in anomaly_result
        assert "outliers" in anomaly_result["result"]

        # Verify we got some outliers but not too many
        outlier_count = sum(anomaly_result["result"]["outliers"])
        assert 0 < outlier_count < len(titanic_df) * 0.1

    def test_nl_query_analysis(self, titanic_df, analysis_engine):
        """Test natural language query analysis with Titanic dataset"""
        # Preprocess the data first
        preprocess_result = analysis_engine.preprocess_data(titanic_df)
        data_id = preprocess_result["data_id"]

        # Analyze with a natural language query
        nl_result = analysis_engine.analyze_data_with_prompt(
            data_id=data_id, prompt="What is the survival rate by passenger class?"
        )

        # Verify NL analysis succeeded
        assert "error" not in nl_result
        assert "result" in nl_result
        assert "query_analysis" in nl_result

        # Check that the analysis plan was generated
        assert "analysis_plan" in nl_result["query_analysis"]
