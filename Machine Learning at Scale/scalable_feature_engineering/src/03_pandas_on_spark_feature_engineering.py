"""
Pandas on Spark Feature Engineering Pipeline
Author: [Your Name]
Date: October 14, 2025
Course: Machine Learning at Scale - MSc Data Science

Description:
This script implements a comprehensive feature engineering pipeline using
Pandas on Spark API for familiar Pandas-style distributed processing.
"""

import pyspark.pandas as ps
from pyspark.sql import SparkSession
import numpy as np
from utils import PerformanceMonitor, save_results
import warnings

warnings.filterwarnings("ignore")


class PandasOnSparkFeatureEngineering:
    """Feature engineering pipeline using Pandas on Spark."""

    def __init__(
        self, data_path: str, app_name: str = "PandasOnSpark_FeatureEngineering"
    ):
        """
        Initialize Spark session for Pandas on Spark.

        Args:
            data_path: Path to input CSV file
            app_name: Spark application name
        """
        self.data_path = data_path

        # Initialize Spark
        self.spark = (
            SparkSession.builder.appName(app_name)
            .config("spark.driver.memory", "8g")
            .config("spark.executor.memory", "8g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.shuffle.partitions", "16")
            .getOrCreate()
        )

        self.spark.sparkContext.setLogLevel("ERROR")

        # Set Pandas on Spark configuration
        ps.set_option("compute.default_index_type", "distributed")
        ps.set_option("compute.ops_on_diff_frames", True)

        self.monitor = PerformanceMonitor("Pandas on Spark")
        self.df = None

    def load_data(self):
        """Load data using Pandas on Spark."""
        print("\n📂 Loading data...")
        self.df = ps.read_csv(self.data_path)

        print(f"   Columns: {len(self.df.columns)}")
        print(f"   Shape: {self.df.shape}")

        self.monitor.checkpoint("Data Loading")

    def handle_missing_values(self):
        """Impute missing values using median/mode strategies."""
        print("\n🔧 Handling missing values...")

        # Numerical columns - use median
        numerical_cols = ["transaction_amount", "product_price", "customer_age"]

        for col in numerical_cols:
            median_val = self.df[col].median()
            self.df[col] = self.df[col].fillna(median_val)
            print(f"   ✓ Imputed {col} with median: {median_val:.2f}")

        # Categorical columns - use mode
        categorical_cols = ["payment_method", "customer_segment", "region"]

        for col in categorical_cols:
            mode_val = self.df[col].mode()[0]
            self.df[col] = self.df[col].fillna(mode_val)
            print(f"   ✓ Imputed {col} with mode: {mode_val}")

        self.monitor.checkpoint("Missing Value Imputation")

    def encode_categorical_features(self):
        """Encode categorical variables using Pandas-style methods."""
        print("\n🏷️  Encoding categorical features...")

        # One-hot encoding for low-cardinality features
        low_cardinality_cols = ["payment_method", "region", "customer_segment"]

        print(f"   One-hot encoding: {low_cardinality_cols}")
        self.df = ps.get_dummies(
            self.df, columns=low_cardinality_cols, drop_first=True, dtype=int
        )

        # Label encoding for high-cardinality feature
        print("   Label encoding: product_category")
        self.df["product_category_encoded"] = (
            self.df["product_category"].astype("category").cat.codes
        )

        self.monitor.checkpoint("Categorical Encoding")

    def scale_numerical_features(self):
        """Standardize numerical features to zero mean and unit variance."""
        print("\n📊 Scaling numerical features...")

        numerical_cols = ["transaction_amount", "product_price", "customer_age"]

        # Compute statistics and standardize
        for col in numerical_cols:
            mean_val = self.df[col].mean()
            std_val = self.df[col].std()
            self.df[f"{col}_scaled"] = (self.df[col] - mean_val) / std_val
            print(f"   {col}: mean={mean_val:.2f}, std={std_val:.2f}")

        self.monitor.checkpoint("Numerical Scaling")

    def extract_temporal_features(self):
        """Extract temporal features from date columns."""
        print("\n📅 Extracting temporal features...")

        # Convert to datetime
        self.df["transaction_date"] = ps.to_datetime(self.df["transaction_date"])
        self.df["registration_date"] = ps.to_datetime(self.df["registration_date"])
        self.df["transaction_timestamp"] = ps.to_datetime(
            self.df["transaction_timestamp"]
        )

        # Extract components using Pandas-style accessors
        self.df["day_of_week"] = self.df["transaction_date"].dt.dayofweek
        self.df["month"] = self.df["transaction_date"].dt.month
        self.df["hour"] = self.df["transaction_timestamp"].dt.hour
        self.df["customer_tenure_days"] = (
            self.df["transaction_date"] - self.df["registration_date"]
        ).dt.days

        print("   ✓ Extracted: day_of_week, month, hour, customer_tenure_days")

        self.monitor.checkpoint("Temporal Feature Extraction")

    def create_aggregation_features(self):
        """Create customer-level aggregation features using groupby."""
        print("\n📈 Creating aggregation features...")

        # Compute aggregations
        agg_features = self.df.groupby("customer_id").agg(
            {"transaction_amount": ["count", "mean", "std", "min", "max"]}
        )

        # Flatten column names
        agg_features.columns = [
            "total_transactions",
            "avg_transaction_amount",
            "std_transaction_amount",
            "min_transaction_amount",
            "max_transaction_amount",
        ]

        # Fill NaN from std
        agg_features["std_transaction_amount"] = agg_features[
            "std_transaction_amount"
        ].fillna(0)

        # Reset index
        agg_features = agg_features.reset_index()

        print(f"   ✓ Computed aggregations for customers")

        # Merge back to main dataframe
        self.df = self.df.merge(agg_features, on="customer_id", how="left")

        self.monitor.checkpoint("Aggregation Features")

    def execute_pipeline(self):
        """Execute complete feature engineering pipeline."""
        self.monitor.start()

        # Execute all stages
        self.load_data()
        self.handle_missing_values()
        self.encode_categorical_features()
        self.scale_numerical_features()
        self.extract_temporal_features()
        self.create_aggregation_features()

        # Materialize final dataset
        print("\n⚡ Materializing final dataset...")
        final_count = len(self.df)
        print(f"   Final row count: {final_count:,}")

        # Display sample
        print("\n📋 Sample of transformed data:")
        sample_cols = [
            "customer_id",
            "transaction_amount",
            "day_of_week",
            "total_transactions",
            "avg_transaction_amount",
        ]
        print(self.df[sample_cols].head())

        # End monitoring
        results = self.monitor.end()

        return results

    def cleanup(self):
        """Stop Spark session and cleanup resources."""
        print("🧹 Cleaning up resources...")
        self.spark.stop()


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("PANDAS ON SPARK FEATURE ENGINEERING EXPERIMENT")
    print("=" * 80)

    # Configuration
    DATA_PATH = "data/ecommerce_data.csv"
    OUTPUT_PATH = "results/pandas_spark_results.txt"

    # Initialize and execute pipeline
    pipeline = PandasOnSparkFeatureEngineering(DATA_PATH)

    try:
        results = pipeline.execute_pipeline()
        save_results(results, OUTPUT_PATH)

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise

    finally:
        pipeline.cleanup()

    print("\n✅ Pandas on Spark experiment completed successfully!\n")


if __name__ == "__main__":
    main()
