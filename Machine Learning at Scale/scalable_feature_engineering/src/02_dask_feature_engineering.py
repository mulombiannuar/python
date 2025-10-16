"""
Dask Feature Engineering Pipeline
Author: [Your Name]
Date: October 14, 2025
Course: Machine Learning at Scale - MSc Data Science

Description:
This script implements a comprehensive feature engineering pipeline using
Dask for scalable Python-native data processing on a 2M row e-commerce dataset.
"""

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from utils import PerformanceMonitor, save_results
import warnings

warnings.filterwarnings("ignore")


class DaskFeatureEngineering:
    """Feature engineering pipeline using Dask."""

    def __init__(self, data_path: str, n_workers: int = 8):
        """
        Initialize Dask client and configuration.

        Args:
            data_path: Path to input CSV file
            n_workers: Number of Dask workers
        """
        self.data_path = data_path
        self.n_workers = n_workers

        # Initialize Dask cluster
        print(f"🚀 Initializing Dask cluster with {n_workers} workers...")
        self.cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit="3GB",
            silence_logs=True,
        )
        self.client = Client(self.cluster)

        print(f"   Dashboard: {self.client.dashboard_link}")

        self.monitor = PerformanceMonitor("Dask")
        self.df = None

    def load_data(self):
        """Load data with optimized blocksize."""
        print("\n📂 Loading data...")
        self.df = dd.read_csv(self.data_path, blocksize="128MB", assume_missing=True)

        # Get basic info
        n_partitions = self.df.npartitions
        print(f"   Partitions: {n_partitions}")
        print(f"   Columns: {len(self.df.columns)}")

        self.monitor.checkpoint("Data Loading")

    def handle_missing_values(self):
        """Impute missing values using median/mode strategies."""
        print("\n🔧 Handling missing values...")

        # Numerical columns - use median
        numerical_cols = ["transaction_amount", "product_price", "customer_age"]

        for col in numerical_cols:
            median_val = self.df[col].median().compute()
            self.df[col] = self.df[col].fillna(median_val)
            print(f"   ✓ Imputed {col} with median: {median_val:.2f}")

        # Categorical columns - use mode
        categorical_cols = ["payment_method", "customer_segment", "region"]

        for col in categorical_cols:
            mode_val = self.df[col].mode().compute().iloc[0]
            self.df[col] = self.df[col].fillna(mode_val)
            print(f"   ✓ Imputed {col} with mode: {mode_val}")

        self.monitor.checkpoint("Missing Value Imputation")

    def encode_categorical_features(self):
        """Encode categorical variables using one-hot and label encoding."""
        print("\n🏷️  Encoding categorical features...")

        # One-hot encoding for low-cardinality features
        low_cardinality_cols = ["payment_method", "region", "customer_segment"]

        print(f"   One-hot encoding: {low_cardinality_cols}")
        self.df = dd.get_dummies(
            self.df,
            columns=low_cardinality_cols,
            prefix=low_cardinality_cols,
            drop_first=True,
        )

        # Label encoding for high-cardinality feature (product_category)
        print("   Label encoding: product_category")

        # Create mapping dictionary
        unique_categories = self.df["product_category"].unique().compute()
        category_map = {cat: idx for idx, cat in enumerate(unique_categories)}

        self.df["product_category_encoded"] = self.df["product_category"].map(
            category_map, meta=("product_category_encoded", "int64")
        )

        self.monitor.checkpoint("Categorical Encoding")

    def scale_numerical_features(self):
        """Standardize numerical features using partition-wise scaling."""
        print("\n📊 Scaling numerical features...")

        numerical_cols = ["transaction_amount", "product_price", "customer_age"]

        # Compute global statistics
        means = {}
        stds = {}

        for col in numerical_cols:
            means[col] = self.df[col].mean().compute()
            stds[col] = self.df[col].std().compute()
            print(f"   {col}: mean={means[col]:.2f}, std={stds[col]:.2f}")

        # Apply standardization
        for col in numerical_cols:
            self.df[f"{col}_scaled"] = (self.df[col] - means[col]) / stds[col]

        self.monitor.checkpoint("Numerical Scaling")

    def extract_temporal_features(self):
        """Extract temporal features from date columns."""
        print("\n📅 Extracting temporal features...")

        # Convert to datetime
        self.df["transaction_date"] = dd.to_datetime(self.df["transaction_date"])
        self.df["registration_date"] = dd.to_datetime(self.df["registration_date"])
        self.df["transaction_timestamp"] = dd.to_datetime(
            self.df["transaction_timestamp"]
        )

        # Extract components
        self.df["day_of_week"] = self.df["transaction_date"].dt.dayofweek
        self.df["month"] = self.df["transaction_date"].dt.month
        self.df["hour"] = self.df["transaction_timestamp"].dt.hour
        self.df["customer_tenure_days"] = (
            self.df["transaction_date"] - self.df["registration_date"]
        ).dt.days

        print("   ✓ Extracted: day_of_week, month, hour, customer_tenure_days")

        self.monitor.checkpoint("Temporal Feature Extraction")

    def create_aggregation_features(self):
        """Create customer-level aggregation features."""
        print("\n📈 Creating aggregation features...")

        # Compute customer aggregations
        agg_features = (
            self.df.groupby("customer_id")
            .agg({"transaction_amount": ["count", "mean", "std", "min", "max"]})
            .compute()
        )

        # Flatten column names
        agg_features.columns = [
            "total_transactions",
            "avg_transaction_amount",
            "std_transaction_amount",
            "min_transaction_amount",
            "max_transaction_amount",
        ]

        # Fill NaN from std (when only 1 transaction)
        agg_features["std_transaction_amount"] = agg_features[
            "std_transaction_amount"
        ].fillna(0)

        # Reset index for merging
        agg_features = agg_features.reset_index()

        print(f"   ✓ Computed aggregations for {len(agg_features):,} customers")

        # Convert to Dask DataFrame for merging
        agg_features_dd = dd.from_pandas(agg_features, npartitions=self.df.npartitions)

        # Merge back to main dataframe
        self.df = self.df.merge(agg_features_dd, on="customer_id", how="left")

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
        self.df_final = self.df.compute()
        print(f"   Final shape: {self.df_final.shape}")

        # Display sample
        print("\n📋 Sample of transformed data:")
        sample_cols = [
            "customer_id",
            "transaction_amount",
            "day_of_week",
            "total_transactions",
            "avg_transaction_amount",
        ]
        print(self.df_final[sample_cols].head())

        # End monitoring
        results = self.monitor.end()

        return results

    def cleanup(self):
        """Close Dask client and cleanup resources."""
        print("🧹 Cleaning up resources...")
        self.client.close()
        self.cluster.close()


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("DASK FEATURE ENGINEERING EXPERIMENT")
    print("=" * 80)

    # Configuration
    DATA_PATH = "data/ecommerce_data.csv"
    OUTPUT_PATH = "results/dask_results.txt"
    N_WORKERS = 8

    # Initialize and execute pipeline
    pipeline = DaskFeatureEngineering(DATA_PATH, n_workers=N_WORKERS)

    try:
        results = pipeline.execute_pipeline()
        save_results(results, OUTPUT_PATH)

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise

    finally:
        pipeline.cleanup()

    print("\n✅ Dask experiment completed successfully!\n")


if __name__ == "__main__":
    main()
