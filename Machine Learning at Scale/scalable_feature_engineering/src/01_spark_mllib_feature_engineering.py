"""
Apache Spark MLlib Feature Engineering Pipeline
Author: [Your Name]
Date: October 14, 2025
Course: Machine Learning at Scale - MSc Data Science

Description:
This script implements a comprehensive feature engineering pipeline using
Apache Spark MLlib for scalable data processing on a 2M row e-commerce dataset.
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import (
    Imputer,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
)
from pyspark.ml import Pipeline
from pyspark.sql.functions import (
    col,
    dayofweek,
    month,
    hour,
    datediff,
    current_date,
    count,
    avg,
    stddev,
)
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from utils import PerformanceMonitor, save_results
import warnings

warnings.filterwarnings("ignore")


class SparkMLlibFeatureEngineering:
    """Feature engineering pipeline using Apache Spark MLlib."""

    def __init__(self, data_path: str, app_name: str = "SparkMLlib_FeatureEngineering"):
        """
        Initialize Spark session and configuration.

        Args:
            data_path: Path to input CSV file
            app_name: Spark application name
        """
        self.data_path = data_path
        self.spark = (
            SparkSession.builder.appName(app_name)
            .config("spark.driver.memory", "8g")
            .config("spark.executor.memory", "8g")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.default.parallelism", "16")
            .getOrCreate()
        )

        self.spark.sparkContext.setLogLevel("ERROR")
        self.monitor = PerformanceMonitor("Apache Spark MLlib")
        self.df = None
        self.df_transformed = None

    def load_data(self):
        """Load data with optimized partitioning."""
        print("📂 Loading data...")
        self.df = self.spark.read.csv(self.data_path, header=True, inferSchema=True)
        self.df = self.df.repartition(16)  # Match available cores
        self.monitor.checkpoint("Data Loading")

        print(f"   Rows: {self.df.count():,}")
        print(f"   Columns: {len(self.df.columns)}")
        print(f"   Partitions: {self.df.rdd.getNumPartitions()}")

    def handle_missing_values(self):
        """Impute missing values using median for numerical columns."""
        print("\n🔧 Handling missing values...")

        numerical_cols = ["transaction_amount", "product_price", "customer_age"]

        imputer = Imputer(
            inputCols=numerical_cols,
            outputCols=[f"imp_{col}" for col in numerical_cols],
        ).setStrategy("median")

        self.df = imputer.fit(self.df).transform(self.df)

        # Handle categorical missing values with mode
        categorical_cols = ["payment_method", "customer_segment", "region"]
        for col_name in categorical_cols:
            mode_value = (
                self.df.groupBy(col_name).count().orderBy(F.desc("count")).first()[0]
            )
            self.df = self.df.fillna({col_name: mode_value})

        self.monitor.checkpoint("Missing Value Imputation")

    def encode_categorical_features(self):
        """Encode categorical variables using StringIndexer and OneHotEncoder."""
        print("\n🏷️  Encoding categorical features...")

        # String indexing for categorical columns
        categorical_cols = [
            "payment_method",
            "region",
            "product_category",
            "customer_segment",
        ]

        indexers = [
            StringIndexer(
                inputCol=col_name, outputCol=f"{col_name}_idx", handleInvalid="keep"
            )
            for col_name in categorical_cols
        ]

        # One-hot encoding for low-cardinality features
        encoder = OneHotEncoder(
            inputCols=["payment_method_idx", "region_idx", "customer_segment_idx"],
            outputCols=["payment_method_vec", "region_vec", "customer_segment_vec"],
            handleInvalid="keep",
        )

        # Build and execute pipeline
        pipeline = Pipeline(stages=indexers + [encoder])
        model = pipeline.fit(self.df)
        self.df = model.transform(self.df)

        self.monitor.checkpoint("Categorical Encoding")

    def scale_numerical_features(self):
        """Standardize numerical features to zero mean and unit variance."""
        print("\n📊 Scaling numerical features...")

        numerical_cols = [
            "imp_transaction_amount",
            "imp_product_price",
            "imp_customer_age",
        ]

        # Assemble features into vector
        assembler = VectorAssembler(
            inputCols=numerical_cols, outputCol="numerical_features"
        )

        # Apply standard scaling
        scaler = StandardScaler(
            inputCol="numerical_features",
            outputCol="scaled_features",
            withMean=True,
            withStd=True,
        )

        self.df = assembler.transform(self.df)
        scaler_model = scaler.fit(self.df)
        self.df = scaler_model.transform(self.df)

        self.monitor.checkpoint("Numerical Scaling")

    def extract_temporal_features(self):
        """Extract temporal features from date columns."""
        print("\n📅 Extracting temporal features...")

        # Convert to timestamp if needed
        self.df = self.df.withColumn(
            "transaction_date", F.to_timestamp("transaction_date")
        )
        self.df = self.df.withColumn(
            "registration_date", F.to_timestamp("registration_date")
        )

        # Extract temporal components
        self.df = (
            self.df.withColumn("day_of_week", dayofweek(col("transaction_date")))
            .withColumn("month", month(col("transaction_date")))
            .withColumn("hour", hour(col("transaction_timestamp")))
            .withColumn(
                "customer_tenure_days",
                datediff(col("transaction_date"), col("registration_date")),
            )
        )

        self.monitor.checkpoint("Temporal Feature Extraction")

    def create_aggregation_features(self):
        """Create customer-level aggregation features using window functions."""
        print("\n📈 Creating aggregation features...")

        # Define window specification
        window_spec = Window.partitionBy("customer_id")

        # Calculate aggregations
        self.df = (
            self.df.withColumn("total_transactions", count("*").over(window_spec))
            .withColumn(
                "avg_transaction_amount", avg("transaction_amount").over(window_spec)
            )
            .withColumn(
                "std_transaction_amount", stddev("transaction_amount").over(window_spec)
            )
            .withColumn(
                "max_transaction_amount", F.max("transaction_amount").over(window_spec)
            )
            .withColumn(
                "min_transaction_amount", F.min("transaction_amount").over(window_spec)
            )
        )

        # Fill NaN from stddev when only one transaction
        self.df = self.df.fillna(0, subset=["std_transaction_amount"])

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

        # Trigger computation and cache
        print("\n⚡ Materializing final dataset...")
        self.df_transformed = self.df.cache()
        final_count = self.df_transformed.count()
        print(f"   Final row count: {final_count:,}")

        # Display sample
        print("\n📋 Sample of transformed data:")
        self.df_transformed.select(
            "customer_id",
            "transaction_amount",
            "day_of_week",
            "total_transactions",
            "avg_transaction_amount",
        ).show(5, truncate=False)

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
    print("APACHE SPARK MLLIB FEATURE ENGINEERING EXPERIMENT")
    print("=" * 80)

    # Configuration
    DATA_PATH = "data/ecommerce_data.csv"
    OUTPUT_PATH = "results/spark_results.txt"

    # Initialize and execute pipeline
    pipeline = SparkMLlibFeatureEngineering(DATA_PATH)

    try:
        results = pipeline.execute_pipeline()
        save_results(results, OUTPUT_PATH)

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise

    finally:
        pipeline.cleanup()

    print("\n✅ Spark MLlib experiment completed successfully!\n")


if __name__ == "__main__":
    main()
