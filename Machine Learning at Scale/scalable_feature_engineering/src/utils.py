"""
Utility Functions for Scalable Feature Engineering Experiments
Author: [Your Name]
Date: October 14, 2025
Course: Machine Learning at Scale - MSc Data Science
"""

import time
import psutil
import os
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd


class PerformanceMonitor:
    """Monitor execution time and memory usage for experiments."""

    def __init__(self, framework_name: str):
        self.framework_name = framework_name
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = None
        self.metrics = {}

    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        print(f"\n{'='*60}")
        print(f"Starting {self.framework_name} Feature Engineering Pipeline")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Initial Memory: {self.start_memory:.2f} GB")
        print(f"{'='*60}\n")

    def checkpoint(self, operation_name: str):
        """Record checkpoint for specific operation."""
        current_time = time.time()
        current_memory = self._get_memory_usage()
        elapsed = current_time - self.start_time

        self.metrics[operation_name] = {
            "elapsed_time": elapsed,
            "memory_usage": current_memory,
        }

        print(f"✓ {operation_name:.<50} {elapsed:.2f}s | {current_memory:.2f} GB")

    def end(self):
        """End monitoring and display summary."""
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        self.peak_memory = self._get_peak_memory()

        print(f"\n{'='*60}")
        print(f"Pipeline Completed: {self.framework_name}")
        print(f"{'='*60}")
        print(f"Total Execution Time: {total_time:.2f} seconds")
        print(f"Peak Memory Usage: {self.peak_memory:.2f} GB")
        print(f"Memory Increase: {(self.peak_memory - self.start_memory):.2f} GB")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        return {
            "framework": self.framework_name,
            "total_time": total_time,
            "peak_memory": self.peak_memory,
            "operations": self.metrics,
        }

    @staticmethod
    def _get_memory_usage() -> float:
        """Get current memory usage in GB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)

    @staticmethod
    def _get_peak_memory() -> float:
        """Get peak memory usage in GB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)


def save_results(results: Dict, output_file: str):
    """Save experiment results to file."""
    with open(output_file, "w") as f:
        f.write(f"Framework: {results['framework']}\n")
        f.write(f"Total Execution Time: {results['total_time']:.2f} seconds\n")
        f.write(f"Peak Memory Usage: {results['peak_memory']:.2f} GB\n\n")
        f.write("Operation Breakdown:\n")
        f.write("-" * 60 + "\n")

        for operation, metrics in results["operations"].items():
            f.write(
                f"{operation}: {metrics['elapsed_time']:.2f}s | "
                f"{metrics['memory_usage']:.2f} GB\n"
            )

    print(f"Results saved to: {output_file}")


def generate_sample_data(
    n_rows: int = 2000000, output_path: str = "ecommerce_data.csv"
):
    """Generate synthetic e-commerce dataset for experiments."""
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    print(f"Generating synthetic dataset with {n_rows:,} rows...")

    np.random.seed(42)

    # Generate base data
    data = {
        "customer_id": np.random.randint(1, 50001, n_rows),
        "transaction_id": range(1, n_rows + 1),
        "transaction_amount": np.random.lognormal(4, 1, n_rows),
        "product_id": np.random.randint(1, 5001, n_rows),
        "product_category": np.random.choice(
            [f"Category_{i}" for i in range(1, 121)], n_rows
        ),
        "product_price": np.random.uniform(5, 500, n_rows),
        "payment_method": np.random.choice(
            [
                "Credit Card",
                "Debit Card",
                "PayPal",
                "Apple Pay",
                "Google Pay",
                "Bank Transfer",
            ],
            n_rows,
        ),
        "customer_age": np.random.randint(18, 86, n_rows),
        "customer_segment": np.random.choice(
            ["Premium", "Standard", "Basic", "New", "VIP"], n_rows
        ),
        "region": np.random.choice(
            [
                "North",
                "South",
                "East",
                "West",
                "Central",
                "Northeast",
                "Southeast",
                "Southwest",
                "Northwest",
                "Midwest",
                "Coastal",
                "Inland",
            ],
            n_rows,
        ),
        "purchase_frequency": np.random.poisson(10, n_rows),
    }

    # Generate temporal data
    start_date = datetime(2023, 1, 1)
    data["transaction_date"] = [
        start_date + timedelta(days=int(x)) for x in np.random.uniform(0, 730, n_rows)
    ]

    data["transaction_timestamp"] = [
        dt + timedelta(hours=np.random.randint(0, 24), minutes=np.random.randint(0, 60))
        for dt in data["transaction_date"]
    ]

    data["registration_date"] = [
        dt - timedelta(days=int(x))
        for dt, x in zip(data["transaction_date"], np.random.uniform(30, 1825, n_rows))
    ]

    # Generate text data
    descriptions = [
        "Premium quality product",
        "Best seller item",
        "Limited edition",
        "New arrival",
        "Clearance sale",
        "Featured product",
        "Top rated",
    ]
    data["product_description"] = np.random.choice(descriptions, n_rows)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Introduce missing values (5-8% for numerical, 2-3% for categorical)
    numerical_cols = ["transaction_amount", "product_price", "customer_age"]
    for col in numerical_cols:
        mask = np.random.random(n_rows) < 0.06
        df.loc[mask, col] = np.nan

    categorical_cols = ["payment_method", "customer_segment", "region"]
    for col in categorical_cols:
        mask = np.random.random(n_rows) < 0.025
        df.loc[mask, col] = np.nan

    # Save to CSV
    df.to_csv(output_path, index=False)
    file_size = os.path.getsize(output_path) / (1024**3)

    print(f"✓ Dataset generated successfully!")
    print(f"  - Rows: {n_rows:,}")
    print(f"  - Columns: {len(df.columns)}")
    print(f"  - File size: {file_size:.2f} GB")
    print(f"  - Saved to: {output_path}\n")

    return df


def compare_results(results_list: List[Dict]):
    """Compare results from multiple frameworks."""
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS SUMMARY")
    print("=" * 80 + "\n")

    # Create comparison table
    comparison_df = pd.DataFrame(
        [
            {
                "Framework": r["framework"],
                "Total Time (s)": f"{r['total_time']:.2f}",
                "Peak Memory (GB)": f"{r['peak_memory']:.2f}",
            }
            for r in results_list
        ]
    )

    print(comparison_df.to_string(index=False))
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Generate sample dataset if needed
    print("Utility module for Scalable Feature Engineering Experiments")
    print("Run individual experiment scripts to execute pipelines.")
