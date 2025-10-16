"""
Run All Feature Engineering Experiments
Author: [Your Name]
Date: October 14, 2025
Course: Machine Learning at Scale - MSc Data Science

Description:
This script runs all three feature engineering pipelines sequentially
and generates a comprehensive comparison report.
"""

import sys
import os
from utils import generate_sample_data, compare_results
import time


def run_experiment(script_name: str, framework_name: str):
    """
    Run a single experiment script.

    Args:
        script_name: Name of the Python script to run
        framework_name: Name of the framework for display
    """
    print(f"\n{'='*80}")
    print(f"Running {framework_name} Experiment")
    print(f"{'='*80}\n")

    try:
        # Import and run the module
        if script_name == "01_spark_mllib_feature_engineering":
            from src import spark_mllib_feature_engineering as module
        elif script_name == "02_dask_feature_engineering":
            from src import dask_feature_engineering as module
        elif script_name == "03_pandas_on_spark_feature_engineering":
            from src import pandas_on_spark_feature_engineering as module
        else:
            raise ValueError(f"Unknown script: {script_name}")

        # Run main function
        module.main()

        print(f"✅ {framework_name} experiment completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error in {framework_name} experiment: {str(e)}")
        return False


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("SCALABLE FEATURE ENGINEERING - COMPARATIVE EXPERIMENTS")
    print("Machine Learning at Scale - MSc Data Science")
    print("=" * 80)

    # Step 1: Generate sample data if it doesn't exist
    data_path = "data/ecommerce_data.csv"

    if not os.path.exists(data_path):
        print("\n📊 Generating synthetic dataset...")
        os.makedirs("data", exist_ok=True)
        generate_sample_data(n_rows=2000000, output_path=data_path)
    else:
        print(f"\n✓ Dataset found at: {data_path}")

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Step 2: Run all experiments
    experiments = [
        ("01_spark_mllib_feature_engineering", "Apache Spark MLlib"),
        ("02_dask_feature_engineering", "Dask"),
        ("03_pandas_on_spark_feature_engineering", "Pandas on Spark"),
    ]

    results_summary = []
    start_time = time.time()

    for script, framework in experiments:
        success = run_experiment(script, framework)
        results_summary.append((framework, success))

        # Add delay between experiments to allow cleanup
        if framework != experiments[-1][1]:
            print("\n⏸️  Waiting 5 seconds before next experiment...")
            time.sleep(5)

    total_time = time.time() - start_time

    # Step 3: Generate summary report
    print("\n\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    for framework, success in results_summary:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{framework:.<50} {status}")

    print(f"\nTotal Experiment Time: {total_time:.2f} seconds")
    print("=" * 80)

    # Step 4: Load and compare results
    print("\n📊 Loading results for comparison...")

    try:
        import pandas as pd

        results_files = [
            "results/spark_results.txt",
            "results/dask_results.txt",
            "results/pandas_spark_results.txt",
        ]

        comparison_data = []

        for filepath in results_files:
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    lines = f.readlines()
                    framework = lines[0].split(": ")[1].strip()
                    exec_time = float(lines[1].split(": ")[1].split()[0])
                    peak_mem = float(lines[2].split(": ")[1].split()[0])

                    comparison_data.append(
                        {
                            "Framework": framework,
                            "Execution Time (s)": exec_time,
                            "Peak Memory (GB)": peak_mem,
                        }
                    )

        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)

            print("\n" + "=" * 80)
            print("PERFORMANCE COMPARISON")
            print("=" * 80 + "\n")
            print(df_comparison.to_string(index=False))

            # Calculate speedups
            print("\n" + "-" * 80)
            print("RELATIVE PERFORMANCE (Spark MLlib as baseline)")
            print("-" * 80)

            baseline_time = df_comparison.loc[
                df_comparison["Framework"] == "Apache Spark MLlib", "Execution Time (s)"
            ].values[0]

            for _, row in df_comparison.iterrows():
                speedup = baseline_time / row["Execution Time (s)"]
                print(f"{row['Framework']:.<40} {speedup:.2f}x")

            print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print(f"⚠️  Could not generate comparison: {str(e)}")

    print("\n✅ All experiments completed!")
    print("📁 Results saved in: results/")
    print("\n")


if __name__ == "__main__":
    main()
