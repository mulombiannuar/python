from abc import ABC, abstractmethod

import pandas as pd


# Abstract Base Class for Data Inspection Strategies
# --------------------------------------------------
# This class defines a common interface for data inspection strategies.
# Subclasses must implement the inspect method.
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of data inspection.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be performed.

        Returns:
        None: This method prints the inspection results directly.
        """
        pass


# Concrete Strategy for Data Types Inspection
# --------------------------------------------
# This strategy inspects the data types of each column and counts non-null values.
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the dataframe columns.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the data types and non-null counts to the console.
        """
        print("\nData Types and Non-null Counts:")
        print(df.info())



# Concrete Strategy for Summary Statistics Inspection
# -----------------------------------------------------
# This strategy provides summary statistics for both numerical and categorical features.
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Prints summary statistics for numerical and categorical features.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints summary statistics to the console.
        """
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))
        


# Concrete Strategy for Dataset Shape Inspection
# ----------------------------------------------
# This strategy prints the shape (rows and columns) of the dataframe.
class DatasetShapeInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the shape of the dataframe (rows, columns).

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the shape to the console.
        """
        print("\nDataset Shape:")
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        


# Concrete Strategy for Column Type Classification
# ------------------------------------------------
# This strategy separates and prints categorical and numeric columns.
class ColumnTypeClassificationStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints categorical and numeric columns from the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints lists of categorical and numeric column names.
        """
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        print("\nCategorical Columns:")
        print(categorical_cols if categorical_cols else "None")

        print("\nNumeric Columns:")
        print(numeric_cols if numeric_cols else "None")
        

# Concrete Strategy for Value Counts Inspection
# ---------------------------------------------
# This strategy prints value counts and unique category counts for specified columns.
class ValueCountsInspectionStrategy(DataInspectionStrategy):
    def __init__(self, columns):
        """
        Initializes the strategy with the columns to inspect.

        Parameters:
        columns (list): List of column names to compute value counts for.
        """
        self.columns = columns

    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints value counts and number of unique categories for each specified column.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None
        """
        for col in self.columns:
            if col in df.columns:
                print(f"\nValue Counts for '{col}':\n" + "-" * 40)
                print(df[col].value_counts(dropna=False))
                print("-" * 40)
                print(f"Unique Categories: {df[col].nunique(dropna=False)}")
                print("*" * 50)
            else:
                print(f"\nColumn '{col}' not found in the DataFrame.")



# Concrete Strategy for Column Names Inspection
# ---------------------------------------------
# This strategy lists all column names in the dataframe.
class ColumnNamesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the column names of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None
        """
        print("\nColumn Names:")
        print("-" * 40)
        for col in df.columns:
            print(col)
        print("-" * 40)
        print(f"Total Columns: {len(df.columns)}")



# Concrete Strategy for Duplicate Rows Inspection
# -----------------------------------------------
# This strategy identifies duplicate rows and counts them.
class DuplicateRowsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and reports the number of duplicate rows in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None
        """
        duplicate_count = df.duplicated().sum()
        total_rows = len(df)

        print("\nDuplicate Rows Check:")
        print("-" * 40)
        print(f"Total Rows: {total_rows}")
        print(f"Duplicate Rows: {duplicate_count}")
        print(f"Unique Rows: {total_rows - duplicate_count}")
        print("-" * 40)


# Context Class that uses a DataInspectionStrategy
# ------------------------------------------------
# This class allows you to switch between different data inspection strategies.
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializes the DataInspector with a specific inspection strategy.

        Parameters:
        strategy (DataInspectionStrategy): The strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Sets a new strategy for the DataInspector.

        Parameters:
        strategy (DataInspectionStrategy): The new strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes the inspection using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Executes the strategy's inspection method.
        """
        self._strategy.inspect(df)


# Example usage
if __name__ == "__main__":
    # Example usage of the DataInspector with different strategies.

    # Load the data
    df = pd.read_csv('../../data/AmesHousing.csv')

    # Initialize the Data Inspector with a specific strategy
    # inspector = DataInspector(DataTypesInspectionStrategy())
    # inspector.execute_inspection(df)

    # Change strategy to Summary Statistics and execute
    # inspector = DataInspector(SummaryStatisticsInspectionStrategy())
    # inspector.execute_inspection(df)
    
    # Change strategy to Dataset Shape and execute
    # inspector = DataInspector(DatasetShapeInspectionStrategy())
    # inspector.execute_inspection(df)
    
    # Change strategy to Column Type Classification and execute
    inspector = DataInspector(ColumnTypeClassificationStrategy())
    inspector.execute_inspection(df)
    
    # Column names inspection
    # col_names_strategy = ColumnNamesInspectionStrategy()
    # col_names_strategy.inspect(df)

    # Duplicate rows inspection
    # dup_rows_strategy = DuplicateRowsInspectionStrategy()
    # dup_rows_strategy.inspect(df)