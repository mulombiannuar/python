from abc import ABC, abstractmethod

import pandas as pd


# Abstract Base Class for Data Ingestion Strategies
# --------------------------------------------------
# This class defines a common interface for data ingestion strategies.
# Subclasses must implement the ingest method.
class DataIngestionStrategy(ABC) :
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Perform a specific type of data ingestion.

        Parameters:
        file_path (str): The path to the file to be ingested.

        Returns:
        pd.DataFrame: The ingested dataframe.
        """
        pass


# Concrete Strategy for Data CSV Data Ingestion 
# --------------------------------------------
# This strategy ingests data from a CSV file and returns a DataFrame.
class CSVDataIngestion(DataIngestionStrategy):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Ingests data from a CSV file.

        Parameters:
        file_path (str): The path to the CSV file.

        Returns:
        pd.DataFrame: The ingested dataframe.
        """
        df = pd.read_csv(file_path)
        print("\nFirst 5 rows of the dataset:")
        print(df.head())

        return df


# Context Class that uses a DataIngestionStrategy
# ------------------------------------------------
# This class allows you to switch between different data ingestion strategies.
class DataIngestor:
    def __init__(self, strategy: DataIngestionStrategy):
        """
        Initializes the DataIngestor with a specific ingestion strategy.

        Parameters:
        strategy (DataIngestionStrategy): The strategy to be used for data ingestion.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataIngestionStrategy):
        """
        Sets a new strategy for the DataIngestor.

        Parameters:
        strategy (DataIngestionStrategy): The new strategy to be used for data ingestion.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_ingestion(self,  file_path: str) -> pd.DataFrame:
        """
        Executes the ingestion using the current strategy.

        Parameters:
        file_path (str): The path to the CSV file.

        Returns:
        pd.DataFrame: The ingested dataframe.
        """

        return self._strategy.ingest(file_path)


# Example usage
if __name__ == "__main__":
    # Example usage of the DataIngestor with different strategies.

    # Load the data
    df = pd.read_csv('../../extracted_data/AmesHousing.csv')

    # Initialize the Data ingestor with a specific strategy
    ingestor = DataIngestor(CSVDataIngestion())
    ingestor.execute_ingestion(df)
