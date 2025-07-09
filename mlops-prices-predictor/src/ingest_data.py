import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Adds project_folder to path

from abc import ABC, abstractmethod
import pandas as pd
from utils.logger import setup_logger

# Setup logging configuration
logger = setup_logger("data_ingestion")

# Define an abstract class for Data Ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Abstract method to ingest data from a given file."""
        pass

# CSV Data Ingestor
class CSVDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Reads a CSV file and returns the content as a pandas DataFrame."""
        if not file_path.endswith(".csv"):
            logger.error("The provided file is not a .csv file.")
            raise ValueError("The provided file is not a .csv file.")

        if not os.path.exists(file_path):
            logger.error(f"The file {file_path} does not exist.")
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        logger.info(f"Reading CSV data from {file_path}")
        df = pd.read_csv(file_path)
      
        return df

# Factory to create Data Ingestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """Returns the appropriate DataIngestor based on file extension."""
        if file_extension == ".csv":
            return CSVDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")

# Example usage
if __name__ == "__main__":
    file_path = "/home/mulombi/Codebase/Python/mlops-prices-predictor/data/AmesHousing.csv"

    # Determine the file extension
    file_extension = os.path.splitext(file_path)[1]

    # Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.ingest(file_path)
    
    print("\nFirst 5 rows of the dataset:")
    print(df.head())