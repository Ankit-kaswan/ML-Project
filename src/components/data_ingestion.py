import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.configuration.data_ingestion_config import DataIngestionConfig
from src.logger import Logger


# Initialize the custom logger
logger = Logger.get_logger()


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Starting data ingestion process...")
        try:
            if not os.path.exists(self.ingestion_config.dataset_file):
                raise CustomException(f"Dataset file not found: {self.ingestion_config.dataset_file}")

            df = pd.read_csv(self.ingestion_config.dataset_file)
            logger.info("Dataset successfully loaded into a DataFrame.")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logger.info("Splitting dataset into training and test sets...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=4)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logger.info("Data ingestion completed successfully.")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException("Data ingestion failed!", cause=e)


if __name__ == '__main__':

    data_ingestion = DataIngestion()
    print(data_ingestion.initiate_data_ingestion())
