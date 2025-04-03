from src.exception import CustomException
from src.logger import Logger
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        self.logger = Logger.get_logger()

    def run_pipeline(self):
        """Executes the full training pipeline: Data Ingestion → Transformation → Model Training."""
        try:
            self.logger.info("Starting Training Pipeline...")

            # Step 1: Data Ingestion
            self.logger.info("Running Data Ingestion...")
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            self.logger.info("Running Data Transformation...")
            data_transformation = DataTransformation()
            train_array, test_array, _ = data_transformation.initiate_data_transformation(train_path, test_path)

            # Step 3: Model Training
            self.logger.info("Running Model Training...")
            model_trainer = ModelTrainer()
            r2_score = model_trainer.initiate_model_trainer(train_array=train_array, test_array=test_array)

            self.logger.info(f"Training Pipeline Completed. Final R² Score: {r2_score:.4f}")

        except Exception as e:
            self.logger.error(f"Training Pipeline failed: {str(e)}")
            raise CustomException("Training Pipeline execution failed!", cause=e)


if __name__ == '__main__':
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
