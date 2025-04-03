import sys
import pandas as pd
import os
from typing import Any, Union
from src.exception import CustomException
from src.utils import load_object
from src.configuration.predict_config import PredictConfig
from src.logger import Logger


class PredictPipeline:
    def __init__(self):
        self.config = PredictConfig()
        self.logger = Logger.get_logger()

    def predict(self, features: pd.DataFrame) -> Union[pd.Series, Any]:
        """Loads the trained model and preprocessor, then makes predictions."""
        try:
            if not os.path.exists(self.config.model_path) or not os.path.exists(self.config.preprocessor_path):
                raise CustomException("Model or preprocessor file not found!")

            self.logger.info("Loading model and preprocessor...")
            model = load_object(file_path=self.config.model_path)
            preprocessor = load_object(file_path=self.config.preprocessor_path)

            self.logger.info("Transforming input features...")
            data_scaled = preprocessor.transform(features)

            self.logger.info("Making predictions...")
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise CustomException("Prediction failed: ", cause=e)

