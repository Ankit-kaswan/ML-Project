import os
import pandas as pd
from dataclasses import dataclass
from . import config
from src.exception import CustomException


@dataclass
class PredictConfig:
    """Configuration for the prediction pipeline."""
    model_path: str = os.path.join(config.BASE_DATA_DIR, "model.pkl")
    preprocessor_path: str = os.path.join(config.BASE_DATA_DIR, "preprocessor.pkl")


@dataclass
class CustomData:
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """Converts the input data into a pandas DataFrame for model prediction."""
        try:
            return pd.DataFrame([self.__dict__])
        except Exception as e:
            raise CustomException("Failed to create DataFrame", cause=e)