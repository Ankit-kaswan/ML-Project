import os
from dataclasses import dataclass
from . import config


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(config.BASE_DATA_DIR, "model.pkl")

