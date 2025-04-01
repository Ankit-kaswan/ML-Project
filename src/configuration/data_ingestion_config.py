import os
from dataclasses import dataclass
from . import config


@dataclass
class DataIngestionConfig:
    base_dir: str = config.BASE_DATA_DIR
    dataset_file: str = config.DATASET_FILE

    train_data_path: str = os.path.join(base_dir, "train.csv")
    test_data_path: str = os.path.join(base_dir, "test.csv")
    raw_data_path: str = os.path.join(base_dir, "data.csv")
