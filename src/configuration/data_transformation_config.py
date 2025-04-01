import os
from dataclasses import dataclass, field
from . import config


@dataclass
class DataTransformationConfig:
    """Configuration class for data transformation settings."""
    preprocessor_obj_file_path: str = field(
        default_factory=lambda: os.path.join(config.BASE_DATA_DIR, "preprocessor.pkl")
    )
