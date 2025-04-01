from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

BASE_DATA_DIR = BASE_DIR / "artifacts"
DATASET_FILE = BASE_DIR / "data/rawData.csv"
LOG_DIR = BASE_DIR / "logs"
