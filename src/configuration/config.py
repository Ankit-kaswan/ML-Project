from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

BASE_DATA_DIR = BASE_DIR / "artifacts"
DATASET_FILE = BASE_DIR / "data/rawData.csv"
LOG_DIR = BASE_DIR / "logs"
TEMPLATES_DIR = BASE_DIR / "templates"

MODEL_PARAMS = {
    "Decision Tree": {
        "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
    },
    "Random Forest": {
        "n_estimators": [8, 16, 32, 64, 128, 256],
    },
    "Gradient Boosting": {
        "learning_rate": [0.1, 0.01, 0.05, 0.001],
        "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        "n_estimators": [8, 16, 32, 64, 128, 256],
    },
    "CatBoosting Regressor": {
        "depth": [6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "iterations": [30, 50, 100],
    },
    "AdaBoost Regressor": {
        "learning_rate": [0.1, 0.01, 0.5, 0.001],
        "n_estimators": [8, 16, 32, 64, 128, 256],
    },
}
