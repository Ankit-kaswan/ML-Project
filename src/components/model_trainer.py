from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor

from src.configuration.model_trainer_config import ModelTrainerConfig
from src.exception import CustomException
from src.utils import save_object, evaluate_models
from src.logger import Logger
from src.configuration import config

# Initialize the custom logger
logger = Logger.get_logger()


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.models = {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            # "XGBRegressor": XGBRegressor(objective="reg:squarederror"),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor(),
        }
        self.model_config = config.MODEL_PARAMS

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Splitting training and test input data")

            # Ensure train/test arrays are not empty
            if train_array.size == 0 or test_array.size == 0:
                raise CustomException("Training or test dataset is empty. Please check data ingestion!")

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            parameters = self.model_config

            logger.info("Starting model evaluation...")
            model_report = evaluate_models(
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                models=self.models, param_grid=parameters
            )

            if not model_report:
                raise CustomException("Model evaluation failed. No valid models found.")

            best_model_name = max(model_report, key=lambda k: model_report[k]["score"])
            best_model_score = model_report[best_model_name]["score"]

            if best_model_name not in self.models:
                raise CustomException(f"Best model {best_model_name} not found in models dictionary!")

            best_model = model_report[best_model_name]["model"]

            if best_model_score < 0.6:
                logger.warning("No sufficiently good model found. Check data quality and feature selection.")
                raise CustomException("No best model found with an acceptable R² score.")

            logger.info(f"Best model: {best_model_name} with R² score: {best_model_score:.4f}")

            # Save the trained model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            logger.info("Model saved successfully.")

            # Predict using the best model
            try:
                predicted = best_model.predict(x_test)
                r2_square = r2_score(y_test, predicted)
                logger.info(f"Final R² Score on test data: {r2_square:.4f}")
            except Exception as e:
                raise CustomException("Prediction failed after model training!", cause=e)

            return r2_square

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise CustomException("Model training failed!", cause=e)


if __name__ == '__main__':
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_path, test_path)

    model_trainer = ModelTrainer()
    output = model_trainer.initiate_model_trainer(train_array=train_array, test_array=test_array)
    print(f"Final Model R² Score: {output:.4f}")
