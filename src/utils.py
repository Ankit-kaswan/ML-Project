import os
import joblib
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import Logger


# Initialize the custom logger
logger = Logger.get_logger()


def save_object(file_path, obj):
    """Saves an object using joblib (optimized for ML models)."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Object saved successfully at: {file_path}")
    except Exception as e:
        logger.error(f"Error saving object: {str(e)}")
        raise CustomException("Failed to save object!", cause=e)


def load_object(file_path):
    """Loads an object using joblib."""
    try:
        obj = joblib.load(file_path)
        logger.info(f"Object loaded successfully from: {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading object from {file_path}: {str(e)}")
        raise CustomException("Object load failed!", cause=e)


def evaluate_models(x_train, y_train, x_test, y_test, models, param_grid):
    """
    Trains multiple models using GridSearchCV, selects the best hyperparameters,
    and evaluates their performance.

    Returns a dictionary with model names and their R² scores.
    """
    try:
        # Check if training data is valid
        if x_train is None or y_train is None or x_test is None or y_test is None:
            raise CustomException("Training or test data is None. Check data ingestion!")

        if x_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise CustomException("Training data is empty. Check data preprocessing!")

        report = {}

        for model_name, model in models.items():
            logger.info(f"Training model: {model_name}")

            # Hyperparameter tuning
            gs = GridSearchCV(
                model, param_grid.get(model_name, {}), cv=min(3, len(x_train)), scoring='r2', n_jobs=-1, verbose=1
            )

            gs.fit(x_train, y_train)

            # Ensure GridSearchCV returned a valid model
            if not hasattr(gs, "best_estimator_") or gs.best_estimator_ is None:
                logger.warning(f"GridSearchCV failed to find a valid model for {model_name}. Skipping...")
                continue

            best_model = gs.best_estimator_
            best_params = gs.best_params_

            logger.info(f"{model_name} Best Params: {best_params}")

            try:
                # Check if the model is properly fitted
                if not hasattr(best_model, "predict"):
                    raise NotFittedError(f"{model_name} model is not fitted yet!")

                # Predictions
                y_train_pred = best_model.predict(x_train)
                y_test_pred = best_model.predict(x_test)

                # Model evaluation
                train_score = r2_score(y_train, y_train_pred)
                test_score = r2_score(y_test, y_test_pred)

                report[model_name] = {'score': test_score, 'model': gs}

                logger.info(f"{model_name}: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")

                # Overfitting detection
                if train_score - test_score > 0.1:
                    logger.warning(
                        f"{model_name} may be overfitting! (Train R²: {train_score:.4f}, Test R²: {test_score:.4f})"
                    )

            except NotFittedError as e:
                logger.error(f"Model {model_name} fitting error: {str(e)}")
                continue  # Skip this model

        return report

    except Exception as e:
        logger.error(f"Error in evaluating models: {str(e)}")
        raise CustomException("Model evaluation failed!", cause=e)
