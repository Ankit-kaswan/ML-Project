import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.configuration.data_transformation_config import DataTransformationConfig
from src.exception import CustomException
from src.utils import save_object
from src.logger import Logger


# Initialize the custom logger
logger = Logger.get_logger()


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a preprocessing pipeline for numerical and categorical features.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # Pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logger.info(f"Defined preprocessing pipelines for numerical & categorical features.")

            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            logger.error(f"Error creating data transformer: {str(e)}")
            raise CustomException("Failed to create preprocessing pipeline", cause=e)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Reads train and test data, applies transformations, and saves the preprocessing object.

        Returns:
            - Transformed train data as a NumPy array
            - Transformed test data as a NumPy array
            - Path to the saved preprocessing object
        """
        try:
            # Ensure train & test files exist
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Train file not found: {train_path}")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Test file not found: {test_path}")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info(f"Successfully loaded train ({train_path}) and test ({test_path}) datasets.")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            # Separate input & target features
            x_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]

            x_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]

            logger.info(f"Applying preprocessing transformations...")

            # Apply transformations
            x_train_transformed = preprocessing_obj.fit_transform(x_train)
            x_test_transformed = preprocessing_obj.transform(x_test)

            # Concatenate transformed features with target column
            train_arr = np.c_[x_train_transformed, y_train.to_numpy()]
            test_arr = np.c_[x_test_transformed, y_test.to_numpy()]

            logger.info(f"Preprocessing completed. Saving preprocessor object...")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except FileNotFoundError as fnf_error:
            logger.error(f"File error: {str(fnf_error)}")
            raise CustomException(str(fnf_error), sys)
        except Exception as e:
            logger.error(f"Data transformation failed: {str(e)}")
            raise CustomException("Data transformation failed!", cause=e)


if __name__ == '__main__':
    from src.components.data_ingestion import DataIngestion

    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    output = data_transformation.initiate_data_transformation(train_path=train_path, test_path=test_path)
    print(output)
