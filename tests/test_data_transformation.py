import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.components.data_transformation import DataTransformation


@pytest.fixture
def mock_data():
    """Mock train and test data as Pandas DataFrames."""
    train_data = pd.DataFrame({
        "writing_score": [70, 80, 90],
        "reading_score": [65, 75, 85],
        "gender": ["male", "female", "male"],
        "race_ethnicity": ["group A", "group B", "group A"],
        "parental_level_of_education": ["bachelor", "master", "high school"],
        "lunch": ["standard", "free/reduced", "standard"],
        "test_preparation_course": ["completed", "none", "completed"],
        "math_score": [75, 85, 95]  # Target variable
    })

    test_data = train_data.copy()

    return train_data, test_data


@patch("src.components.data_transformation.pd.read_csv")
@patch("src.components.data_transformation.save_object")
@patch("src.components.data_transformation.DataTransformation.get_data_transformer_object")
@patch("src.components.data_transformation.os.path.exists")
def test_data_transformation(mock_exists, mock_get_data_transformer_object, mock_save_object, mock_read_csv, mock_data):
    """Test DataTransformation with mocked preprocessing and file operations."""

    train_data, test_data = mock_data

    # Mock file existence check to always return True
    mock_exists.return_value = True

    # Mock file reading
    mock_read_csv.side_effect = [train_data, test_data]

    # Mock transformer
    mock_transformer = MagicMock()
    mock_transformer.fit_transform.return_value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mock_transformer.transform.return_value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mock_get_data_transformer_object.return_value = mock_transformer

    # Initialize class and call function
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
        "/tmp/mock_train.csv", "/tmp/mock_test.csv"
    )

    # Assertions
    assert train_arr.shape == (3, 4)  # 3 samples, 3 transformed + 1 target column
    assert test_arr.shape == (3, 4)   # Same for test data
    assert isinstance(preprocessor_path, str)

    # Verify calls
    mock_read_csv.assert_any_call("/tmp/mock_train.csv")
    mock_read_csv.assert_any_call("/tmp/mock_test.csv")
    mock_transformer.fit_transform.assert_called_once()  # Fit-transform should be called for train
    mock_transformer.transform.assert_called_once()      # Transform should be called for test
    mock_save_object.assert_called_once()               # Preprocessor should be saved
    mock_exists.assert_any_call("/tmp/mock_train.csv")   # Verify that os.path.exists was called
    mock_exists.assert_any_call("/tmp/mock_test.csv")

