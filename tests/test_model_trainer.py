import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.components.model_trainer import ModelTrainer


@pytest.fixture
def mock_train_test_data():
    """Mock training and test datasets as NumPy arrays."""
    train_array = np.array([
        [1, 2, 3, 50],
        [4, 5, 6, 60],
        [7, 8, 9, 70]
    ])

    test_array = np.array([
        [10, 11, 12, 80],
        [13, 14, 15, 90]
    ])
    return train_array, test_array


@patch("src.utils.evaluate_models")
@patch("src.utils.save_object")
def test_initiate_model_trainer(mock_save_object, mock_evaluate_models, mock_train_test_data):
    """Test the model training function with mocked dependencies."""
    train_array, test_array = mock_train_test_data

    # Mock evaluation results
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([78, 88])
    mock_evaluate_models.return_value = {
        "Random Forest": {"model": mock_model, "score": 0.85}
    }

    model_trainer = ModelTrainer()
    r2_score = model_trainer.initiate_model_trainer(train_array, test_array)

    # Assertions
    assert isinstance(r2_score, float)
    assert r2_score > 0  # Ensure R² score is positive
    mock_save_object.assert_called_once()
    mock_model.predict.assert_called_once()