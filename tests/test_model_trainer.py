import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException


# @patch("src.utils.evaluate_models")
# @patch("src.utils.save_object")
# def test_initiate_model_trainer(mock_save_object, mock_evaluate_models):
#     """Test the initiate_model_trainer function with mocked dependencies."""
#
#     # Mock training and test arrays
#     train_array = np.array([[1, 2, 3, 4, 5, 100], [6, 7, 8, 9, 10, 200]])
#     test_array = np.array([[11, 12, 13, 14, 15, 300], [16, 17, 18, 19, 20, 400]])
#
#     # Mock a high-performing model
#     mock_model = MagicMock()
#     mock_model.predict.return_value = np.array([300, 400])  # Simulated predictions
#
#     # Mock evaluate_models function to return a high-score model
#     mock_evaluate_models.return_value = {
#         "Random Forest": {"model": mock_model, "score": 0.85}
#     }
#
#     model_trainer = ModelTrainer()
#     r2_score = model_trainer.initiate_model_trainer(train_array, test_array)
#
#     # **Assertions**
#     assert isinstance(r2_score, float)
#     assert r2_score > 0.6  # Ensure a reasonable model performance
#     mock_save_object.assert_called_once()  # Ensure the model was saved
#     mock_model.predict.assert_called()  # Ensure prediction was performed


@patch("src.utils.evaluate_models", return_value={})  # Simulate no valid models found
@patch("src.utils.save_object")
def test_model_trainer_no_valid_model(mock_save_object, mock_evaluate_models):
    """Test ModelTrainer when no valid model is found, ensuring the exact exception message is matched."""

    # Simulated train/test data (minimum valid)
    train_array = np.array([[1.0, 2.0, 50.0], [2.0, 3.0, 60.0]])  # 2 training samples
    test_array = np.array([[3.0, 4.0, 70.0]])  # 1 test sample

    model_trainer = ModelTrainer()
    model_trainer.models = {}

    # Expect `CustomException` with the exact message
    with pytest.raises(CustomException) as exc_info:
        model_trainer.initiate_model_trainer(train_array, test_array)

    # ✅ Match the exact exception message
    assert "Model evaluation failed. No valid models found." in str(exc_info.value)

    # ✅ Ensure `save_object` was NEVER called
    mock_save_object.assert_not_called()