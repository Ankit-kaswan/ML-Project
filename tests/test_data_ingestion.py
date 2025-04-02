import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.components.data_ingestion import DataIngestion
from src.exception import CustomException


@pytest.fixture
def mock_config(tmpdir):
    """Creates a temporary config object for testing."""
    class MockConfig:
        base_dir = tmpdir
        dataset_file = os.path.join(tmpdir, "dataset.csv")
        train_data_path = os.path.join(tmpdir, "train.csv")
        test_data_path = os.path.join(tmpdir, "test.csv")
        raw_data_path = os.path.join(tmpdir, "raw.csv")

    return MockConfig()


@patch("src.configuration.data_ingestion_config.DataIngestionConfig", autospec=True)
def test_data_ingestion_success(mock_config_class, mock_config):
    """Test successful data ingestion."""
    mock_config_class.return_value = mock_config  # Inject mock config
    data_ingestion = DataIngestion()
    data_ingestion.ingestion_config = mock_config  # Override the instance inside

    # Create a sample dataset
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    df.to_csv(mock_config.dataset_file, index=False)

    train_path, test_path = data_ingestion.initiate_data_ingestion()

    # Check if output files are created
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)

    # Verify train-test split
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    assert len(train_df) == 4  # 80% of 5 rows
    assert len(test_df) == 1  # 20% of 5 rows


@patch("src.configuration.data_ingestion_config.DataIngestionConfig", autospec=True)
def test_data_ingestion_file_not_found(mock_config_class, mock_config):
    """Test if CustomException is raised when dataset file is missing."""
    mock_config_class.return_value = mock_config
    data_ingestion = DataIngestion()
    data_ingestion.ingestion_config = mock_config  # Override the instance inside

    with pytest.raises(CustomException) as excinfo:
        data_ingestion.initiate_data_ingestion()

    assert "Dataset file not found" in str(excinfo.value)


@patch("src.components.data_ingestion.pd.read_csv")
@patch("src.components.data_ingestion.os.makedirs")
@patch("src.components.data_ingestion.pd.DataFrame.to_csv")
@patch("src.configuration.data_ingestion_config.DataIngestionConfig", autospec=True)
def test_data_ingestion_mocked(mock_config_class, mock_to_csv, mock_makedirs, mock_read_csv, mock_config):
    """Test ingestion process with mocked operations to avoid file system access."""
    mock_config_class.return_value = mock_config
    data_ingestion = DataIngestion()
    data_ingestion.ingestion_config = mock_config  # Override the instance inside
    mock_read_csv.return_value = pd.DataFrame({"A": [1, 3, 5], "B": [2, 4, 6]})

    # # ✅ Create an empty dataset file (important!)
    with open(mock_config.dataset_file, "w") as f:
        f.write("A,B\n1,2\n3,4\n5,6")  # Sample data

    train_path, test_path = data_ingestion.initiate_data_ingestion()

    assert train_path.split('/')[-1] == "train.csv"
    assert test_path.split('/')[-1] == "test.csv"

    # Verify CSV saving was called correctly
    assert mock_to_csv.call_count == 3  # raw, train, test
    mock_makedirs.assert_called_once()  # Ensure directory is created
    # ✅ Mock the config class and return the object

