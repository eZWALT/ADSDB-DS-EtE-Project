import pytest
import shutil
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.file_manager import FileManager  # Adjust the import as per your project structure

@pytest.fixture
def file_manager():
    """Fixture to create an instance of FileManager for testing."""
    return FileManager()

def test_new_data_source(file_manager):
    """Test creating a new data source folder."""
    with patch('pathlib.Path.mkdir') as mock_mkdir, patch('pathlib.Path.exists', return_value=False):
        result = file_manager.newDataSource('new_source')
        mock_mkdir.assert_called_once()
        assert result is True

    with patch('pathlib.Path.exists', return_value=True):
        result = file_manager.newDataSource('existing_source')
        assert result is False  # Should return False if the folder exists

@patch('shutil.copy')
@patch('pathlib.Path.exists')
@patch('pathlib.Path.mkdir')
@patch('pathlib.Path.iterdir', return_value=[])
def test_persist_file(mock_iterdir, mock_mkdir, mock_exists, mock_copy, file_manager):
    """Test persisting a file from the temporal landing zone."""
    # Set up the mock for exists to simulate the presence of the file
    mock_exists.side_effect = [True, False]  # temp file exists, data source folder does not

    result = file_manager.persistFile('test_2020.csv', 'data_source')
    assert result is False  # Data source folder does not exist

    mock_exists.side_effect = [True, True]  # Now both exist
    mock_copy.return_value = None  # Simulate successful copy

    # Test successful persist
    result = file_manager.persistFile('test_2020.csv', 'data_source')
    assert result is True

    # Test when the persistent file already exists
    mock_exists.side_effect = [True, True, True]  # Temp file exists, data source folder exists, persistent file exists
    result = file_manager.persistFile('test_2020.csv', 'data_source')
    assert result is False  # Should return False if persistent file exists

    # Test when the temp file does not exist
    mock_exists.side_effect = [False]  # Temp file does not exist
    result = file_manager.persistFile('test_2020.csv', 'data_source')
    assert result is False  # Should return False if temp file does not exist

@patch('shutil.rmtree')
@patch('pathlib.Path.iterdir', return_value=[])
def test_empty_temporal_landing_zone(mock_iterdir, mock_rmtree, file_manager):
    """Test emptying the temporal landing zone."""
    with patch('pathlib.Path.exists', return_value=True):
        result = file_manager.emptyTemporalLandingZone()
        assert result is True  # Should return True if it exists and empties successfully

    with patch('pathlib.Path.exists', return_value=False):
        result = file_manager.emptyTemporalLandingZone()
        assert result is False  # Should return False if it doesn't exist

@patch('shutil.rmtree')
@patch('pathlib.Path.iterdir', return_value=[])
def test_empty_persistent_landing_zone(mock_iterdir, mock_rmtree, file_manager):
    """Test emptying the persistent landing zone."""
    with patch('pathlib.Path.exists', return_value=True):
        result = file_manager.emptyPersistentLandingZone()
        assert result is True  # Should return True if it exists and empties successfully

    with patch('pathlib.Path.exists', return_value=False):
        result = file_manager.emptyPersistentLandingZone()
        assert result is False  # Should return False if it doesn't exist

@patch('shutil.rmtree')
@patch('pathlib.Path.iterdir', return_value=[Path('landing_zone.ipynb')])
def test_empty_persistent_landing_zone_skips_ipynb(mock_iterdir, mock_rmtree, file_manager):
    """Test that emptying the persistent landing zone skips the landing_zone.ipynb file."""
    with patch('pathlib.Path.exists', return_value=True):
        result = file_manager.emptyPersistentLandingZone()
        mock_rmtree.assert_not_called()  # Should not try to delete the .ipynb file
        assert result is True
