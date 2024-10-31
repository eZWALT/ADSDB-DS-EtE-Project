import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import shutil
from src.file_manager import FileManager

class TestFileManager(unittest.TestCase):

    @patch("pathlib.Path.exists", new_callable=MagicMock, return_value=False)
    @patch("pathlib.Path.mkdir")
    def test_new_data_source_created(self, mock_mkdir, mock_exists):
        manager = FileManager()
        result = manager.newDataSource("test_source")
        mock_mkdir.assert_called_once()
        self.assertTrue(result)

    @patch("pathlib.Path.exists", new_callable=MagicMock, return_value=True)
    def test_new_data_source_already_exists(self, mock_exists):
        manager = FileManager()
        result = manager.newDataSource("test_source")
        self.assertFalse(result)


    @patch("shutil.copy")
    @patch("pathlib.Path.exists", new_callable=MagicMock, return_value=False)
    def test_persist_file_missing_temp_file(self, mock_exists, mock_copy):
        manager = FileManager()
        result = manager.persistFile("missing_file.csv", "test_source")
        mock_copy.assert_not_called()
        self.assertFalse(result)

    @patch("shutil.copy")
    @patch("pathlib.Path.exists", new_callable=MagicMock, side_effect=[True, False])
    def test_persist_file_missing_data_source_folder(self, mock_exists, mock_copy):
        manager = FileManager()
        result = manager.persistFile("test_file_2023.csv", "missing_source")
        mock_copy.assert_not_called()
        self.assertFalse(result)

    @patch("shutil.copy")
    @patch("pathlib.Path.exists", new_callable=MagicMock, side_effect=[True, True, True])
    def test_persist_file_already_exists_in_persistent(self, mock_exists, mock_copy):
        manager = FileManager()
        result = manager.persistFile("test_file_2023.csv", "test_source")
        mock_copy.assert_not_called()
        self.assertFalse(result)

    @patch("pathlib.Path.exists", new_callable=MagicMock, return_value=False)
    def test_empty_persistent_landing_zone_missing_path(self, mock_exists):
        manager = FileManager()
        result = manager.emptyPersistentLandingZone()
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()
