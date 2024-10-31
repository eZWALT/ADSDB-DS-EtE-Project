import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from duckdb import DuckDBPyConnection
from src.duckdb_manager import DuckDBManager
import pandas as pd

class TestDuckDBManager(unittest.TestCase):

    @patch("duckdb.connect")
    def setUp(self, mock_connect):
        self.mock_connection = MagicMock(spec=DuckDBPyConnection)
        mock_connect.return_value = self.mock_connection
        self.manager = DuckDBManager()
        
    def test_init_paths_creation(self):
        self.assertTrue(self.manager.landing_persistent_path.exists())
        self.assertTrue(self.manager.formatted_zone_path.exists())
        self.assertTrue(self.manager.trusted_zone_path.exists())
        self.assertTrue(self.manager.exploitation_zone_path.exists())
        

    def test_list_tables_success(self):
        self.mock_connection.execute.return_value.fetchall.return_value = [('table1',), ('table2',)]
        tables = self.manager.list_tables(self.mock_connection)
        self.assertEqual(tables, ['table1', 'table2'])

    def test_list_tables_error(self):
        self.mock_connection.execute.side_effect = Exception("Execution error")
        with self.assertRaises(Exception):
            self.manager.list_tables(self.mock_connection)

    def test_create_tables_from_csv_no_csv_files(self):
        self.manager.landing_persistent_path = Path("/boot")
        with self.assertRaises(FileNotFoundError):
            self.manager.create_tables_from_csv(self.mock_connection)

    def test_delete_all_tables(self):
        self.mock_connection.execute.return_value.fetchall.return_value = [('table1',), ('table2',)]
        self.manager.delete_all_tables(self.mock_connection)
        self.mock_connection.execute.assert_any_call("DROP TABLE IF EXISTS table1")
        self.mock_connection.execute.assert_any_call("DROP TABLE IF EXISTS table2")

    def test_delete_database_success(self):
        self.manager.create_tables_from_csv(self.mock_connection)
        self.manager.set_up_formatted_db()
        with self.assertRaises(FileNotFoundError):
            self.manager.delete_formatted()
            non_existent_db = Path(self.manager.formatted_db_path)  
            #Try to remove it
            non_existent_db.unlink()

    def test_delete_formatted_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.manager.delete_formatted()

    @patch("pandas.concat")
    @patch("src.duckdb_manager.DuckDBManager.list_tables")
    @patch("duckdb.connect")
    def test_unify_tables_by_dataset(self, mock_connect, mock_list_tables, mock_concat):
        mock_connect.side_effect = [self.mock_connection, self.mock_connection]
        mock_list_tables.return_value = ['dataset1_2020', 'dataset1_2021']
        
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_concat.return_value = mock_df
        self.mock_connection.execute.return_value.fetchdf.return_value = mock_df
        
        self.manager.unify_tables_by_dataset()
        mock_concat.assert_called_once()
        self.mock_connection.execute.assert_called_with("CREATE TABLE dataset1 AS SELECT * FROM combined_df")

    @patch("duckdb.connect")
    def test_delete_all_trusted_tables(self, mock_connect):
        mock_connect.return_value = self.mock_connection
        self.mock_connection.execute.return_value.fetchall.return_value = [('table1',), ('table2',)]
        
        self.manager.delete_all_trusted_tables()
        self.mock_connection.execute.assert_any_call("DROP TABLE IF EXISTS table1")
        self.mock_connection.execute.assert_any_call("DROP TABLE IF EXISTS table2")

    @patch("duckdb.connect")
    def test_delete_all_exploitation_tables(self, mock_connect):
        mock_connect.return_value = self.mock_connection
        self.mock_connection.execute.return_value.fetchall.return_value = [('table1',), ('table2',)]
        
        self.manager.delete_all_exploitation_tables()
        self.mock_connection.execute.assert_any_call("DROP TABLE IF EXISTS table1")
        self.mock_connection.execute.assert_any_call("DROP TABLE IF EXISTS table2")

if __name__ == "__main__":
    unittest.main()
