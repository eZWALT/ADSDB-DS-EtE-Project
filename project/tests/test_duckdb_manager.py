import pytest
from unittest.mock import MagicMock, patch
from src.duckdb_manager import DuckDBManager  # Adjust the import as per your project structure

@pytest.fixture
def duckdb_manager():
    """Fixture to create an instance of DuckDBManager for testing."""
    return DuckDBManager()

def test_initialization(duckdb_manager):
    """Test that DuckDBManager initializes paths correctly."""
    assert duckdb_manager.base_path.is_absolute()
    assert duckdb_manager.landing_persistent_path.exists()
    assert duckdb_manager.formatted_zone_path.exists()
    assert duckdb_manager.trusted_zone_path.exists()
    assert duckdb_manager.exploitation_zone_path.exists()

@patch('src.data_management.duckdb.connect')  # Adjust import as needed
def test_set_up_duck_db(mock_connect, duckdb_manager):
    """Test connection setup for DuckDB database."""
    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection
    con = duckdb_manager.set_up_duck_db()
    assert con == mock_connection
    mock_connect.assert_called_once_with(str(duckdb_manager.db_path))

@patch('src.data_management.duckdb.connect')
@patch('src.data_management.duckdb.DuckDBPyConnection.execute')
def test_list_tables(mock_execute, mock_connect, duckdb_manager):
    """Test listing tables from DuckDB database."""
    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection
    mock_execute.return_value.fetchall.return_value = [('table1',), ('table2',)]
    
    tables = duckdb_manager.list_tables(mock_connection)
    assert tables == ['table1', 'table2']
    mock_execute.assert_called_once_with("SHOW TABLES")

@patch('src.data_management.duckdb.connect')
@patch('src.data_management.duckdb.DuckDBPyConnection.execute')
def test_create_tables_from_csv(mock_execute, mock_connect, duckdb_manager):
    """Test creating tables from CSV files."""
    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection
    mock_execute.return_value.fetchall.return_value = []
    
    # Create a mock for the glob function
    with patch('pathlib.Path.glob', return_value=[Path("mock_path/file.csv")]):
        duckdb_manager.create_tables_from_csv(mock_connection)
        
    mock_execute.assert_called_once_with(
        "CREATE TABLE file AS SELECT * FROM read_csv_auto('mock_path/file.csv')"
    )

@patch('src.data_management.duckdb.connect')
@patch('src.data_management.duckdb.DuckDBPyConnection.execute')
def test_delete_all_tables(mock_execute, mock_connect, duckdb_manager):
    """Test deletion of all tables in DuckDB database."""
    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection
    mock_execute.return_value.fetchall.return_value = [('table1',), ('table2',)]
    
    duckdb_manager.delete_all_tables(mock_connection)
    
    # Check if drop table was called for each table
    calls = [mock_execute.call(f"DROP TABLE IF EXISTS {table[0]}") for table in mock_execute.return_value.fetchall.return_value]
    mock_execute.assert_has_calls(calls)
    assert len(calls) == 2  # Ensure that it tried to drop both tables

@patch('pathlib.Path.unlink')
def test_delete_database(mock_unlink, duckdb_manager):
    """Test deletion of the DuckDB database file."""
    duckdb_manager.delete_database()
    mock_unlink.assert_called_once_with()

@patch('src.data_management.duckdb.connect')
def test_unify_tables_by_dataset(mock_connect, duckdb_manager):
    """Test unifying tables by dataset into the trusted database."""
    mock_connection = MagicMock()
    mock_connect.side_effect = [mock_connection, mock_connection]  # Use the same mock for both connections
    mock_connection.execute.return_value.fetchall.return_value = [('table_2020',), ('table_2021',)]

    with patch('pandas.concat', return_value=MagicMock()) as mock_concat:
        duckdb_manager.unify_tables_by_dataset()
        assert mock_concat.called  # Ensure pandas.concat was called

@patch('src.data_management.duckdb.connect')
@patch('src.data_management.duckdb.DuckDBPyConnection.execute')
def test_delete_all_trusted_tables(mock_execute, mock_connect, duckdb_manager):
    """Test deletion of all tables in the trusted DuckDB database."""
    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection
    mock_execute.return_value.fetchall.return_value = [('table1',), ('table2',)]
    
    duckdb_manager.delete_all_trusted_tables()
    
    calls = [mock_execute.call(f"DROP TABLE IF EXISTS {table[0]}") for table in mock_execute.return_value.fetchall.return_value]
    mock_execute.assert_has_calls(calls)
    assert len(calls) == 2  # Ensure that it tried to drop both tables

@patch('src.data_management.duckdb.connect')
@patch('src.data_management.duckdb.DuckDBPyConnection.execute')
def test_delete_all_exploitation_tables(mock_execute, mock_connect, duckdb_manager):
    """Test deletion of all tables in the exploitation zone DuckDB database."""
    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection
    mock_execute.return_value.fetchall.return_value = [('table1',), ('table2',)]
    
    duckdb_manager.delete_all_exploitation_tables()
    
    calls = [mock_execute.call(f"DROP TABLE IF EXISTS {table[0]}") for table in mock_execute.return_value.fetchall.return_value]
    mock_execute.assert_has_calls(calls)
    assert len(calls) == 2  # Ensure that it tried to drop both tables
