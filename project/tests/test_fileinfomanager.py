import os
import pandas as pd
import pytest
import duckdb
from src.files import FileInfoManager, FileInfo

@pytest.fixture
def setup_file_manager(tmp_path):
    # Setup a temporary directory for testing
    test_dir = tmp_path / "test_files"
    test_dir.mkdir()
    
    # Create sample CSV files
    sample_csv_content = "column1,column2\n1,2\n3,4"
    sample_file = test_dir / "source_subsource_2023.csv"
    
    with open(sample_file, "w") as f:
        f.write(sample_csv_content)

    # Initialize FileInfoManager
    file_manager = FileInfoManager(str(test_dir))
    return file_manager, test_dir

def test_initialization_and_file_loading(setup_file_manager):
    file_manager, _ = setup_file_manager
    assert len(file_manager.files) == 1
    assert "source_subsource_2023.csv" in file_manager.files
    assert file_manager.files["source_subsource_2023.csv"].source == "source"
    assert file_manager.files["source_subsource_2023.csv"].sub_source == "subsource"
    assert file_manager.files["source_subsource_2023.csv"].year == 2023

def test_load_dataframe(setup_file_manager):
    file_manager, _ = setup_file_manager
    df = file_manager.load_dataframe("source_subsource_2023.csv")
    assert df is not None
    assert df.shape == (2, 2)  # 2 rows, 2 columns
    assert df.columns.tolist() == ["column1", "column2"]

def test_add_file(setup_file_manager):
    file_manager, _ = setup_file_manager
    new_file_info = FileInfo(filename="new_file.csv", filepath="/dummy/path/new_file.csv")
    file_manager.add_file(new_file_info)
    assert "new_file.csv" in file_manager.files

def test_remove_file(setup_file_manager):
    file_manager, _ = setup_file_manager
    file_manager.remove_file("source_subsource_2023.csv")
    assert "source_subsource_2023.csv" not in file_manager.files

def test_update_filepaths(setup_file_manager):
    file_manager, _ = setup_file_manager
    file_manager.update_filepaths("/new/base/path")
    updated_filepath = file_manager.files["source_subsource_2023.csv"].filepath
    assert updated_filepath == "/new/base/path/source/subsource/source_subsource_2023.csv"

def test_save_to_duckdb(setup_file_manager):
    file_manager, test_dir = setup_file_manager
    conn = duckdb.connect(database=':memory:')  # Use in-memory database
    df = file_manager.load_dataframe("source_subsource_2023.csv")
    file_manager.update_dataframe("source_subsource_2023.csv", df)
    
    file_manager.save_df_to_duckdb(conn, "source_subsource_2023.csv", "my_table")
    result_df = conn.execute("SELECT * FROM my_table").fetchdf()
    
    assert result_df.shape == (2, 2)  # Should match the original data
    assert result_df["column1"].tolist() == [1, 3]

def test_retrieve_data_from_duckdb(setup_file_manager):
    file_manager, test_dir = setup_file_manager
    conn = duckdb.connect(database=':memory:')  # Use in-memory database
    df = file_manager.load_dataframe("source_subsource_2023.csv")
    file_manager.save_df_to_duckdb(conn, "source_subsource_2023.csv", "my_table")

    retrieved_data = file_manager.retrieve_all_from_duckdb(conn)
    assert "my_table" in retrieved_data
    assert retrieved_data["my_table"].shape == (2, 2)  # Should match the original data

def test_error_handling(setup_file_manager):
    file_manager, _ = setup_file_manager
    result = file_manager.load_dataframe("non_existent_file.csv")
    assert result is None  # Should return None for a non-existent file
