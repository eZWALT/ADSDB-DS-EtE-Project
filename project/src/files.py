import os
from datetime import datetime
import glob
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import loguru
import pandas as pd
import duckdb


##
## Definition of Files and FileManager for ease of data manipulation
##

@dataclass
class FileInfo:
    filename: str
    filepath: str
    df: Optional[pd.DataFrame] = field(default=None)
    source: Optional[str] = field(default=None)
    sub_source: Optional[str] = field(default=None)
    year: Optional[int] = field(default=None)

    def __repr__(self):
        return (f"FileInfo(filename='{self.filename}', filepath='{self.filepath}', "
                f"df_shape={self.df.shape if self.df is not None else None}, "
                f"source='{self.source}', sub_source='{self.sub_source}', year={self.year})")


"""
  CRUD operations defined for a set of files that are stored in multiple nested directories
  The manager provides abstractions for handling files (Even though doesn't follow RAII priniciples!!!)

"""
@dataclass
class FileInfoManager:
    directory: str
    files: Dict[str, FileInfo] = field(default_factory=dict)

    def __post_init__(self):
        self.files = self._load_csv_files(self.directory)

    """
    Recursively load all CSV files from the directory and create FileInfo objects.
    """
    def _load_csv_files(self, directory: str) -> Dict[str, FileInfo]:
        csv_file_paths = glob.glob(os.path.join(directory, '**', '*.csv'), recursive=True)
        file_info_dict = {}

        for path in csv_file_paths:
            filename = os.path.basename(path)
            file_info = self._create_file_info(filename, path)
            file_info_dict[filename] = file_info

        loguru.logger.info(f"Loaded {len(file_info_dict)} files from directory {directory}.")
        return file_info_dict

    """
    Extracts source, sub-source, and year from the filename using regex.
    Example pattern: 'source_subsource_year.csv'
    """
    def _create_file_info(self, filename: str, filepath: str, df: Optional[pd.DataFrame]=None) -> FileInfo:
        match = re.match(r'([^_]+)_([^_]+)_(\d{4})', filename)
        if match:
            source = match.group(1)
            sub_source = match.group(2)
            year = int(match.group(3))
        else:
            source = None
            sub_source = None
            year = None

        return FileInfo(filename=filename, filepath=filepath, source=source, sub_source=sub_source, year=year,df=df)

    """
    Loads a DataFrame from a CSV file if not already loaded. Returns None if file doesn't exist.
    """
    def load_dataframe(self, filename: str) -> Optional[pd.DataFrame]:
        file_info = self.files.get(filename)
        if file_info:
            if file_info.df is None:
                try:
                    file_info.df = pd.read_csv(file_info.filepath)
                    loguru.logger.success(f"DataFrame loaded for file: {filename}")
                except Exception as e:
                    loguru.logger.error(f"Failed to load DataFrame for {filename}: {e}")
                    return None
            return file_info.df
        else:
            loguru.logger.error(f"File '{filename}' not found in FileInfoManager.")
            return None

    """
      Loads all dataframes from its paths
    """
    def load_all_dataframes(self):
      for filename in self.files.keys():
        self.load_dataframe(filename)

    """
    Retrieves the FileInfo object for a given filename.
    """
    def get_file_info(self, filename: str) -> Optional[FileInfo]:
        return self.files.get(filename)

    """
    Adds a new FileInfo object to the manager.
    """
    def add_file(self, file_info: FileInfo):
        self.files[file_info.filename] = file_info

    """
    Adds a new FileInfo object to the manager by filepath.
    """
    def add_file_by_path(self, filepath: str, df: Optional[pd.DataFrame] = None):
        filename = os.path.basename(filepath)
        self.files[filename] = self._create_file_info(filename, filepath, df)

    """
    Removes a file from the manager by filename.
    """
    def remove_file(self, filename: str):
        if filename in self.files:
            del self.files[filename]
            loguru.logger.success(f"File '{filename}' removed from FileInfoManager.")
        else:
            loguru.logger.warning(f"Tried to remove '{filename}', but it was not found.")

    """
    Updates the file paths of each FileInfo in the FileInfoManager using the base path,
    source, sub-source, and year.

    Filepath format: base_path/source/sub_source/year.csv
    """
    def update_filepaths(self, base_path: str):
      for filename, file_info in self.files.items():
          if file_info.source and file_info.sub_source and file_info.year:
              # Construct the new file path
              new_filepath = os.path.join(base_path, file_info.source, file_info.sub_source, file_info.filename)

              # Update the FileInfo with the new path
              file_info.filepath = new_filepath
              loguru.logger.success(f"Updated {filename} -> {new_filepath}")
          else:
              loguru.logger.error(f"Skipping {filename}: source/sub_source/year information is missing")




    """
    Saves all loaded DataFrames to the specified base path.
    The directory structure is 'base_path/source/sub_source/year.csv'.
    If add_timestamp is True, appends a timestamp to the filenames.
    """
    def save_all_files(self, base_path: str, add_timestamp: bool = False):
        for file_info in self.files.values():
            if file_info.df is not None:
                # Create structured directory
                save_directory = os.path.join(base_path, file_info.source, file_info.sub_source)
                os.makedirs(save_directory, exist_ok=True)

                # Construct the filename with optional timestamp
                name, ext = os.path.splitext(file_info.filename)
                if add_timestamp:
                    timestamp = datetime.now().strftime('%Y%m%d')
                    name = f"{name}_{timestamp}"
                save_path = os.path.join(save_directory, f"{name}{ext}")

                try:
                    file_info.df.to_csv(save_path, index=False)
                    loguru.logger.success(f"File '{file_info.filename}' saved as '{name}' in '{save_directory}'")
                except Exception as e:
                    loguru.logger.error(f"Error saving '{file_info.filename}' to '{save_path}': {e}")
            else:
                loguru.logger.warning(f"File '{file_info.filename}' has no DataFrame loaded, skipping save.")

    """
    Saves a single file's DataFrame to the specified base path.
    The file will be saved as 'base_path/source/sub_source/year.csv'.
    If add_timestamp is True, appends a timestamp to the filename.
    """
    def save_file(self, filename: str, base_path: str, add_timestamp: bool = False):

        file_info = self.files.get(filename)
        if file_info and file_info.df is not None:
            save_directory = os.path.join(base_path, file_info.source, file_info.sub_source)
            os.makedirs(save_directory, exist_ok=True)

            # Construct the filename with optional timestamp
            name = f"{file_info.year}"  # Default: year.csv
            if add_timestamp:
                timestamp = datetime.now().strftime('%Y%m%d')
                name = f"{name}_{timestamp}.csv"
            else:
                name = f"{name}.csv"

            save_path = os.path.join(save_directory, name)

            try:
                file_info.df.to_csv(save_path, index=False)
                loguru.logger.success(f"File '{filename}' saved as '{name}' in '{save_directory}'")
            except Exception as e:
                loguru.logger.error(f"Error saving '{filename}' to '{save_path}': {e}")
        else:
            loguru.logger.warning(f"File '{filename}' not found or has no DataFrame loaded.")

    """
    Returns a list of all filenames
    """
    def list_files(self) -> List[str]:
        return list(self.files.keys())

    """
    Creates DuckDB tables for each CSV file in the FileInfoManager.
    The DuckDB connection is passed as an argument.
    """
    def create_duckdb_tables(self, connection: duckdb.DuckDBPyConnection):
        for filename, file_info in self.files.items():
            table_name = os.path.splitext(file_info.filename)[0]
            try:
                connection.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} AS
                    SELECT * FROM read_csv_auto('{file_info.filepath}')
                """)
                loguru.logger.success(f"Table {table_name} created successfully.")
            except Exception as e:
                loguru.logger.error(f"Error creating DuckDB table {table_name}: {e}")

    def create_or_update_duckdb_tables_from_dataframes(self, connection: duckdb.DuckDBPyConnection):
        for filename, file_info in self.files.items():
            table_name = os.path.splitext(file_info.filename)[0]
            if file_info.df is not None:
                try:
                    # Register the DataFrame as a temporary table in DuckDB
                    connection.register(f"temp_{table_name}", file_info.df)

                    # Check if the table exists
                    table_exists = connection.execute(
                        f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
                    ).fetchone()[0] > 0

                    if table_exists:
                        # If the table exists, truncate it to avoid duplications
                        connection.execute(f"TRUNCATE TABLE {table_name}")
                    else:
                        # If the table does not exist, create it
                        connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_{table_name}")

                    # Insert data from the temporary table into the main table
                    connection.execute(f"INSERT INTO {table_name} SELECT * FROM temp_{table_name}")
                    loguru.logger.success(f"Data inserted successfully into table {table_name} from in-memory DataFrame.")

                    # Unregister the temporary table
                    connection.unregister(f"temp_{table_name}")

                except Exception as e:
                    loguru.logger.error(f"Error creating or updating DuckDB table {table_name} from DataFrame: {e}")
            else:
                loguru.logger.warning(f"No DataFrame loaded for {filename}; skipping table creation.")

    """
    Retrieves all tables from DuckDB and returns them as a dictionary of DataFrames.
    The dictionary keys are table names, and values are DataFrames.
    """
    def retrieve_all_from_duckdb(self, connection: duckdb.DuckDBPyConnection) -> Dict[str, pd.DataFrame]:
        try:
            tables = connection.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]

            dataframes = {}
            # Loop through each table and retrieve it as a DataFrame
            for table_name in table_names:
                df = connection.execute(f"SELECT * FROM {table_name}").fetchdf()
                dataframes[table_name] = df
                loguru.logger.success(f"Data retrieved from DuckDB table '{table_name}' successfully.")

            return dataframes
        except Exception as e:
            loguru.logger.error(f"Failed to retrieve tables from DuckDB: {e}")
            return {}

    """
    Saves an in-memory DataFrame from FileInfoManager to a DuckDB table.
    The DuckDB connection is passed as an argument.
    """
    def save_df_to_duckdb(self, connection: duckdb.DuckDBPyConnection, filename: str, table_name: str):
        file_info = self.get_file_info(filename)
        if file_info and file_info.df is not None:
            try:
                connection.register('df_temp', file_info.df)
                connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_temp")
                loguru.logger.success(f"DataFrame saved to DuckDB table {table_name} successfully.")
            except Exception as e:
                loguru.logger.error(f"Failed to save DataFrame to DuckDB table {table_name}: {e}")
        else:
            loguru.logger.error(f"No DataFrame found for {filename}.")

    """
    Searches files by optional metadata such as source, sub_source, and year.
    If a parameter is None it won't be used, therefore if all parameters are None
    it will return all files.
    """
    def search_files(self, source: Optional[str] = None, sub_source: Optional[str] = None, year: Optional[int] = None) -> List[FileInfo]:
        results = []
        for file_info in self.files.values():
            match = True
            if source is not None and file_info.source != source:
                match = False
            if sub_source is not None and file_info.sub_source != sub_source:
                match = False
            if year is not None and file_info.year != year:
                match = False
            if match:
                results.append(file_info)
        return results

    ### UPDATE FUNCTIONS

    def update_dataframe(self, filename: str, new_df: pd.DataFrame):
      file_info = self.files.get(filename)
      if file_info:
          file_info.df = new_df
      else:
          loguru.logger.error(f"File '{filename}' not found in FileInfoManager.")

    def update_file_info(self, filename: str, source: Optional[str] = None, sub_source: Optional[str] = None, year: Optional[int] = None):
      file_info = self.files.get(filename)
      if file_info:
          if source is not None:
              file_info.source = source
          if sub_source is not None:
              file_info.sub_source = sub_source
          if year is not None:
              file_info.year = year
      else:
          loguru.logger.error(f"File '{filename}' not found in FileInfoManager.")

