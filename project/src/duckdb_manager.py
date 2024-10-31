import duckdb
from pathlib import Path
import re
import pandas as pd
import glob
from loguru import logger

class DuckDBManager:
    """
    A class to manage a DuckDB database within a specified project directory structure.
    """
    def __init__(self):
        """
        Initializes the DuckDBManager with a specified base path.

        Args:
            path (str): Path of the main project directory.

        Raises:
            ValueError: If the path does not exist or is not a directory.
        """

        self.base_path = Path.cwd()
        self.landing_persistent_path = self.base_path / "src" / "data_management" / "landing_zone" / "persistent"
        self.formatted_zone_path = self.base_path / "src" / "data_management" / "formatted_zone"
        self.db_path = self.formatted_zone_path / "formatted_zone.duckdb"
        self.trusted_zone_path = self.base_path / "src" / "data_management" / "trusted_zone"
        self.trusted_db_path = self.trusted_zone_path / "trusted_zone.duckdb"
        self.exploitation_zone_path = self.base_path / "src" / "data_management" / "exploitation_zone"
        self.exploitation_db_path = self.exploitation_zone_path / "exploitation_zone.duckdb"

        # Create necessary directories
        self.landing_persistent_path.mkdir(parents=True, exist_ok=True)
        self.formatted_zone_path.mkdir(parents=True, exist_ok=True)
        self.trusted_zone_path.mkdir(parents=True, exist_ok=True)
        self.exploitation_zone_path.mkdir(parents=True, exist_ok=True)

    def set_up_duck_db(self):
        """
        Connects to an existing DuckDB database in the formatted zone directory
        or creates one if it does not exist.

        Returns:
            con (duckdb.DuckDBPyConnection): Connection object for the DuckDB database.
        """
        con = duckdb.connect(str(self.db_path))
        logger.info(f"Connected to DuckDB database at '{self.db_path}'.")
        return con

    def list_tables(self, con):
        """
        Lists all tables in the DuckDB database.

        Args:
            con (duckdb.DuckDBPyConnection): Connection object to the DuckDB database.

        Returns:
            list: List of table names in the database.
        """
        try:
            tables = con.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]
            return table_names
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            raise

    def create_tables_from_csv(self, con):
        """
        Creates tables in the DuckDB database for each CSV file found in the data source folders.
        Raises an error if a table already exists with the same name.

        Args:
            con (duckdb.DuckDBPyConnection): Connection object to the DuckDB database.

        Raises:
            FileNotFoundError: If no CSV files are found in the landing zone.
            RuntimeError: If a table already exists with the same name.
        """
        try:
            # Get all existing tables to prevent overwriting
            existing_tables = self.list_tables(con)

            data_source_folders = list(self.landing_persistent_path.glob('*'))

            if not data_source_folders:
                raise FileNotFoundError("No data source folders found in the persistent landing zone.")

            for data_source_folder in data_source_folders:
                if not data_source_folder.is_dir():
                    logger.warning(f"{data_source_folder} is not a directory.")
                    continue

                csv_files = list(data_source_folder.glob('*.csv'))
                if not csv_files:
                    logger.warning(f"No CSV files found in data source folder '{data_source_folder}'.")
                    continue

                for file_path in csv_files:
                    table_name = file_path.stem
                    if table_name in existing_tables:
                        logger.warning(f"Table '{table_name}' already exists in the database.")
                        continue

                    con.execute(f"""
                        CREATE TABLE {table_name} AS
                        SELECT * FROM read_csv_auto('{file_path}')
                    """)
                    logger.info(f"Table '{table_name}' created successfully from '{file_path}'.")

        except FileNotFoundError as fnf_error:
            logger.error(fnf_error)
            raise
        except Exception as e:
            logger.error(f"Failed to create tables from CSV files: {e}")
            raise

    def delete_all_tables(self, con):
        """
        Deletes all tables in the DuckDB database connection.

        Args:
            con (duckdb.DuckDBPyConnection): Connection object to the DuckDB database.
        """
        try:
            tables = con.execute("SHOW TABLES").fetchall()
            for table in tables:
                table_name = table[0]
                con.execute(f"DROP TABLE IF EXISTS {table_name}")
                logger.info(f"Table '{table_name}' deleted successfully.")
            logger.info("All tables deleted successfully.")
        except Exception as e:
            logger.error(f"Failed to delete all tables: {e}")
            raise

    def delete_database(self):
        """
        Deletes the entire DuckDB database file from the formatted zone.

        Raises:
            FileNotFoundError: If the database file does not exist.
        """
        try:
            if self.db_path.exists():
                self.db_path.unlink()
                logger.info(f"DuckDB database at '{self.db_path}' deleted successfully.")
            else:
                raise FileNotFoundError(f"DuckDB database at '{self.db_path}' does not exist.")
        except Exception as e:
            logger.error(f"Failed to delete the DuckDB database: {e}")
            raise

    # TRUSTED ZONE FUNCTIONS --------------------------------------------------------------------

    def set_up_trusted_db(self):
        """
        Sets up the trusted DuckDB database.

        Returns:
            trusted_con (duckdb.DuckDBPyConnection): Connection object for the trusted DuckDB database.
        """
        trusted_con = duckdb.connect(str(self.trusted_db_path))
        logger.info(f"Connected to trusted DuckDB database at '{self.trusted_db_path}'.")
        return trusted_con

    def unify_tables_by_dataset(self):
        """
        Unifies tables from the formatted database into a single table per dataset in the trusted database.
        """
        con = self.set_up_duck_db()  # Connect to the formatted database
        trusted_con = self.set_up_trusted_db()  # Connect to the trusted database

        table_names = self.list_tables(con)
        dataset_tables = {}

        # Group tables by dataset name (ignore year)
        for table_name in table_names:
            dataset_name = re.sub(r'_\d{4}', '', table_name)  # Remove year to get the dataset name
            if dataset_name not in dataset_tables:
                dataset_tables[dataset_name] = []
            dataset_tables[dataset_name].append(table_name)

        # Merge tables for each dataset
        for dataset_name, tables in dataset_tables.items():
            # Combine all tables for the dataset
            dataframes = [con.execute(f"SELECT * FROM {table}").fetchdf() for table in tables]
            combined_df = pd.concat(dataframes, ignore_index=True)

            # Obtain the name of the new table as the data source name
            dataset_name = tables[0].split('_')[0]

            # Save to the trusted database
            trusted_con.execute(f"CREATE TABLE {dataset_name} AS SELECT * FROM combined_df")
            logger.info(f"Unified table '{dataset_name}' created in the trusted zone with data from {len(tables)} years.")

        # Close both connections
        con.close()
        trusted_con.close()

    def delete_all_trusted_tables(self):
        """
        Deletes all tables in the trusted DuckDB database.
        """
        trusted_con = self.set_up_trusted_db()
        try:
            tables = trusted_con.execute("SHOW TABLES").fetchall()
            for table in tables:
                table_name = table[0]
                trusted_con.execute(f"DROP TABLE IF EXISTS {table_name}")
                logger.info(f"Table '{table_name}' deleted from the trusted zone.")
            logger.info("All tables deleted from the trusted zone.")
        finally:
            trusted_con.close()

    # EXPLOITATION ZONE FUNCTIONS ---------------------------------------------

    def set_up_exploitation_db(self):
        """
        Sets up the exploitation zone DuckDB database.

        Returns:
            exploitation_db (duckdb.DuckDBPyConnection): Connection object for the exploitation DuckDB database.
        """
        exploitation_db = duckdb.connect(str(self.exploitation_db_path))
        logger.info(f"Connected to exploitation DuckDB database at '{self.exploitation_db_path}'.")
        return exploitation_db

    def delete_all_exploitation_tables(self):
        """
        Deletes all tables in the exploitation zone DuckDB database.
        """
        exploitation_con = self.set_up_exploitation_db()
        try:
            tables = exploitation_con.execute("SHOW TABLES").fetchall()
            for table in tables:
                table_name = table[0]
                exploitation_con.execute(f"DROP TABLE IF EXISTS {table_name}")
                logger.info(f"Table '{table_name}' deleted from the exploitation zone.")
            logger.info("All tables deleted from the exploitation zone.")
        finally:
            exploitation_con.close()
