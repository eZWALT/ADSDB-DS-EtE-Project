import duckdb
from pathlib import Path
import re
import pandas as pd
import glob
from loguru import logger

import duckdb
from pathlib import Path
import pandas as pd
from loguru import logger

class DuckDBManager:
    """
    A class to manage a DuckDB database within a specified project directory structure.
    """
    def __init__(self):
        """
        Initializes the DuckDBManager with a specified base path.
        """

        self.base_path = Path.cwd()
        self.landing_persistent_path = self.base_path / "src" / "data_management" / "landing_zone" / "persistent"
        self.formatted_zone_path = self.base_path / "src" / "data_management" / "formatted_zone"
        self.trusted_zone_path = self.base_path / "src" / "data_management" / "trusted_zone"
        self.exploitation_zone_path = self.base_path / "src" / "data_management" / "exploitation_zone"

        self.analytical_sandbox_path = self.base_path / "src" / "analytical" / "analytical_sandbox"
        self.feature_generation_path = self.base_path / "src" / "analytical" / "feature_engineering"  
        self.data_preparation_path = self.base_path / "src" / "analytical" / "feature_engineering" 
        self.labeling_path = self.base_path / "src" / "analytical" / "feature_engineering"
        self.train_test_path = self.base_path / "src" / "analytical" / "feature_engineering"


        # Initialize all database paths for each zone
        self.formatted_db_path = self.formatted_zone_path / "formatted_zone.duckdb"
        self.trusted_db_path = self.trusted_zone_path / "trusted_zone.duckdb"
        self.exploitation_db_path = self.exploitation_zone_path / "exploitation_zone.duckdb"
        
        self.analytical_sandbox_db_path = self.analytical_sandbox_path / "analytical_sandbox.duckdb"
        self.feature_generation_db_path = self.feature_generation_path / "feature_generation.duckdb"
        self.data_preparation_db_path = self.data_preparation_path / "data_preparation.duckdb"
        self.labeling_db_path = self.labeling_path / "labeling.duckdb"
        self.train_test_db_path = self.train_test_path / "train_test.duckdb"

        # Create necessary directories
        self.formatted_zone_path.mkdir(parents=True, exist_ok=True)
        self.trusted_zone_path.mkdir(parents=True, exist_ok=True)
        self.exploitation_zone_path.mkdir(parents=True, exist_ok=True)
        self.analytical_sandbox_path.mkdir(parents=True, exist_ok=True)
        self.feature_generation_path.mkdir(parents=True, exist_ok=True)
        self.data_preparation_path.mkdir(parents=True, exist_ok=True)
        self.labeling_path.mkdir(parents=True, exist_ok=True)
        self.train_test_path.mkdir(parents=True, exist_ok=True)
        
    def _connect_to_db(self, db_path: Path):
        """
        Establishes a connection to a DuckDB database.

        Args:
            db_path (Path): The path to the DuckDB database file.

        Returns:
            duckdb.DuckDBPyConnection: The connection object to the database.
        """
        con = duckdb.connect(str(db_path))
        logger.info(f"Connected to DuckDB database: '{db_path.name}'.")
        return con

    # NEW ZONE SETUP FUNCTIONS -------------------------------------------------------

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

            # Initialize a flag to track if any CSV files were found
            csv_found = False

            for data_source_folder in data_source_folders:
                if not data_source_folder.is_dir():
                    logger.warning(f"{data_source_folder} is not a directory.")
                    continue

                csv_files = list(data_source_folder.glob('*.csv'))
                if not csv_files:
                    logger.warning(f"No CSV files found in data source folder '{data_source_folder}'.")
                    continue

                csv_found = True  # Mark that we've found at least one CSV file

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

            # If no CSV files were found at all, raise an error
            if not csv_found:
                raise FileNotFoundError("No CSV files found in any data source folders.")

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

    def delete_formatted(self):
        """
        Deletes the entire DuckDB database file from the formatted zone.

        Raises:
            FileNotFoundError: If the database file does not exist.
        """
        try:
            if self.formatted_db_path.exists():
                self.formatted_db_path.unlink()
                logger.info(f"DuckDB database at '{self.formatted_db_path}' deleted successfully.")
            else:
                raise FileNotFoundError(f"DuckDB database at '{self.formatted_db_path}' does not exist.")
        except Exception as e:
            logger.error(f"Failed to delete the DuckDB database: {e}")
            raise

    def unify_tables_by_dataset(self):
        """
        Unifies tables from the formatted database into a single table per dataset in the trusted database.
        """
        con = self.set_up_formatted_db()  # Connect to the formatted database
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

    def set_up_formatted_db(self):
        return self._connect_to_db(self.formatted_db_path)
    
    def set_up_trusted_db(self):
        return self._connect_to_db(self.trusted_db_path)
    
    def set_up_exploitation_db(self):
        return self._connect_to_db(self.exploitation_db_path)


    def set_up_analytical_sandbox_db(self):
        return self._connect_to_db(self.analytical_sandbox_db_path)

    def set_up_feature_generation_db(self):
        return self._connect_to_db(self.feature_generation_db_path)

    def set_up_data_preparation_db(self):
        return self._connect_to_db(self.data_preparation_db_path)

    def set_up_labeling_db(self):
        return self._connect_to_db(self.labeling_db_path)

    def set_up_train_test_db(self):
        return self._connect_to_db(self.train_test_db_path)

    # GENERALIZED FUNCTION TO TRANSFER ORIGINAL TABLES TO NEW ZONES ---------------------

    def transfer_tables_to_zone(self, source_con, target_con, tables_to_copy):
        """
        Transfer tables from the source connection (e.g., formatted, trusted, exploitation) to the target zone (e.g., analytical sandbox).

        Args:
            source_con (duckdb.DuckDBPyConnection): Source connection to the original database.
            target_con (duckdb.DuckDBPyConnection): Target connection to the new zone database.
            tables_to_copy (list): List of table names to copy from source to target zone.
        """
        for table in tables_to_copy:
            try:
                # Copy each table to the target zone
                target_con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM {table}")
                logger.info(f"Table '{table}' transferred from source to target zone.")
            except Exception as e:
                logger.error(f"Error transferring table '{table}': {e}")

    # DELETE ALL TABLES FUNCTIONS ------------------------------------------------------

    def delete_all_tables(self, con, db_name):
        try:
            tables = con.execute("SHOW TABLES").fetchall()
            for table in tables:
                table_name = table[0]
                con.execute(f"DROP TABLE IF EXISTS {table_name}")
                logger.info(f"Table '{table_name}' deleted from {db_name}.")
            logger.info(f"All tables deleted from {db_name}.")
        except Exception as e:
            logger.error(f"Failed to delete all tables from {db_name}: {e}")
            raise

    # DELETE ALL TABLES IN EACH ZONE -----------------------------------------------

    def delete_all_exploitation_tables(self):
        con = self.set_up_exploitation_db()
        self.delete_all_tables(con, "exploitation")
        con.close()


    def delete_all_analytical_sandbox_tables(self):
        con = self.set_up_analytical_sandbox_db()
        self.delete_all_tables(con, "analytical sandbox")
        con.close()

    def delete_all_feature_generation_tables(self):
        con = self.set_up_feature_generation_db()
        self.delete_all_tables(con, "feature generation")
        con.close()

    def delete_all_data_preparation_tables(self):
        con = self.set_up_data_preparation_db()
        self.delete_all_tables(con, "data preparation")
        con.close()

    def delete_all_labeling_tables(self):
        con = self.set_up_labeling_db()
        self.delete_all_tables(con, "labeling")
        con.close()

    def delete_all_train_test_tables(self):
        con = self.set_up_train_test_db()
        self.delete_all_tables(con, "train-test")
        con.close()