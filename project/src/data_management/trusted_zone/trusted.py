import pandas as pd 
import duckdb 

from loguru import logger
from src.duckdb_manager import DuckDBManager

def trusted_driver() -> None:  
    db_manager = DuckDBManager()
    # Clean the database to ensure the loading will correctly made from the formatted zone.
    db_manager.delete_all_trusted_tables()
    # Create the unified tables in the trusted zone
    db_manager.unify_tables_by_dataset()
    #list all tables in the trusted zone
    logger.info("Tables in the trusted zone database: " + str(db_manager.list_tables(db_manager.set_up_trusted_db())))
