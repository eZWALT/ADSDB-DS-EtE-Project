from loguru import logger

from src.duckdb_manager import DuckDBManager

def formatted_driver() -> None:  
    # Initialize the DuckDBManager
    db_manager = DuckDBManager()
    # Set up DuckDB
    connection = db_manager.set_up_duck_db()
    # Ensure the db is empty to proceed with the loading of databases from the landing zone
    db_manager.delete_all_tables(connection)
    # Create tables from CSV files
    db_manager.create_tables_from_csv(connection)
    # List created tables
    db_manager.list_tables(connection)
    #save changes in the db
    connection.commit()
    connection.close()

    logger.info("Tables in the formatted zone database: " + str(db_manager.list_tables(db_manager.set_up_duck_db())))

