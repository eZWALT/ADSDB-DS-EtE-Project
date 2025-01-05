from loguru import logger
from src.duckdb_manager import DuckDBManager

def sandbox_driver() -> None: 
    manager = DuckDBManager()
    exploitation_con = manager.set_up_exploitation_db()
    sandbox_con = manager.set_up_analytical_sandbox_db()
    manager.delete_all_analytical_sandbox_tables()
    df = exploitation_con.execute(f"SELECT * FROM UnifiedData").fetchdf()

    # Create table in database
    sandbox_con.execute("CREATE TABLE UnifiedData AS SELECT * FROM df")
    logger.info("Tables in the analytical sandbox database: " + str(manager.list_tables(sandbox_con)))
    sandbox_con.commit()
    sandbox_con.close()