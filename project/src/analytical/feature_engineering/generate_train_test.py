from loguru import logger
from src.duckdb_manager import DuckDBManager
from sklearn.model_selection import train_test_split

def split_driver() -> None: 
    #Initial data loading and database connections
    manager = DuckDBManager()
    dataprep_con = manager.set_up_data_preparation_db()
    split_con = manager.set_up_train_test_db()
    manager.delete_all_train_test_tables()

    df = dataprep_con.execute(f"SELECT * FROM UnifiedData").fetchdf()
    dfprepared = dataprep_con.execute(f"SELECT * FROM UnifiedDataNormalizedEncoded").fetchdf()

    #Split the heck out of this dataset
    trainMODEL, testMODEL = train_test_split(dfprepared, test_size=0.2, random_state=42)
    split_con.execute("CREATE TABLE UnifiedDataBasic AS SELECT * FROM df")

    split_con.execute("CREATE TABLE TrainDataPREPARED AS SELECT * FROM trainMODEL")
    split_con.execute("CREATE TABLE TestDataPREPARED AS SELECT * FROM testMODEL")
    split_con.execute("CREATE TABLE PREPAREDData AS SELECT * FROM dfprepared")

    logger.info("Tables in the train test split database: " + str(manager.list_tables(split_con)))
    split_con.commit()
    split_con.close()