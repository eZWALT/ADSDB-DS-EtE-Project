from loguru import logger
from src.duckdb_manager import DuckDBManager

import pandas as pd

def labeling_driver() -> None: 
    #Initial data loading and database connections
    manager = DuckDBManager()
    featuregen_con = manager.set_up_feature_generation_db()
    labeling_con = manager.set_up_labeling_db()
    manager.delete_all_labeling_tables()
    df = featuregen_con.execute(f"SELECT * FROM UnifiedData").fetchdf()
    
    #Create the new labels Y
    #Thresholds for considering high (TODO: Put these important values in a configuration file)
    RATIO_THRESHOLD = 0.45
    STRESS_THRESHOLD = 0.45

    df['Factor_Ratio'] = df['Ratio_Lloguer_Renda'].apply(
        lambda x: 'High' if x > RATIO_THRESHOLD else 'Low'
    ).astype('category')  

    df['Factor_Estres'] = df['Estres_Economic'].apply(
        lambda x: 'High' if x > STRESS_THRESHOLD else 'Low'
    ).astype('category')  

    # Create table in the database and close the connection
    labeling_con.execute("CREATE TABLE UnifiedData AS SELECT * FROM df")
    logger.info("Tables in the labeling database: " + str(manager.list_tables(labeling_con)))
    labeling_con.commit()
    labeling_con.close()