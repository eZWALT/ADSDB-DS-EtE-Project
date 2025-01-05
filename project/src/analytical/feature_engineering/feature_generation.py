from loguru import logger
from src.duckdb_manager import DuckDBManager

import pandas as pd

def feature_generation_driver() -> None: 
    #Initial data loading and database connections
    manager = DuckDBManager()
    sandbox_con = manager.set_up_analytical_sandbox_db()
    featuregen_con = manager.set_up_feature_generation_db()
    manager.delete_all_feature_generation_tables()
    df = sandbox_con.execute(f"SELECT * FROM UnifiedData").fetchdf()
    
    #Scale Gini Index in a range [0,1] instead of [0, 100]
    df['Index_Gini'] = df['Index_Gini'] / 100

    #Create the new features X
    df['Preu_mitja_Anual'] = df['Preu_mitja'] * 12
    df['Ratio_Lloguer_Renda'] = df['Preu_mitja_Anual'] / df['Import_Renda_Disponible']
    df['Estres_Economic'] = df['Ratio_Lloguer_Renda'] * 0.8 + (df["Index_Gini"]) * 0.2
    df['Lloguer_Historic_mitja'] = df.groupby('Nom_Barri')['Preu_mitja'].transform('mean')

    # Create table in the database and close the connection
    featuregen_con.execute("CREATE TABLE UnifiedData AS SELECT * FROM df")
    logger.info("Tables in the feature generation database: " + str(manager.list_tables(featuregen_con)))
    featuregen_con.commit()
    featuregen_con.close()
