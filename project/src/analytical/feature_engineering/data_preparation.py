from loguru import logger
from src.duckdb_manager import DuckDBManager
import pandas as pd


def normalize_column(df, column, method="min-max"):
    """
    Normalize a column in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to be normalized.
        method (str): The normalization method ("min-max" or "z-score").

    Returns:
        pd.Series: A pandas Series with the normalized values.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    if method == "min-max":
        # Min-Max Normalization: (x - min) / (max - min)
        col_min = df[column].min()
        col_max = df[column].max()
        normalized = (df[column] - col_min) / (col_max - col_min)
    elif method == "z-score":
        # Z-Score Normalization: (x - mean) / std
        col_mean = df[column].mean()
        col_std = df[column].std()
        normalized = (df[column] - col_mean) / col_std
    else:
        raise ValueError("Unsupported normalization method. Use 'min-max' or 'z-score'.")

    return normalized


def encode_categorical_columns(df, categorical_columns):
    """
    Encode categorical columns using One-Hot Encoding.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        categorical_columns (list): List of categorical columns to encode.

    Returns:
        pd.DataFrame: The DataFrame with encoded categorical columns.
    """
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df_encoded

def data_preparation_driver() -> None: 
    #Initial data loading and database connections
    manager = DuckDBManager()
    labeling_con = manager.set_up_labeling_db()
    dataprep_con = manager.set_up_data_preparation_db()
    manager.delete_all_data_preparation_tables()
    df = labeling_con.execute(f"SELECT * FROM UnifiedData").fetchdf()

    #Normalize dataset
    numerical_columns = [
    'Import_Renda_Disponible', 'Index_Gini', 'Import_Renda_Neta',
    'Distribucio_P80_20', 'Import_Renda_Bruta', 'Edat_Mitjana',
    'Preu_mitja', 'Recompte', 'Preu_mitja_Anual', 'Ratio_Lloguer_Renda',
    'Estres_Economic', 'Lloguer_Historic_mitja'
    ]

    df_normalized = df.copy()  
    for column in numerical_columns:
        df_normalized[column] = normalize_column(df_normalized, column, method="min-max")

    #Encode dataset
    categorical_columns = ['Any', 'Nom_Barri', 'Seccio_Censal', 'Factor_Ratio', 'Factor_Estres']
    df_normalized_encoded = encode_categorical_columns(df_normalized, categorical_columns)

    # Create final table in the Feature generation database
    dataprep_con.execute("CREATE TABLE UnifiedData AS SELECT * FROM df")
    dataprep_con.execute("CREATE TABLE UnifiedDataNormalizedEncoded AS SELECT * FROM df_normalized_encoded")
    logger.info("Tables in the data preparation database: " + str(manager.list_tables(dataprep_con)))
    dataprep_con.commit()
    dataprep_con.close()