from src.duckdb_manager import DuckDBManager
from loguru import logger
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# TODO: Parametrize this default value inside a configuration file
path_to_model = "src/analytical/modeling/rent_forecasting_boosting_model.pkl"

from typing import List, Tuple, Dict

# Modify the validation driver function to return the metrics and summary
def validation_driver() -> Dict:
    try:
        manager = DuckDBManager()
        data_con = manager.set_up_train_test_db()
        test = data_con.execute("SELECT * FROM TestDataPREPARED").fetchdf()
        test = test.drop(columns=['Ratio_Lloguer_Renda', 'Estres_Economic'])
        model = joblib.load(path_to_model)

        target_columns = ['Factor_Ratio_Low', 'Factor_Estres_Low']
        X_test = test.drop(columns=target_columns)
        y_test = test[target_columns]

        logger.info("Making predictions on the test set.")
        y_pred = model.predict(X_test)

        metrics = []
        for i, target in enumerate(['Factor_Ratio', 'Factor_Estres']):
            mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            metrics.append((target, mse, r2))
            logger.info(f"Performance for {target}: MSE = {mse:.4f}, R² = {r2:.4f}")

        # Log insights into predictions
        logger.info("Providing summary statistics for predictions.")
        pred_summary = pd.DataFrame(y_pred, columns=['Pred_Factor_Ratio', 'Pred_Factor_Estres']).describe()
        logger.debug(f"Prediction summary:\n{pred_summary}")

        logger.success("Validation process completed successfully.")
        
        # Return the results as a dictionary
        return {
            "metrics": metrics,
            "prediction_summary": pred_summary
        }
        
    except Exception as e:
        logger.error(f"An error occurred during validation: {e}", exc_info=True)
        raise
