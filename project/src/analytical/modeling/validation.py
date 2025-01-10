from src.duckdb_manager import DuckDBManager
from loguru import logger
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score,  accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import joblib

# TODO: Parametrize this default value inside a configuration file
path_to_model = "src/analytical/modeling/rent_forecasting_boosting_model.pkl"

from typing import List, Tuple, Dict

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return accuracy, precision, recall, f1

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
        # Predict on the test set
        y_pred_prob = model.predict(X_test)
        # Convert probabilities to binary predictions using a threshold
        y_pred = (y_pred_prob > 0.5).astype(int)

        for i, target in enumerate(['Factor_Ratio', 'Factor_Estres']):
            accuracy, precision, recall, f1 = calculate_metrics(y_test.iloc[:, i], y_pred[:, i])
            logger.info(f"Performance for {target}:\n Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

        # Log insights into predictions
        logger.info("Providing summary statistics for predictions.")
        pred_summary = pd.DataFrame(y_pred, columns=['Pred_Factor_Ratio', 'Pred_Factor_Estres']).describe()
        logger.debug(f"Prediction summary:\n{pred_summary}")
        logger.success("Validation process completed successfully.")

        
    except Exception as e:
        logger.error(f"An error occurred during validation: {e}", exc_info=True)
        raise


