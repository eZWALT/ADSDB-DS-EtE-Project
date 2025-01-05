from src.duckdb_manager import DuckDBManager
from loguru import logger
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import time

# TODO: Parametrize this default value inside a configuration file
path_to_model = "src/analytical/modeling/rent_forecasting_boosting_model.pkl"

def training_driver() -> None:
    try:
        start_time = time.time()
        manager = DuckDBManager()
        split_con = manager.set_up_train_test_db()
        manager.delete_all_analytical_sandbox_tables()
        train = split_con.execute("SELECT * FROM TrainDataPREPARED").fetchdf()

        # Drop numerical target columns
        train = train.drop(columns=['Ratio_Lloguer_Renda', 'Estres_Economic'])
        target_columns = ['Factor_Ratio_Low', 'Factor_Estres_Low']
        X_train = train.drop(columns=target_columns)
        y_train = train[target_columns]
        logger.info(f"Prepared training dataset: X shape: {X_train.shape}, Y shape: {y_train.shape}")
        xgb_model = XGBRegressor(random_state=42)
        multi_output_model = MultiOutputRegressor(xgb_model)

        param_grid = {
            'estimator__n_estimators': [100, 200],
            'estimator__max_depth': [3, 5, 7],
            'estimator__learning_rate': [0.01, 0.1, 0.2],
            'estimator__subsample': [0.8, 1.0],
            'estimator__colsample_bytree': [0.8, 1.0],
        }
        grid_search = GridSearchCV(
            multi_output_model,
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        logger.info("Starting Grid Search training.")
        grid_search_start = time.time()
        grid_search.fit(X_train, y_train)
        logger.success(f"GridSearchCV completed in {time.time() - grid_search_start:.2f} seconds.")

        best_model = grid_search.best_estimator_
        logger.success(f"Model successfully trained and tuned. Best parameters: {grid_search.best_params_}")

        joblib.dump(best_model, path_to_model)
        logger.success(f"Model saved successfully at {path_to_model}")

        logger.success(f"Training process completed in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        raise
