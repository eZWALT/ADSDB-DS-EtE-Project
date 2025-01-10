import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import warnings
from pandas.errors import PerformanceWarning
from src.duckdb_manager import DuckDBManager

warnings.simplefilter("ignore", PerformanceWarning)

# --- Constants and Configuration ---
st.set_page_config(page_title="Rental Modelling - Barcelona Affordability")
NUM_TARGETS = ["Ratio_Lloguer_Renda", "Estres_Economic"]
CAT_TARGETS = ["Factor_Ratio", "Factor_Estres"]
TARGET_COLUMNS = ["Factor_Ratio_Low", "Factor_Estres_Low"]

NUM_COLS = [
    "Import_Renda_Disponible",
    "Index_Gini",
    "Import_Renda_Neta",
    "Distribucio_P80_20",
    "Import_Renda_Bruta",
    "Edat_Mitjana",
    "Preu_mitja",
    "Recompte",
    "Preu_mitja_Anual",
    "Lloguer_Historic_mitja",
]
CAT_COLS = ["Any", "Nom_Barri", "Seccio_Censal"]

# --- Utility Functions ---
def load_data_and_model():
    """Load the model and data from DuckDB."""
    model = joblib.load("src/analytical/modeling/rent_forecasting_boosting_model.pkl")
    data = DuckDBManager().set_up_data_preparation_db().execute("SELECT * FROM UnifiedData").fetchdf()
    train = DuckDBManager().set_up_train_test_db().execute("SELECT * FROM TrainDataPREPARED").fetchdf()
    test = DuckDBManager().set_up_train_test_db().execute("SELECT * FROM TestDataPREPARED").fetchdf()
    
    # Drop unnecessary columns
    train = train.drop(columns=NUM_TARGETS)
    test = test.drop(columns=NUM_TARGETS)
    X_train, y_train = train.drop(columns=TARGET_COLUMNS), train[TARGET_COLUMNS]
    X_test, y_test = test.drop(columns=TARGET_COLUMNS), test[TARGET_COLUMNS]
    return model, data, X_train, y_train, X_test, y_test

def normalize_column(df, column, method="min-max"):
    """Normalize a column using Min-Max or Z-Score normalization."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    if method == "min-max":
        col_min, col_max = df[column].min(), df[column].max()
        return (df[column] - col_min) / (col_max - col_min)
    elif method == "z-score":
        col_mean, col_std = df[column].mean(), df[column].std()
        return (df[column] - col_mean) / col_std
    else:
        raise ValueError("Unsupported normalization method. Use 'min-max' or 'z-score'.")

def encode_cat_cols(df, cat_cols):
    """Encode categorical columns using One-Hot Encoding."""
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

def preprocess_input(input_values, num_cols, cat_cols, model_features):
    """Preprocess input data for prediction."""
    input_df = pd.DataFrame([input_values])
    for col in num_cols:
        if col in input_df.columns:
            input_df[col] = normalize_column(input_df, col)
    input_df = encode_cat_cols(input_df, cat_cols)
    
    # Ensure all model features are present
    missing_cols = set(model_features) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    return input_df[model_features]

def evaluate_model(X, y, model):
    """Compute model performance metrics."""
    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob > 0.5).astype(int)
    metrics = {}
    for i, target in enumerate(y.columns):
        metrics[target] = {
            "Accuracy": accuracy_score(y.iloc[:, i], y_pred[:, i]),
            "Precision": precision_score(y.iloc[:, i], y_pred[:, i]),
            "Recall": recall_score(y.iloc[:, i], y_pred[:, i]),
            "F1 Score": f1_score(y.iloc[:, i], y_pred[:, i]),
        }
    return metrics

# --- Component Functions ---
def interface_page(model, reference_data, num_cols, cat_cols, targets):
    """Display the user interface for input and prediction."""
    st.title("Housing Affordability Prediction Dashboard")
    st.subheader("Provide Input Data for Prediction")
    col1, col2, col3 = st.columns(3)

    user_input = {}
    unique_categorical_values = {col: reference_data[col].unique() for col in cat_cols}

    for idx, col in enumerate(num_cols):
        with [col1, col2, col3][idx % 3]:
            user_input[col] = st.number_input(
                f"{col}:", 
                min_value=float(reference_data[col].min()), 
                max_value=float(reference_data[col].max()), 
                value=float(reference_data[col].mean())
            )
    for idx, col in enumerate(cat_cols):
        with [col1, col2, col3][idx % 3]:
            user_input[col] = st.selectbox(f"{col}:", options=unique_categorical_values[col])

    if st.button("Predict"):
        input_df = preprocess_input(user_input, num_cols, cat_cols, model.feature_names_in_)
        predictions = model.predict(input_df)
        st.subheader("Prediction Results")
        for i, target in enumerate(targets):
            st.write(f"{target}: {'Low' if predictions[0, i] == 0 else 'High'}")

def prediction_distribution_page(y_test, y_pred, y_pred_prob):
    """Display the distribution of predictions."""
    st.title("Predicted vs True Distribution")
    output_df = pd.DataFrame({
        "Predicted_Factor_Ratio": y_pred[:, 0],
        "Predicted_Factor_Estres": y_pred[:, 1],
        "True_Factor_Ratio": y_test.iloc[:, 0],
        "True_Factor_Estres": y_test.iloc[:, 1],
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(output_df["Predicted_Factor_Ratio"], kde=False, color="blue", ax=axes[0], bins=2, label="Predicted")
    sns.histplot(output_df["True_Factor_Ratio"], kde=False, color="red", ax=axes[0], bins=2, label="True")
    axes[0].set_title("Factor_Ratio Distribution")
    axes[0].legend(title="Legend")    
    sns.histplot(output_df["Predicted_Factor_Estres"], kde=False, color="blue", ax=axes[1], bins=2, label="Predicted")
    sns.histplot(output_df["True_Factor_Estres"], kde=False, color="red", ax=axes[1], bins=2, label="True")
    axes[1].set_title("Factor_Estres Distribution")
    axes[1].legend(title="Legend") 
    st.pyplot(fig)

def performance_metrics_page(X_train, y_train, X_test, y_test, model):
    """Display model performance metrics."""
    st.title("Model Performance Metrics")
    train_metrics = evaluate_model(X_train, y_train, model)
    test_metrics = evaluate_model(X_test, y_test, model)
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Train Metrics")
        for target, metrics in train_metrics.items():
            st.write(f"**{target}:**")
            for metric, value in metrics.items():
                st.write(f"{metric}: {value:.2f}")
            st.write("-" * 30)
    with col2:
        st.write("### Test Metrics")
        for target, metrics in test_metrics.items():
            st.write(f"**{target}:**")
            for metric, value in metrics.items():
                st.write(f"{metric}: {value:.2f}")
            st.write("-" * 30)

def evaluation_page(y_test, y_pred, y_pred_prob):
    """Display confusion matrices and ROC curves."""
    st.title("Model Evaluation Metrics")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i, (target, ax) in enumerate(zip(CAT_TARGETS, axes)):
        cm = confusion_matrix(y_test.iloc[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"Confusion Matrix: {target}")
    st.pyplot(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i, (target, ax) in enumerate(zip(CAT_TARGETS, axes)):
        fpr, tpr, _ = roc_curve(y_test.iloc[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_title(f"ROC Curve: {target}")
        ax.legend(loc="lower right")
    st.pyplot(fig)

# --- Main Application ---
def main():
    st.title("Analytical Backbone 2 - Rental Modelling 🏠")
    with st.expander("Click to see the dashboard overview"):
        st.write("""
            This model predicts economic factors affecting housing affordability in Barcelona, 
            classifying areas as low or high in rent affordability and economic stress. 
            Using XGBoost with hyperparameter tuning, it provides reliable predictions 
            evaluated through accuracy, precision, recall, and F1 scores, aiding in identifying 
            economically stressed and unaffordable areas. Also further model evaluation is performed through
            confusion matrix and AUC-ROC curve. Finally, we present a user-friendly interface for interacting
            with the boosted model to forecast economic stress and the Rent-to-Income Ratio.
        """)
    model, data, X_train, y_train, X_test, y_test = load_data_and_model()
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Interface", "Prediction Distribution", "Performance", "Evaluation"])
    if selection == "Interface":
        interface_page(model, data, NUM_COLS, CAT_COLS, CAT_TARGETS)
    elif selection == "Prediction Distribution":
        prediction_distribution_page(y_test, y_pred, y_pred_prob)
    elif selection == "Performance":
        performance_metrics_page(X_train, y_train, X_test, y_test, model)
    elif selection == "Evaluation":
        evaluation_page(y_test, y_pred, y_pred_prob)

if __name__ == "__main__":
    main()
