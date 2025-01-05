
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

from src.duckdb_manager import DuckDBManager

# Load trained model and data
model = joblib.load("src/analytical/modeling/rent_forecasting_boosting_model.pkl")
data = DuckDBManager().set_up_data_preparation_db().execute("SELECT * FROM UnifiedData")
train = DuckDBManager().set_up_train_test_db().execute("SELECT * FROM TrainDataPREPARED").fetchdf()
train = train.drop(columns=['Ratio_Lloguer_Renda', 'Estres_Economic'])
target_columns = ['Factor_Ratio_Low', 'Factor_Estres_Low']
X_train = train.drop(columns=target_columns)
y_train = train[target_columns]

test = DuckDBManager().set_up_train_test_db().execute("SELECT * FROM TestDataPREPARED").fetchdf()
test = test.drop(columns=['Ratio_Lloguer_Renda', 'Estres_Economic'])
X_test = test.drop(columns=target_columns)
y_test = test[target_columns]

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_output_distribution(model, X_test, y_test):
    """
    Plots the distribution of the predicted output values from the model.

    Parameters:
        model: The trained model (multi-output regressor wrapped around XGBoost).
        X_test: The input features for prediction (test set).
        y_test: The true target values (test set).
    """
    # Predict using the model
    y_pred = model.predict(X_test)

    # Convert predictions and actual values into a DataFrame for easy plotting
    output_df = pd.DataFrame({
        'Predicted_Factor_Ratio': y_pred[:, 0],  # First target
        'Predicted_Factor_Estres': y_pred[:, 1],  # Second target
        'True_Factor_Ratio': y_test.iloc[:, 0],  # First target
        'True_Factor_Estres': y_test.iloc[:, 1]  # Second target
    })

    # Create Streamlit layout
    st.subheader("Predicted vs True Factor_Ratio and Factor_Estres Distribution")

    # Plot the distribution of Factor_Ratio
    fig, ax = plt.subplots(1, 2, figsize=(12, 10))
    n_bins = 20
    sns.histplot(output_df['Predicted_Factor_Ratio'], kde=True, color="blue", ax=ax[0], label="Predicted", bins=n_bins)
    sns.histplot(output_df['True_Factor_Ratio'], kde=True, color="red", ax=ax[0], label="True", bins=n_bins)
    ax[0].set_title("Factor_Ratio - Predicted vs True")
    ax[0].legend()

    sns.histplot(output_df['Predicted_Factor_Estres'], kde=True, color="blue", ax=ax[1], label="Predicted", bins=n_bins)
    sns.histplot(output_df['True_Factor_Estres'], kde=True, color="red", ax=ax[1], label="True", bins=n_bins)
    ax[1].set_title("Factor_Estres - Predicted vs True")
    ax[1].legend()

    # Display the plot in Streamlit
    st.pyplot(fig)


# Function to compute model performance metrics
def evaluate_model(X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, r2, y_pred

    
# Model Performance Metrics
def model_performance_metrics():
    st.subheader("Model Performance Metrics")
    columns = st.columns(2)
    
    with columns[0]:
          mse_train, r2_train, _ = evaluate_model(X_train, y_train)
          st.write(f"Train Data - Mean Squared Error (MSE): {mse_train:.4f}")
          st.write(f"Train Data - R-squared (R²): {r2_train:.4f}")
    with columns[1]:
          mse_test, r2_test, _ = evaluate_model(X_test, y_test)
          st.write(f"Test Data - Mean Squared Error (MSE): {mse_test:.4f}")
          st.write(f"Test Data - R-squared (R²): {r2_test:.4f}")


# Example input fields for the user to provide input data
def model_input_output():
    st.subheader("Model Input - Rental Prediction")
    
    user_input = {}
    
    # Income and Economic Data
    user_input['Import_Renda_Disponible'] = st.number_input("Income Available (Import_Renda_Disponible)", min_value=0.0, value=1.0)
    user_input['Distribucio_P80_20'] = st.number_input("P80/P20 Distribution Ratio (Distribucio_P80_20)", min_value=0.0, value=1.0)
    user_input['Index_Gini'] = st.number_input("Gini Index (Index_Gini)", min_value=0.0, value=0.5)
    # Demographic Data
    user_input['Edat_Mitjana'] = st.number_input("Average Age (Edat_Mitjana)", min_value=0.0, value=35.0)
    # Price Data
    user_input['Preu_mitja'] = st.number_input("Average Price (Preu_mitja)", min_value=0.0, value=1000.0)
    
    input_df = pd.DataFrame(user_input, index=[0])

    # Make predictions
    if st.button("Predict"):
        y_pred = model.predict(input_df)
        st.subheader("Prediction Results")
        st.write(f"Predicted Rent-to-Income Ratio: {y_pred[0][0]:.2f}")
        st.write(f"Predicted Economic Stress Factor: {y_pred[0][1]:.2f}")
# Main Function
st.set_page_config(page_title="Rental Modelling - Barcelona Affordability")

if __name__ == "__main__":
    st.title("Analytical Backbone 2 - Rental Modelling 🏠")
    with st.expander("Click to see the dashboard overview"):
        st.write("""This dashboard focuses on predictive analysis within Barcelona's rental market""")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Model Input/Output", "Prediction Distribution", "Model Performance"])

    if selection == "Model Input/Output":
        model_input_output()

    elif selection == "Prediction Distribution":
        plot_output_distribution(model, X_test, y_test)

    elif selection == "Model Performance":
        model_performance_metrics()
