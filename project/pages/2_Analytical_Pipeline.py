import streamlit as st
from loguru import logger
from io import StringIO

from src.analytical.analytical_sandbox import sandbox
from src.analytical.feature_engineering import feature_generation
from src.analytical.feature_engineering import labeling
from src.analytical.feature_engineering import data_preparation
from src.analytical.feature_engineering import generate_train_test
from src.analytical.modeling import training 
from src.analytical.modeling import validation

# DRIVER CODE FOR THE ANALYTICAL PIPELINE 
# Global settings
st.set_page_config(page_title="Analytical Pipeline - Barcelona Affordability")
zone_avg_duration = [5, 5, 5, 5, 5, 25, 5]  # Estimated duration for each zone in seconds

# Wrapper function to execute ETL tasks with a spinner and ETA display
def execute_zone_with_animation(zone_name, etl_function, log_area, duration=2):
    logger.info(f"Starting {zone_name}.")
    log_area.text(log_capture.getvalue())
    
    st.subheader(zone_name)
    with st.spinner(f"Processing... ETA: {duration} seconds"):
        try:
            etl_function()
            logger.success(f"Completed {zone_name}.")
            log_area.text(log_capture.getvalue())
            st.success(f"{zone_name} completed!")
        except Exception as e:
            logger.error(f"Failed {zone_name}: {str(e)}")
            log_area.text(log_capture.getvalue())
            st.error(f"Error in {zone_name}")
            st.stop()

def run_pipeline(log_area):
    etl_functions = [
        ("Analytical Sandbox", sandbox.sandbox_driver),
        ("Feature Generation", feature_generation.feature_generation_driver),
        ("Labeling", labeling.labeling_driver),
        ("Data Preparation", data_preparation.data_preparation_driver),
        ("Generate Train Test", generate_train_test.split_driver),
        ("Model Training", training.training_driver),
        ("Model Validation", validation.validation_driver),
    ]

    # Group zones into rows with a maximum of 3 columns per row
    max_columns_per_row = 3
    for i in range(0, len(etl_functions), max_columns_per_row):
        cols = st.columns(max_columns_per_row)

        # Iterate over each column and corresponding ETL function
        for j, (zone_name, function) in enumerate(etl_functions[i:i + max_columns_per_row]):
            with cols[j]:
                execute_zone_with_animation(zone_name, function, log_area, duration=zone_avg_duration[i + j])
                    
    st.success("Analytical Pipeline completed successfully!")


if __name__ == "__main__":
    # Capture logs in-memory
    log_capture = StringIO()
    logger.remove()
    logger.add(log_capture, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")

    st.title("Analytical Pipeline 💾")
    st.write("This dashboard takes you through a comprehensive 3-stage data pipeline, building on initial data management to deliver advanced feature engineering, model training, and validation.")

    st.subheader("Analytical Pipeline Log")
    log_area = st.empty()

    if st.button("Run Entire Pipeline"):
        run_pipeline(log_area)
