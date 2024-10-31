import streamlit as st
from loguru import logger
from io import StringIO

from src.data_management.landing_zone import landing
from src.data_management.formatted_zone import formatted 
from src.data_management.trusted_zone import trusted
from src.data_management.exploitation_zone import exploitation
from src.data_management.data_ingestion import ingestion


# DRIVER CODE FOR THE DATA MANAGEMENT BACKBONE
# Global settings
st.set_page_config(page_title="Data Management - Barcelona Housing")
zone_avg_duration = [5, 5, 5, 5, 5]  # Estimated duration for each zone in seconds

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
    cols = st.columns(5)
    etl_functions = {
        "Ingestion": ingestion.ingestion_driver,
        "Landing": landing.landing_driver,
        "Formatted": formatted.formatted_driver,
        "Trusted": trusted.trusted_driver,
        "Exploitation": exploitation.exploitation_driver
    }

    for index, (zone_name, function) in enumerate(etl_functions.items()):
        with cols[index]:
            execute_zone_with_animation(zone_name, function, log_area, duration=zone_avg_duration[index])
            log_area.text(log_capture.getvalue())

    st.success("Pipeline completed successfully!")

if __name__ == "__main__":
    # Capture logs in-memory
    log_capture = StringIO()
    logger.remove()
    logger.add(log_capture, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")

    st.title("Data Management 🛢️")
    st.write("This dashboard guides you through a 4-stage data pipeline, from raw data ingestion to clean, analysis-ready data.")

    st.subheader("Pipeline Log")
    log_area = st.empty()

    if st.button("Run Entire Pipeline"):
        run_pipeline(log_area)
