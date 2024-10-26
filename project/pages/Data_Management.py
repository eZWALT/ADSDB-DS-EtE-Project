import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Data Pipeline - Barcelona Housing", layout="centered")

# Title and description
st.title("Data Management Backbone 🛢️")
st.write("This dashboard guides you through a 4-stage data pipeline, from raw data ingestion to clean, analysis-ready data.")

# Function to simulate processing time with progress bars
def simulate_process(seconds=2):
    for i in range(100):
        time.sleep(seconds / 100)
        yield i + 1

# Function to generate fake data for each stage
def generate_fake_data(num_rows=10):
    # Generate raw data
    raw_data = pd.DataFrame({
        'ID': np.arange(1, num_rows + 1),
        'Rent': np.random.randint(800, 3000, num_rows),  # Random rent prices
        'Area': np.random.randint(20, 120, num_rows),  # Random areas in square meters
        'Neighborhood': np.random.choice(['Eixample', 'Gracia', 'Sants', 'Ciutat Vella'], num_rows)
    })
    return raw_data

# Main function to run the entire pipeline with visuals and progress bars
def run_pipeline():
    # Create columns for each stage
    cols = st.columns(4)
    
    # Placeholder for final results
    results = {}

    # Landing Zone
    with cols[0]:
        st.subheader("Landing Zone")
        raw_data = generate_fake_data()  # Generate fake data
        st.success("Raw data ingested into Landing Zone.")
        st.write(raw_data)

        # Progress bar for Landing Zone
        progress_bar = st.progress(0)
        eta_message = st.empty()  # Placeholder for ETA message
        for i in simulate_process(2):
            progress_bar.progress(i)
            eta_message.text(f"Estimated time (ETA): {2 - (i / 100) * 2:.2f} seconds")
        results['Landing'] = raw_data

    # Formatted Zone
    with cols[1]:
        st.subheader("Formatted Zone")
        formatted_data = results['Landing'].rename(columns={"Rent": "Monthly Rent", "Area": "Size (sqm)"})
        st.success("Data formatted successfully.")
        st.write(formatted_data)

        # Progress bar for Formatted Zone
        progress_bar = st.progress(0)
        eta_message = st.empty()  # Placeholder for ETA message
        for i in simulate_process(2):
            progress_bar.progress(i)
            eta_message.text(f"Estimated time (ETA): {2 - (i / 100) * 2:.2f} seconds")
        results['Formatted'] = formatted_data

    # Trusted Zone
    with cols[2]:
        st.subheader("Trusted Zone")
        trusted_data = results['Formatted'].dropna()  # Drop rows with NaN values as an example
        st.success("Data validated and trusted.")
        st.write(trusted_data)

        # Progress bar for Trusted Zone
        progress_bar = st.progress(0)
        eta_message = st.empty()  # Placeholder for ETA message
        for i in simulate_process(2):
            progress_bar.progress(i)
            eta_message.text(f"Estimated time (ETA): {2 - (i / 100) * 2:.2f} seconds")
        results['Trusted'] = trusted_data

    # Exploitation Zone
    with cols[3]:
        st.subheader("Exploitation Zone")
        exploitation_data = results['Trusted'].groupby('Neighborhood').agg({'Monthly Rent': 'mean', 'Size (sqm)': 'mean'}).reset_index()
        st.success("Data is ready for analysis.")
        st.write(exploitation_data)

        # Progress bar for Exploitation Zone
        progress_bar = st.progress(0)
        eta_message = st.empty()  # Placeholder for ETA message
        for i in simulate_process(2):
            progress_bar.progress(i)
            eta_message.text(f"Estimated time (ETA): {2 - (i / 100) * 2:.2f} seconds")
        results['Exploitation'] = exploitation_data

    return exploitation_data  # Return the final aggregated data

# Main Pipeline Button
if st.button("Run Entire Pipeline"):
    # Run the pipeline and show all stages with individual progress bars
    exploitation_data = run_pipeline()
    st.success("Pipeline completed successfully!")

    if exploitation_data is not None:
        st.write("### Aggregated Data Preview")
        st.dataframe(exploitation_data)
