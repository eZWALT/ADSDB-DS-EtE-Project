import streamlit as st

# Set page configuration
st.set_page_config(page_title="Home - Barcelona Affordability", layout="centered")

# Title and description
st.title("🏡 Barcelona Affordability and Inequality Analysis")
st.write("""
Housing in Barcelona has seen dramatic changes over the last decade, leading to concerns over affordability and inequality.
This project seeks to analyze data related to housing costs and provide insights into these critical issues. Explore the data management and analytical features to understand the dynamics of the housing market.
""")
st.image("resources/dramatic_housing.png", caption="Barcelona housing prices is a hot topic right now :)")



st.header("How to Navigate")
st.write("""
- **Data Management**: This section provides an interactive interface for managing data through a 4-stage ETL pipeline.
- **Analytical_Pipeline**: Following up, this data manipulation pipeline generates features, labels, spitting, training and validation.
- **Rental_Affordability**: 1st analytical backbone consisting of a dashboard of exhaustive data analytics and unsupervised techniques results. 
- **Rental_Modelling**: Second backbone, an interactive interface to explore and evaluate the trained model.
""")



# Call to action
st.header("Get Started!")
st.write("""
Click below to execute the 2 data pipelines and visualize the interactive analytics:
""")
st.page_link("pages/1_Data_Management.py", label="Data Management Backbone", icon="🛢️", use_container_width=True)
st.page_link("pages/2_Analytical_Pipeline.py", label="Analytical Pipeline", icon="💾", use_container_width=True)
st.page_link("pages/3_Rental_Affordability.py", label="Rental Affordability Analytical Backbone", icon="📊", use_container_width=True)
st.page_link("pages/4_Rental_Modelling.py", label="Rental Modelling Analytical Backbone", icon="🏠", use_container_width=True)

# Team credits
st.header("Credits")
st.write("### Team Members")
st.write("- **Walter J. Troiani Vargas**")
st.write("- **Joan Acero Pousa**")
st.caption("We hope you find this project insightful!")