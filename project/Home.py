import streamlit as st

# Set page configuration
st.set_page_config(page_title="Home - Barcelona Housing", layout="centered")

# Title and description
st.title("🏡 Barcelona Housing Inequality Insights")
st.write("""
Housing in Barcelona has seen dramatic changes over the last decade, leading to concerns over affordability and inequality.
This project seeks to analyze data related to housing costs and provide insights into these critical issues. Explore the data management and analytical features to understand the dynamics of the housing market.
""")
st.image("resources/dramatic_housing.png", caption="Barcelona housing prices is a hot topic right now :)")



st.header("How to Navigate")
st.write("""
- **Data Management**: This section provides an interactive interface for managing data through a 4-stage ETL pipeline.
- **Analytical**: Here, you can explore visualizations and perform analysis, gather KPI's and visualize model forecastings.
""")



# Call to action
st.header("Get Started!")
st.write("""
Click below to explore our data management tools or dive into the analytical features:
""")
st.page_link("pages/Data_Management.py", label="Data Management Backbone", icon="🛢️", use_container_width=True)
st.page_link("pages/Analytical.py", label="Analytical Backbone", icon="📊", use_container_width=True)

# Team credits
st.header("Credits")
st.write("""
This project was created by:
- Walter J. Troiani Vargas
- Joan Acero Pousa
""")
