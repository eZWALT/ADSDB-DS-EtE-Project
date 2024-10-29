# Barcelona Housing Inequality

## Description

This MLOps project seeks to provide insights into questions familiar to anyone who has lived in Barcelona over the past decade, such as:

> _"Is it just me, or are rents ridiculously sky high?"_

Using data sets related to Barcelona's demographics, rental prices, economic indicators, and incomes, this project aims to provide a clear view of the current state of the economy and the cost of living in this marvelous city through data-driven knowledge. These insights will help identify levels of social inequality and explore potential trends.

A key goal of the modeling is to predict the future evolution of incomes and rental prices, to assess how the situation could potentially be with a 10-year outlook. 

## Structure

The project is divided into two main components, or "backbones," in line with MLOps methodologies, and includes a Streamlit-based web UI for managing ETL processes, visualizations, and model predictions.

1. **Data Management Backbone**: This component ingests and transforms the source data into analytics-ready data through rigorous data engineering and data governance practices. It is divided into four stages:
   - **Landing**
   - **Formatted**
   - **Trusted**
   - **Exploitation**

2. **Analytical Backbone**: This component (currently under development) is where data mining, modeling, and further analysis occur.

## Tech Stack

- **Python**
- **DuckDB**
- **Streamlit**
- **Docker** 
- **Pandas**
- **Torch**

## Usage

You can run this project either by setting up the environment locally or using Docker. Moreover monitoring scripts can be found at /monitoring.

### Local Setup

1. **Install Python** ```Python 3.10```
2. **Install dependencies**:```bash pip install -r requirements txt```
3. **Launch the UI**: ```streamlit run Home.py``` 
4. **Open the UI**

### Dockerized Setup 
```bash
docker build -t adsdb_rocks .
docker run -p 8501:8501 adsdb_rocks
```

This will start the Streamlit UI Web application on http://localhost:8501


## Credits
This project was made by: 

- Walter J. Troiani Vargas 
- Joan Acero Pousa

The data used in this project is courtesy of Barcelona's town hall:
1. Opendata 
2. Portaldades 

## License
This project is licensed under the GPLv3 License. See the [LICENSE](../LICENSE) file for details.
