import streamlit as st 
import pandas as pd 
import geopandas as gpd
from shapely import wkt
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Clustering
from sklearn.cluster import AgglomerativeClustering 



# Importing necessary custom modules
from src.barcelona_manager import BarcelonaManager
from src.duckdb_manager import DuckDBManager

# Set up the page configuration
st.set_page_config(page_title="Rental Affordability - Barcelona Affordability")


def plot_affordability_and_stress(df):
    # Top 10 most affordable neighborhoods
    top_10_affordable = df.groupby("Nom_Barri")['Preu_mitja'].mean().nsmallest(10)
    
    # Top 10 least affordable neighborhoods
    top_10_least_affordable = df.groupby("Nom_Barri")['Preu_mitja'].mean().nlargest(10)
    
    # Count of economically stressed areas by neighborhood
    stress_by_neighborhood = df[df['Factor_Estres'] == 'High']['Nom_Barri'].value_counts()

    # Set up the figure and subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=False, gridspec_kw={'hspace': 0.4})

    # Subplot 1: Top 10 Most Affordable Neighborhoods
    sns.barplot(
        x=top_10_affordable.values,
        y=top_10_affordable.index,
        hue=top_10_affordable.index,
        palette="YlGnBu",
        ax=axes[0],
        legend=False
    )
    axes[0].set_title("Top 10 Most Affordable Neighborhoods", fontsize=14)
    axes[0].set_xlabel("Average Rent Price", fontsize=12)
    axes[0].set_ylabel("Neighborhood", fontsize=12)
    axes[0].bar_label(axes[0].containers[0], fmt="%.2f")

    # Subplot 2: Top 10 Least Affordable Neighborhoods
    sns.barplot(
        x=top_10_least_affordable.values,
        y=top_10_least_affordable.index,
        hue=top_10_least_affordable.index,
        palette="Reds",
        ax=axes[1],
        legend=False
    )
    axes[1].set_title("Top 10 Least Affordable Neighborhoods", fontsize=14)
    axes[1].set_xlabel("Average Rent Price", fontsize=12)
    axes[1].set_ylabel("Neighborhood", fontsize=12)
    axes[1].bar_label(axes[1].containers[0], fmt="%.2f")

    # Subplot 3: Count of Economically Stressed Areas by Neighborhood
    sns.barplot(
        x=stress_by_neighborhood.values,
        y=stress_by_neighborhood.index,
        hue=stress_by_neighborhood.index,
        palette="vlag",
        ax=axes[2],
        legend=False
    )
    axes[2].set_title("Count of Economically Stressed Areas by Neighborhood", fontsize=14)
    axes[2].set_xlabel("Count of Economically Stressed Areas", fontsize=12)
    axes[2].set_ylabel("Neighborhood", fontsize=12)
    axes[2].bar_label(axes[2].containers[0], fmt="%d")

    # Add a common footer
    fig.suptitle("Neighborhood Insights: Affordability and Economic Stress", fontsize=16, weight='bold')
    
    # Adjust layout for better visual appearance
    fig.tight_layout(pad=2.0)  # Adjust padding for a clean layout

    # Display the plot in Streamlit
    st.pyplot(fig)
    
def plot_boxplots(df):
    # Create a dictionary to hold the data for each metric
    metrics = {
        "Rent-to-Income Ratio": "Ratio_Lloguer_Renda",
        "Gini Index": "Index_Gini",
        "Economic Stress Index": "Estres_Economic"
    }

    # Create the figure for plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), dpi=80)

    # Iterate through the metrics to create each boxplot
    for idx, (title, metric) in enumerate(metrics.items(), 0):
        # Calculate the median value for each neighborhood and order them
        median_values = df.groupby('Nom_Barri')[metric].median().sort_values().index
        
        # Create the boxplot for the current metric
        sns.boxplot(data=df, x="Nom_Barri", y=metric, hue="Nom_Barri", order=median_values, ax=axes[idx])
        axes[idx].set_title(f"Distribution of {title} by Neighborhood")
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=90)
        axes[idx].set_ylabel(title)
        axes[idx].set_xlabel("Neighborhood")
    
    # Adjust layout for Streamlit
    plt.subplots_adjust(hspace=0.5)

    # Show the plot in Streamlit
    st.pyplot(fig)

def plot_pie_charts(df):
    # Pie Chart Data
    stress_counts = df['Factor_Estres'].value_counts()
    ratio_counts = df['Factor_Ratio'].value_counts()

    # Side-by-Side Pie Charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Pie chart for Factor_Estres
    axes[0].pie(
        stress_counts,
        labels=stress_counts.index,
        autopct='%1.1f%%',
        colors=["red", "cyan"],
        startangle=90
    )
    axes[0].set_title("Proportion of High vs Low Economic Stress")

    # Pie chart for Factor_Ratio
    axes[1].pie(
        ratio_counts,
        labels=ratio_counts.index,
        autopct='%1.1f%%',
        colors=["red", "cyan"],
        startangle=90
    )
    axes[1].set_title("Proportion of High vs Low Rent-to-Income Ratio")

    # Adjust layout and display the plot
    plt.tight_layout()
    
    # Show the plot in Streamlit
    st.pyplot(fig)

def plot_correlation_heatmap(corr_df): 
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title("Correlation among the variables")
    st.pyplot(plt)

def prepare_geo_data():
    # Join datasets
    merged = geo_data.merge(estres_df, left_on='nom_barri', right_on='Nom_Barri', how='left')
    merged = merged.merge(ratio_df, on='Nom_Barri', how='left')
    return merged

def plot_heatmap(geo_data, merged, column, title, legend_label, cmap, vmin=None, vmax=None):
    """
    Plots a heatmap for a specific column.
    
    Args:
    - geo_data: GeoDataFrame containing all neighborhoods.
    - merged: Merged GeoDataFrame with the column to plot.
    - column: Column name to visualize.
    - title: Title for the plot.
    - legend_label: Label for the legend.
    - cmap: Color map for the heatmap.
    - vmin: Minimum value for the color scale.
    - vmax: Maximum value for the color scale.
    """
    # Boolean placeholder for available data
    merged['has_data'] = ~merged[column].isnull()

    # Plot the heatmap
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Plot all neighborhoods in grey as the base layer
    geo_data.plot(ax=ax, color='lightgrey', edgecolor='black', label='No Data')

    # Overlay neighborhoods with data
    merged[merged['has_data']].plot(
        ax=ax,
        column=column,
        cmap=cmap,
        edgecolor='black',
        legend=True,
        vmin=vmin,
        vmax=vmax,
        legend_kwds={
            'label': legend_label,
            'orientation': "vertical"
        }
    )

    # Add missing neighborhoods in grey
    merged[~merged['has_data']].plot(
        ax=ax,
        color='lightgrey',
        label='No Information'
    )

    # Customize plot
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

def pca_biplot(data, binary_variable, n_components=2):
    # Scale the input data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(scaled_data)
    explained_ratios = pca.explained_variance_ratio_
    pca_loadings = pca.components_

    # Scale PCA scores for consistent plot dimensions
    xs = pca_scores[:, 0]
    ys = pca_scores[:, 1]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = binary_variable.map({"Low": 'blue', "High": 'red'})

    # Scatter plot of PCA scores
    ax.scatter(xs * scalex, ys * scaley, alpha=0.4, s=10, c=colors)

    # Add arrows for PCA loadings
    for i in range(pca_loadings.shape[1]):
        ax.arrow(0, 0, pca_loadings[0, i], pca_loadings[1, i], color='r', alpha=0.5)
        label = data.columns[i] if data.columns is not None else f"Var{i+1}"
        ax.text(
            pca_loadings[0, i] * 1.15, pca_loadings[1, i] * 1.15, 
            label, color='b', ha='center', va='center'
        )

    # Labels and title
    ax.set_xlabel(f"Principal Component 1 ({100 * explained_ratios[0]:.2f}%)")
    ax.set_ylabel(f"Principal Component 2 ({100 * explained_ratios[1]:.2f}%)")
    ax.grid()
    ax.set_title(f"PCA Biplot (Scaled) - {100 * sum(explained_ratios):.2f}% Variance Explained")

    # Legend
    ax.scatter([], [], color='blue', label='Low')
    ax.scatter([], [], color='red', label='High')
    ax.legend(loc='upper right', title='Factor Estres')

    # Render plot in Streamlit
    st.pyplot(fig)

    return pca_scores


def cluster_analysis_pca(data, n_clusters=5):
    # Perform Agglomerative Clustering
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="complete")
    cluster_labels = agg_clustering.fit_predict(data)

    # Predefined cluster titles based on domain knowledge
    cluster_titles = [
        "Cluster 0: Moderate-Income, Diverse Rent Prices, Old Residents",
        "Cluster 1: Low-Income, High Rent, Young Residents",
        "Cluster 2: High Income, High Rent, Diverse Neighborhoods",
        "Cluster 3: Low Income, Mid Rent, Young Residents",
        "Cluster 4: Low Income, Low Rent, Old Residents"
    ]
    # Ensure titles match number of clusters
    cluster_titles = cluster_titles[:n_clusters]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7)

    # Annotate each cluster
    for i in range(n_clusters):
        cluster_center_x = np.mean(data[cluster_labels == i, 0])
        cluster_center_y = np.mean(data[cluster_labels == i, 1])

        # Add a marker for cluster center
        ax.scatter(cluster_center_x, cluster_center_y, color='black', marker='x', s=100, edgecolor='white', linewidth=2)

        # Add text label for the cluster
        ax.text(
            cluster_center_x, cluster_center_y, cluster_titles[i],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=8, color='black', weight='bold'
        )

    # Customize the plot
    ax.set_title('Agglomerative Clustering - PCA Projection')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.grid(True)

    # Render plot in Streamlit
    st.pyplot(fig)

    return cluster_labels


# Descriptive Analysis Section
def descriptive_analysis():
    with st.expander("Descriptive Analysis 📊"):
        st.subheader("Descriptive Analysis 📊")
        
        # Correlation Heatmap
        st.write("### Correlation Heatmap")
        plot_correlation_heatmap(corr_df)
        # Affordable vs Expensive Neighborhoods
        st.write("### Affordable vs Expensive Neighborhoods")
        plot_affordability_and_stress(df)
        # Economic Stress
        st.write("### Economic Stress Analysis")
        plot_boxplots(df)
        #Proportion
        st.write("### Proportion of Economically Stressed Areas")
        plot_pie_charts(df)
        
# Main Geographical Visualization Section
def geographical_visualization():
    with st.expander("Geographical Visualization 🗺️"):
        st.subheader("Geographical Visualization 🗺️")

        merged = prepare_geo_data()

        # Economic Stress Heatmap
        st.write("### Economic Stress Heatmap")
        plot_heatmap(
            geo_data=merged,
            merged=merged,
            column='Estres_Economic',
            title="Economic Stress Heatmap (Including Missing Data)",
            legend_label="Economic Stress",
            cmap='YlOrRd',
            vmin=0.1,
            vmax=0.75
        )

        # Rent-to-Income Ratio Heatmap
        st.write("### Rent-to-Income Ratio Heatmap")
        plot_heatmap(
            geo_data=merged,
            merged=merged,
            column='Ratio_Lloguer_Renda',
            title="Rent-to-Income Ratio Heatmap (Including Missing Data)",
            legend_label="Rent-to-Income Ratio",
            cmap='YlOrRd',
            vmin=0.0
        )
# PCA Analysis Section
def pca_analysis():
    with st.expander("PCA Analysis 🔍"):
        st.subheader("PCA Analysis 🔍")
        return pca_biplot(numeric_df, df["Factor_Estres"], n_components=2)

# Clustering Section
def clustering_analysis(pca_data):
    with st.expander("Clustering and Segment Profiling 🔍"):
        st.subheader("Clustering and Segment Profiling 🔍")
        cluster_analysis_pca(pca_data, n_clusters=5)

# Main Run Function
def run():
    # Data overview
    st.title("Rental Affordability Analysis - Barcelona 🏡")
    with st.expander("Click to see the dashboard overview"):
        st.write("""
            This dashboard provides an insightful analysis of Barcelona’s rental market, focusing on affordability and equity across neighborhoods. 
            It uses unsupervised methods like **Dimensionality Reduction (PCA)** and **Clustering (Agglomerative Hierarchical Clustering)** to segment neighborhoods 
            and highlight key patterns. Key analyses include the relationship between variables, statistical comparisons of inequality (e.g., **Gini Index**, 
            **Rent-to-Income Ratios**), and visualizations such as geographical heatmaps of economic stress. By using these techniques, we aim to provide an 
            easy-to-understand, data-driven view of how different areas in Barcelona fare in terms of affordability and equity in the rental market.
        """)
        st.markdown("""
            **For a full, analytically detailed breakdown of the analysis, you can view the Analytical Backbone 1 Notebook inside /resources.**
        """)
    
    # Descriptive Analysis
    descriptive_analysis()

    # Geographical Visualization
    geographical_visualization()

    # PCA Analysis
    pca_data = pca_analysis()

    # Clustering Analysis
    clustering_analysis(pca_data)

# Run the Streamlit app
if __name__ == "__main__":
    # Data to be used
    df = DuckDBManager().set_up_train_test_db().execute("SELECT * FROM UnifiedDataBasic").fetchdf().dropna()
    data = pd.read_json("resources/barcelona_barris.json").dropna()
    data['geometry'] = data['geometria_wgs84'].apply(wkt.loads)
    geo_data = gpd.GeoDataFrame(data, geometry='geometry').dropna()
    geo_data = geo_data[["nom_barri", "geometry"]]
 
    numeric_df = df[["Any", "Import_Renda_Disponible", "Import_Renda_Bruta", "Import_Renda_Neta", "Index_Gini", "Distribucio_P80_20", "Edat_Mitjana", "Preu_mitja", "Recompte", "Preu_mitja_Anual", "Ratio_Lloguer_Renda", "Estres_Economic", "Lloguer_Historic_mitja"]]
    # Numerically relevant data for correlation
    corr_df = df[["Any", "Import_Renda_Disponible", "Import_Renda_Bruta", "Import_Renda_Neta", "Index_Gini", "Distribucio_P80_20", "Edat_Mitjana", "Preu_mitja", "Recompte"]]
    # Load and prepare data
    estres_df = df[["Nom_Barri", "Estres_Economic"]].groupby("Nom_Barri").mean().reset_index()
    ratio_df = df[["Nom_Barri", "Ratio_Lloguer_Renda"]].groupby("Nom_Barri").mean().reset_index()
    run()
