manager = DuckDBManager("drive/MyDrive/Assignatures/ADSDB")
data_con = manager.set_up_train_test_db()
df = data_con.execute(f"SELECT * FROM UnifiedDataBasic").fetchdf()
#Drop unnecessary variables

#Only numerical variables
numeric_df = df[["Any", "Import_Renda_Disponible", "Import_Renda_Bruta", "Import_Renda_Neta", "Index_Gini", "Distribucio_P80_20", "Edat_Mitjana", "Preu_mitja", "Recompte", "Preu_mitja_Anual", "Ratio_Lloguer_Renda", "Estres_Economic", "Lloguer_Historic_mitja"]]
numeric_df

#Numerical not artificially created data
corr_df = df[["Any", "Import_Renda_Disponible", "Import_Renda_Bruta", "Import_Renda_Neta", "Index_Gini", "Distribucio_P80_20", "Edat_Mitjana", "Preu_mitja", "Recompte"]]

"""# 2. Descriptive Analysis

## 2.1 Variables relations

Excluding our artificially generated variables, its clear that there is a strong correlation between the 3 income kinds and also a strong correlation between rent (Preu_mitja) and incomes. Another easy insight to gather is that the average age is mildly correlated with the income, thus midly inversely correlated with Gini Index and 80/20 Distribution variable. Finally, the count of rents per neighborhood (Recompte) is also mildly positively correlated with Preu_Mitja, Gini_Index and Income.
"""

plt.figure(figsize=(8, 6))
sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title("Correlation among the variables")
plt.show()

"""## 2.2 Most affordable and Least Affordable Neighborhoods

In this section we can easily recognise the most and least affordable neighborhoods in their respective top 10 ranking followed by the aggrupation of how many censal code areas are economically stressed by each neighborhood. **For more insights this section should be contrasted with the next 2.3 section** in order to acquire familiarity with the most stressed areas in comparison to the most expensive/cheapest ones, which at first glance may seem equivalent but they in fact aren't! Not so expensive areas can be much more stressed than wealthy neighborhoods, due to various factors (People are living above their possibilities, there is economic stagnation in certain neighborhoods...)

This plots confirm the intuition of many residents of barcelona, the top 3 cheapest neighborhoods to live are *Torre Baró*, *Ciutat Meridiana* and *Baró de Viver*. In contrast, the 3 most expensive are *Vila Olimpica de Poblenou*, *Les Corts* and *Diagonal Mar i Front maritim de Poblenou*. Despite the last 3 being the most expensive they have really low count of Economically stressed censal areas. This can be due to a wide variety of reasons, but mainly that despite the rental affordability being low, the average income of those areas must be high.
"""

top_10_affordable = df.groupby("Nom_Barri")['Preu_mitja'].mean().nsmallest(10)
top_10_least_affordable = df.groupby("Nom_Barri")['Preu_mitja'].mean().nlargest(10)
stress_by_neighborhood = df[df['Factor_Estres'] == 'High']['Nom_Barri'].value_counts()

# Set up the figure and subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=False, gridspec_kw={'hspace': 0.4})

# Subplot 1: Top 10 Most Affordable Neighborhoods
sns.barplot(
    x=top_10_affordable.values,
    y=top_10_affordable.index,
    palette="YlGnBu",
    ax=axes[0]
)
axes[0].set_title("Top 10 Most Affordable Neighborhoods", fontsize=14)
axes[0].set_xlabel("Average Rent Price", fontsize=12)
axes[0].set_ylabel("Neighborhood", fontsize=12)
axes[0].bar_label(axes[0].containers[0], fmt="%.2f")

# Subplot 2: Top 10 Least Affordable Neighborhoods
sns.barplot(
    x=top_10_least_affordable.values,
    y=top_10_least_affordable.index,
    palette="Reds",
    ax=axes[1]
)
axes[1].set_title("Top 10 Least Affordable Neighborhoods", fontsize=14)
axes[1].set_xlabel("Average Rent Price", fontsize=12)
axes[1].set_ylabel("Neighborhood", fontsize=12)
axes[1].bar_label(axes[1].containers[0], fmt="%.2f")

# Subplot 3: Count of Economically Stressed Areas by Neighborhood
sns.barplot(
    x=stress_by_neighborhood.values,
    y=stress_by_neighborhood.index,
    palette="vlag",
    ax=axes[2]
)
axes[2].set_title("Count of Economically Stressed Areas by Neighborhood", fontsize=14)
axes[2].set_xlabel("Count of Economically Stressed Areas", fontsize=12)
axes[2].set_ylabel("Neighborhood", fontsize=12)
axes[2].bar_label(axes[2].containers[0], fmt="%d")

# Add a common footer
fig.suptitle("Neighborhood Insights: Affordability and Economic Stress", fontsize=16, weight='bold')
fig.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout for the suptitle

# Show the plot
plt.show()

"""## 2.3 Statistical Comparison of Neighborhoods

Distribution of Rent-to-Income Ratios by Neighborhood
The rent-to-income ratios show significant variation across neighborhoods, reflecting differences in housing affordability. Some areas consistently demonstrate low ratios, indicating better affordability, while others have higher ratios, signaling housing stress. Neighborhoods like "el Raval" and "la Barceloneta" exhibit substantial variability, suggesting a mix of affordable and unaffordable housing options within these areasOutliers with exceptionally high ratios were observed in certain neighborhoods, potentially pointing to isolated cases of extreme financial burden or luxury housing. This variation underscores economic disparities and highlights neighborhoods that may benefit from affordable housing policies or interventions.

---

Distribution of Gini Index by Neighborhood
The Gini Index reveals income inequality trends across neighborhoods. Areas like "Canyelles" and "Sant Martí de Provençals" have consistently low indices, indicating relatively equitable income distribution. In contrast, neighborhoods such as "el Barri Gòtic," "Sant Pere, Santa Caterina i la Ribera," and "la Barceloneta" show high Gini indices with significant variability, suggesting pronounced income inequality and socioeconomic segmentation. Notable outliers in neighborhoods with higher Gini indices highlight small pockets of extreme inequality.

---

Even though the affordablity analysis points out that the most expensive neighborhoods are the 3 aforementioned, none of the 3 have a statistical highly skewed distribution of economic stress. In fact, "Les Corts" has an astonishingly low economic stress and little variation in the interquartile range. Just as in every other analysis, neighborhoods like "El Raval", "Barceloneta", "Sant Pere, Santa Caterina i la Ribera", "Barri Gotic" and "Besos", have extremely high median values of economic stress with extreme variation. There are censal areas that are above a 100% of economic stress, pointing out two probable insights:

- Touristic censal areas like Barceloneta, Barri Gotic and Raval are much more expensive than the habitants average income, signaling that most residents may have properties there, or that current tenants are living beyond their possibilities.

- Areas like Besos, Trinitat Vella and Ciutat Meridiana, despite being affordable areas suffer a high economic stress and high gini index, pointing out a possible generalized poverty in those neighborhoods.


"""

median_values = df.groupby('Nom_Barri')['Ratio_Lloguer_Renda'].median().sort_values().index

# Create the boxplot, ordering the neighborhoods based on the median value
plt.figure(figsize=(12, 12), dpi=80)
sns.boxplot(data=df, x="Nom_Barri", y="Ratio_Lloguer_Renda", hue="Nom_Barri", order=median_values)
plt.title("Distribution of Rent-to-Income Ratios by Neighborhood")
plt.xticks(rotation=90)
plt.ylabel("Rent-to-Income Ratio")
plt.xlabel("Neighborhood")
plt.show()

# Calculate the median value of index_gini for each neighborhood
median_gini = df.groupby('Nom_Barri')['Index_Gini'].median().sort_values().index

# Create the boxplot for index_gini, ordering the neighborhoods based on the median value
plt.figure(figsize=(12, 12), dpi=80)
sns.boxplot(data=df, x="Nom_Barri", y="Index_Gini", hue="Nom_Barri", order=median_gini)
plt.title("Distribution of Gini Index by Neighborhood")
plt.xticks(rotation=90)
plt.ylabel("Gini Index")
plt.xlabel("Neighborhood")
plt.show()

# Calculate the median value of estres for each neighborhood
median_estres = df.groupby('Nom_Barri')['Estres_Economic'].median().sort_values().index

# Create the boxplot for estres, ordering the neighborhoods based on the median value
plt.figure(figsize=(12, 12), dpi=80)
sns.boxplot(data=df, x="Nom_Barri", y="Estres_Economic", hue="Nom_Barri", order=median_estres)
plt.title("Distribution of Stress Index by Neighborhood")
plt.xticks(rotation=90)
plt.ylabel("Stress Index")
plt.xlabel("Neighborhood")
plt.show()

"""## 2.4 Proportion of Economic Stress

Below one can see the comparison between the proportion of High vs Low economic stress, which points out that around 2/3 neighborhoods of barcelona are under economic stress. Moreover, almost 84% of neighborhoods on average spend more than 45% of its income in rent.
"""

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

plt.tight_layout()
plt.show()

"""## 2.5 Geographical visualization of Stress

From both plots, similar insights can easily be gathered: All the neighborhoods of barcelona included in our analysis are atleast somewhat economically stressed on average, where some of them like Barceloneta, Barri Gotic and Raval to name a few are experiencing an astonishing economic burden. The same logic can be applied to the Rent-to-Income ratio.

## 2.5.1 Prepare geographical data
"""

#Aggregate datasets
estres_df = df[["Nom_Barri", "Estres_Economic"]]
estres_df = estres_df.groupby("Nom_Barri").mean().reset_index()

ratio_df = df[["Nom_Barri", "Ratio_Lloguer_Renda"]]
ratio_df = ratio_df.groupby("Nom_Barri").mean().reset_index()


# Load geo data and convert geometries
data = pd.read_json("drive/MyDrive/Assignatures/ADSDB/barcelona_barris.json")
data['geometry'] = data['geometria_wgs84'].apply(wkt.loads)
geo_data = gpd.GeoDataFrame(data, geometry='geometry')
geo_data = geo_data[["nom_barri", "geometry"]]

# Join estres_df and ratio_df with geo_data
merged = geo_data.merge(estres_df, left_on='nom_barri', right_on='Nom_Barri', how='left')
merged = merged.merge(ratio_df, on='Nom_Barri', how='left')

"""## 2.5.2 Stress Heatmap"""

# Boolean placeholder
merged['has_data'] = ~merged['Estres_Economic'].isnull()

# Plot the heatmap
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Plot all neighborhoods in grey as the base layer
geo_data.plot(ax=ax, color='lightgrey', edgecolor='black', label='No Data')

# Overlay neighborhoods with data
merged[merged['has_data']].plot(
    ax=ax,
    column='Estres_Economic',
    cmap='YlOrRd',
    edgecolor='black',
    legend=True,
    vmin=0.1,
    vmax=0.75,
    legend_kwds={
        'label': "Economic Stress",
        'orientation': "vertical"
    }
)

# Add missing neighborhoods (null values) in black
merged[~merged['has_data']].plot(
    ax=ax,
    color='lightgrey',
    label='No Information'
)

# Customize plot
ax.set_title("Economic Stress Heatmap (Including Missing Data)", fontsize=16)
plt.show()

"""## 2.5.3 Rent-to-Income Ratio Heatmap"""

#Boolean placeholder
merged['has_data'] = ~merged['Ratio_Lloguer_Renda'].isnull()

# Plot the heatmap
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Plot all neighborhoods in grey as the base layer
geo_data.plot(ax=ax, color='lightgrey', edgecolor='black', label='No Data')

# Overlay neighborhoods with data
merged[merged['has_data']].plot(
    ax=ax,
    column='Ratio_Lloguer_Renda',  # Change this to 'Ratio_Lloguer_Renda' for a different metric
    cmap='YlOrRd',
    edgecolor='black',
    legend=True,
    vmin=0.0,
    legend_kwds={
        'label': "Rent-to-Income Ratio",
        'orientation': "vertical"
    }
)

# Add missing neighborhoods (null values) in black
merged[~merged['has_data']].plot(
    ax=ax,
    color='lightgrey',
    label='No Information',
    legend=True
)

# Customize plot
ax.set_title("Rent-to-Income Ratio Heatmap (Including Missing Data)", fontsize=16)
plt.show()

"""# 3. Principal Component Analysis

Following up a dimensionality reduction unsupervised technique is applied in order to extract further insights. Applying scaling is crucial for avoiding issues related to scales given the nature of PCA.

**Axis and interpretation**:

1. **PC1**: is dominated by the contributions of the Income variables (Import_Renda_Bruta, Import_Renda_Disposable...) and Preu_mitja, which have a direct relationship.

2. **PC2**: is dominated by strong contributions of Economic Stress, The Rent-to-income Ratio and the inequality indicators (Gini and P80/20). Just as expected after the correlation analysis, age has a strong inverse contribution to this component.

Therefore, the higher the value of the 1st axis, the higher the income and rent and lower the economic stress  The higher the value of the 2nd axis, the higher the economic stress, inequality and average age. Note that the inverse of both statements is true. However, there not all variables have a perfect alingment with both axes, so Economic Stress is more related to the negative diagonal (2nd to 4th quadrant).

**Patterns**: Thanks to the target variable Factor_Estres and the aforementioned interpretation of the axes, we can see that there is an extremely clear division between Low and High stress neighborhoods. The upper region (and somewhat left) region of the new dimensions is associated with high economic stress neighborhoods (High Rent-to-income, High economic stress index and high Gini_Index)

The lower right section of the new plane is populated of low economic stress neighborhoods, effectively clustering wealthier areas with higher disposable income.

Although there is some overlap in the middle, it is clear that linear discriminant methods could yield great results in chategorizing this data into whether low or high economically stressed areas.


**Outliers**: In the lower left and upper right quadrants we can observe some outliers that disrupt the almost perfect linear separation between the 2 clusters (High vs Low economic stress). The latter individuals,  show atypical demography (Average and low age), high rent prices, and mildly high inequality. However this kind of observations are a really small percentage of the overall picture.
"""

#Scale the input data
scaler = StandardScaler()
scaler.fit(numeric_df)
X = scaler.transform(numeric_df)

pca = PCA(n_components=2)
XX = pca.fit_transform(X)

def myplot(score, coeff, labels, ratios, binary_variable):
    """
    Plots the PCA biplot with given scores, coefficients, labels, and colors based on a binary variable.

    Parameters:
    - score: The PCA-transformed data (scores)
    - coeff: The PCA coefficients (loadings)
    - labels: List of variable names corresponding to the coefficients
    - binary_variable: A binary factor variable (e.g., Ratio_Lloguer_Renda)
    """
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())

    # Assign colors based on the binary variable (e.g., 0 = red, 1 = blue)
    colors = binary_variable.map({"Low": 'blue', "High": 'red'})

    plt.figure(figsize=(10, 8))

    # Plot the PCA points, colored by the binary variable
    plt.scatter(xs * scalex, ys * scaley, alpha=0.4, s=10, c=colors)

    # Add arrows and labels for the PCA loadings
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        label = labels[i] if labels else f"Var{i+1}"  # Use label if provided, else default to Var{i+1}
        plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, label, color='b', ha='center', va='center')


    plt.xlabel(f"Principal Component 1 ({100 * ratios[0]:.2f}%)")
    plt.ylabel(f"Principal Component 2 ({100 * ratios[1]:.2f}%)")
    plt.grid()
    plt.title(f"PCA Projected Data and Feature Importance ({100 * sum(ratios):.2f}%)")

    # Legend
    plt.scatter([], [], color='blue', label='Low')
    plt.scatter([], [], color='red', label='High')
    plt.legend(loc='upper right', title='Factor Estres')
    plt.show()


binary_variable = df['Factor_Estres']

myplot(XX[:, 0:2], np.transpose(pca.components_[0:2, :]), list(numeric_df.columns), ratios=pca.explained_variance_ratio_, binary_variable=binary_variable)

"""# 4. Clustering Analysis

One of the biggest concerns of our analysis lies in clustering the neighborhoods for identifying potential trends among them. The main aim is to categorize into smaller than only high or low economic stress, more informative aggroupations of neighbourhoods.

Two suitable clustering algorithms for this purpose would be K-means++ and Hierarchical Clustering. However, to not leave any pontential gains on the table we will perform a general quick exploration of other options, and use metrics like silhouette index, Davies-Bouldin index and visual assessement to determine the best fit for a fixed number of clusters. The value chosen in this case has been 5.

Given that the Principal Component Analysis produced a couple of new axes that achieve explaining almost 75% of variance, it would also be highly profitable to perform clustering on this plane that can easily be visualized.

## 4.1 Clustering Algorithms Comparison

After several executions of this exploration snippet, DBSCAN has obviously been discarded for its poor performance. Then after comparing through visual assessment and quantitative metrics, Agglomerative clustering using complete linkage seems to have performed the best. K-means comes second but due to the lack of elbow on the inertia plot, it does seem advisable to choose this algorithm.
"""

num_clusters = 5
linkage_method = "complete"
clustering_algorithms = {
    'K-Means': KMeans(n_clusters=num_clusters, random_state=42, max_iter=1000),
    'Gaussian Mixture': GaussianMixture(n_components=num_clusters, random_state=42),
    'DBSCAN': DBSCAN(eps=0.15, min_samples=10),
    'Agglomerative': AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_method),
}

# Store results for comparison
clustering_results = {}
scores = {}

for name, algo in clustering_algorithms.items():
    print(f"Running {name}...")

    if name == 'Gaussian Mixture':
        cluster_labels = algo.fit_predict(XX)
    else:
        cluster_labels = algo.fit(XX).labels_

    clustering_results[name] = cluster_labels

    silhouette = silhouette_score(XX, cluster_labels)
    db_score = davies_bouldin_score(XX, cluster_labels)
    scores[name] = {'Silhouette Score': silhouette, 'Davies-Bouldin Score': db_score}

# Compare Scores
print("\n=== Clustering Metrics Comparison ===")
for algo, metrics in scores.items():
    print(f"{algo}: Silhouette = {metrics['Silhouette Score']:.2f}, Davies-Bouldin = {metrics['Davies-Bouldin Score']:.2f}")

# Visualize Results
fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharex=True, sharey=True)

for idx, (name, labels) in enumerate(clustering_results.items()):
    axes[idx].scatter(XX[:, 0], XX[:, 1], c=labels, cmap='tab10', alpha=0.7)
    axes[idx].set_title(name)
    axes[idx].set_xlabel('PCA1')
    axes[idx].set_ylabel('PCA2')

plt.tight_layout()
plt.show()

"""## 4.2 Agglomerative Hierarchical Clustering"""

# Perform hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=5, linkage="complete")
labels = hierarchical.fit_predict(XX)

linked = linkage(X, 'complete')

plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='level')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Calculate silhouette score for hierarchical clustering
silhouette_hierarchical = silhouette_score(XX, labels)
print(f"Silhouette Score for Hierarchical Clustering: {silhouette_hierarchical}")
df['Cluster'] = labels

"""## 4.3 Cluster Profiling and Analysis

Finally, an extensive profiling and analysis of the clusters yielded by the chosen algorithm will be presented, to capture trends and patterns among the clusters. To perform an easy profiling, we have somewhat replicated the FactoMineR function catdes/condes below.
### Cluster 0: Moderate-Income, Diverse Rent Prices, Older Residents
### Cluster 1: Lower-Income, High Rent, Younger Residents
### Cluster 2: High-Income, High Rent, Diverse Neighborhoods
### Cluster 3: Low-Income, Affordable Rent, Younger Residents
### Cluster 4: Low-Income, Low Rent, Older Residents
"""
# Plotting the results for Agglomerative Clustering
num_clusters = 5
agg_clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage="complete")
agg_labels = agg_clustering.fit_predict(XX)


plt.figure(figsize=(12, 8))
scatter = plt.scatter(XX[:, 0], XX[:, 1], c=agg_labels, cmap='tab10', alpha=0.7)

# Labeling clusters with markers
titles = ["Cluster 0: Moderate-Income, Diverse Rent Prices, Old Residents",
          "Cluster 1: Low-Income, High Rent, Young Residents",
          "Cluster 2: High Income, High Rent, Diverse Neighborhoods",
          "Cluster 3: Low Income, Mid Rent, Young Residents",
          "Cluster 4: Low Income, Low Rent, Old Residents"]

for i in range(num_clusters):
    # Calculate the mean position for each cluster
    cluster_center_x = np.mean(XX[agg_labels == i, 0])
    cluster_center_y = np.mean(XX[agg_labels == i, 1])

    # Add a marker
    plt.scatter(cluster_center_x, cluster_center_y, color='black', marker='x', s=100, edgecolor='white', linewidth=2)

    # Add the text label
    plt.text(cluster_center_x, cluster_center_y, titles[i],
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=12,
             color='black',
             weight='bold')

plt.title('Agglomerative Clustering - PCA Projection')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()