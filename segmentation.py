import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load the Dataset
# Make sure 'Mall_Customers.csv' is in your working directory
df = pd.read_csv('Mall_Customers.csv')

print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# 2. Feature Selection
# Selecting 'Annual Income (k$)' and 'Spending Score (1-100)' for segmentation
X = df.iloc[:, [3, 4]].values

# 3. Scale Features (Optional but recommended for distance-based models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Find Optimal Number of Clusters using the Elbow Method
wcss = []  # Within-Cluster Sum of Square
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Based on the Elbow plot, the optimal number of clusters is K = 5

# 5. Train K-Means Model on K = 5
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Add cluster labels to original dataframe
df['Cluster'] = y_kmeans

# 6. Visualize the Segments
plt.figure(figsize=(10, 6))

colors = ['red', 'blue', 'green', 'cyan', 'magenta']
labels = [
    'Low Income, Low Spending (Sensible)',
    'Low Income, High Spending (Careless)',
    'Target / High Income, High Spending',
    'High Income, Low Spending (Miser)',
    'Average Income & Spending (Standard)'
]

for i in range(optimal_k):
    plt.scatter(
        X[y_kmeans == i, 0], 
        X[y_kmeans == i, 1], 
        s=100, 
        c=colors[i], 
        label=labels[i]
    )

# Plot Centroids (converting back to original scale)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centroids[:, 0], 
    centroids[:, 1], 
    s=300, 
    c='yellow', 
    marker='*', 
    edgecolor='black', 
    label='Centroids'
)

plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Segment Analysis Summary
print("\nCluster Aggregation Summary:")
cluster_summary = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Customer Count'})

print(cluster_summary)