# Step 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Step 2: Load Iris dataset
iris = load_iris()
X = iris.data[:, :2]   # Using first two features for 2D visualization

# Step 3: Apply K-Means clustering (K = 3)
kmeans = KMeans(n_clusters=3, random_state=0)

# Step 4: Fit the model
kmeans.fit(X)

# Step 5: Get cluster labels
labels = kmeans.labels_

# Step 6: Get centroids
centroids = kmeans.cluster_centers_

# Step 7: Visualize clusters
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')

# Step 8: Plot centroids
plt.scatter(centroids[:,0], centroids[:,1], 
            s=200, c='red', marker='X')

plt.title("K-Means Clustering on Iris Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.show()