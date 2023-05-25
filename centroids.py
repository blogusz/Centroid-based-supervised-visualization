from sklearn.cluster import KMeans
import numpy as np


# algorithm is a clustering algorithm we want to use
# method - global, local, hybrid
def compute_centroids(algorithm, method, x_data, y_data, n_centroids_local, n_centroids_global):
    centroids=[]

    if algorithm=="kmeans" and method=="global":
        centroids=global_kmeans(x_data, n_centroids_global)
    elif algorithm=="kmeans" and method=="local":
        centroids=local_kmeans(x_data, y_data, n_centroids_local)
    else:
        print("Not defined yet")
    return np.array(centroids)


# n_centroids is a number of centroids we want to find in whole dataset
def global_kmeans(x_data, n_centroids):
    kmeans = KMeans(n_clusters=n_centroids)
    kmeans.fit(x_data)
    centroids = kmeans.cluster_centers_
    return centroids

# n_centroids here is a number of centroids we want to find in each class
def local_kmeans(x_data, y_data, n_centroids):
    centroids=[]

    for label in set(y_data):
        centroids.append(global_kmeans(x_data[y_data==label], n_centroids))
    centroids=np.concatenate(centroids, axis=0)
    return centroids
