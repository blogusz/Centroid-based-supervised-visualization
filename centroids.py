import numpy as np
from typing_extensions import Literal
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


# algorithm is a clustering algorithm we want to use
# method - global, local, hybrid
def compute_centroids(algorithm: Literal["kmeans", "agglomerative", "dbscan"], method: Literal["global", "local"], x_data: np.ndarray, y_data: np.ndarray, n_centroids_local: int = 10, n_centroids_global: int = 10, epsilon: float = 1.0, min_samples: int = 1):
    centroids = []

    if algorithm == "kmeans" and method == "global":
        centroids = global_kmeans(x_data, n_centroids_global)
    elif algorithm == "kmeans" and method == "local":
        centroids = local_kmeans(x_data, y_data, n_centroids_local)
    elif algorithm == "agglomerative" and method == "global":
        centroids = global_agglomerative(x_data, n_centroids_global)
    elif algorithm == "agglomerative" and method == "local":
        centroids = local_agglomerative(x_data, y_data, n_centroids_local)
    elif algorithm == "dbscan" and method == "global":
        centroids = global_dbscan(x_data, epsilon, min_samples)
    elif algorithm == "dbscan" and method == "local":
        centroids = local_dbscan(x_data, y_data, epsilon, min_samples)
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
    centroids = []

    for label in set(y_data):
        centroids.append(global_kmeans(x_data[y_data == label], n_centroids))
    centroids = np.concatenate(centroids, axis=0)
    return centroids


# chyba nie dziala prawidlowo, nawet dla jednego klastra liczylo sie 15 minut i nie starczylo mi pamieci w komputerze
def global_agglomerative(x_data, n_centroids):
    centroids = []

    agglomerative = AgglomerativeClustering(n_clusters=n_centroids)
    labels = agglomerative.fit_predict(x_data)

    print("MY TO SUKCES")

    for i in range(n_centroids):
        centroid = np.mean(x_data[labels == i], axis=0)
        centroids.append(centroid)

    return centroids


def local_agglomerative(x_data, y_data, n_centroids):
    centroids = []

    for label in set(y_data):
        x_labeled = x_data[y_data == label]
        agglomerative = AgglomerativeClustering(n_clusters=n_centroids)
        labels = agglomerative.fit_predict(x_labeled)

        for i in range(n_centroids):
            centroid = np.mean(x_labeled[labels == i], axis=0)
            centroids.append(centroid)

    return centroids


def global_dbscan(x_data, epsilon, min_samples):
    centroids = []

    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    dbscan.fit(x_data)
    labels = dbscan.labels_
    unique_labels = np.unique(labels)
    print(labels)
    print(unique_labels)

    for label in unique_labels:
        if label != -1:
            centroid = np.mean(x_data[labels == label], axis=0)
            centroids.append(centroid)

    return np.array(centroids)


def local_dbscan(x_data, y_data, epsilon, min_samples):
    centroids = []

    for label in set(y_data):
        x_labeled = x_data[y_data == label]
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        labels = np.unique(dbscan.fit_predict(x_labeled))

        for label in labels:
            if label != -1:
                centroid = np.mean(x_labeled[labels == label], axis=0)
                centroids.append(centroid)

    return np.array(centroids)
