from metrics import LocalMetric
import numpy as np
from typing_extensions import Literal
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# algorithm is a clustering algorithm we want to use
# method - global, local, hybrid
def compute_centroids(algorithm: Literal["kmeans", "agglomerative", "dbscan"], method: Literal["global", "local"], x_data: np.ndarray, y_data: np.ndarray, n_centroids_local: int = 10, n_centroids_global: int = 10, epsilon: float = 1.0, min_samples: int = 1):
    centroids = []

    if algorithm == "kmeans" and method == "global":
        centroids, cluster_labels = global_kmeans(x_data, n_centroids_global)
    elif algorithm == "kmeans" and method == "local":
        centroids, cluster_labels = local_kmeans(
            x_data, y_data, n_centroids_local)
    elif algorithm == "agglomerative" and method == "global":
        centroids, cluster_labels = global_agglomerative(
            x_data, n_centroids_global)
    elif algorithm == "agglomerative" and method == "local":
        centroids, cluster_labels = local_agglomerative(
            x_data, y_data, n_centroids_local)
    elif algorithm == "dbscan" and method == "global":
        centroids, cluster_labels = global_dbscan(x_data, epsilon, min_samples)
    elif algorithm == "dbscan" and method == "local":
        centroids, cluster_labels = local_dbscan(
            x_data, y_data, epsilon, min_samples)
    else:
        print("Not defined yet")
    return np.array(centroids), np.array(cluster_labels)


# n_centroids is a number of centroids we want to find in whole dataset
def global_kmeans(x_data, n_centroids):
    kmeans = KMeans(n_clusters=n_centroids)
    kmeans.fit(x_data)
    centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    return centroids, cluster_labels


# n_centroids here is a number of centroids we want to find in each class
def local_kmeans(x_data, y_data, n_centroids):
    centroids = []
    cluster_labels = []

    for label in set(y_data):
        centroid, labels = global_kmeans(x_data[y_data == label], n_centroids)
        centroids.append(centroid)
        cluster_labels.append(labels + len(cluster_labels)*n_centroids)
    centroids = np.concatenate(centroids, axis=0)
    cluster_labels = np.concatenate(cluster_labels, axis=0)
    return centroids, cluster_labels


def global_agglomerative(x_data, n_centroids):
    agglomerative = AgglomerativeClustering(n_clusters=n_centroids)
    labels = agglomerative.fit_predict(x_data)
    centroids = np.array([np.mean(x_data[labels == i], axis=0)
                         for i in range(n_centroids)])
    return centroids, labels


def local_agglomerative(x_data, y_data, n_centroids):
    centroids = []
    cluster_labels = []

    for label in set(y_data):
        x_labeled = x_data[y_data == label]
        agglomerative = AgglomerativeClustering(n_clusters=n_centroids)
        labels = agglomerative.fit_predict(x_labeled)
        centroids.extend([np.mean(x_labeled[labels == i], axis=0)
                         for i in range(n_centroids)])
        cluster_labels.extend(labels + len(cluster_labels)*n_centroids)

    return centroids, cluster_labels


def global_dbscan(x_data, epsilon, min_samples):
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = dbscan.fit_predict(x_data)
    unique_labels = [label for label in np.unique(labels) if label != -1]
    centroids = [np.mean(x_data[labels == label], axis=0)
                 for label in unique_labels]
    return centroids, labels


def local_dbscan(x_data, y_data, epsilon, min_samples):
    centroids = []
    cluster_labels = []

    for idx, label in enumerate(set(y_data)):
        x_labeled = x_data[y_data == label]
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        labels = dbscan.fit_predict(x_labeled)
        unique_labels = [label for label in np.unique(labels) if label != -1]
        centroids.extend([np.mean(x_labeled[labels == label], axis=0)
                         for label in unique_labels])
        cluster_labels.extend(labels + idx*len(unique_labels))

    return centroids, cluster_labels


def measure_distances(x_data: np.ndarray, centroids: np.ndarray):
    distances = np.empty((len(x_data), len(centroids)))

    for i, point in enumerate(x_data):
        for j, centroid in enumerate(centroids):
            distance = np.linalg.norm(point - centroid)
            distances[i, j] = distance

    return distances


def visualize_tsne_with_clusters(x_data: np.ndarray, cluster_labels: np.ndarray, centroids: np.ndarray):
    combined_data = np.concatenate((x_data, centroids), axis=0)

    tsne = TSNE(n_components=2, random_state=42)
    embedded_data = tsne.fit_transform(combined_data)

    embedded_x_data = embedded_data[:len(x_data)]
    embedded_centroids = embedded_data[len(x_data):]

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embedded_x_data[:, 0], embedded_x_data[:, 1], c=cluster_labels, cmap='tab10')
    plt.scatter(embedded_centroids[:, 0],
                embedded_centroids[:, 1], c='red', marker='x')

    legend_labels = [f"Cluster {label}" for label in np.unique(cluster_labels)]
    legend_labels.append("Centroids")
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)

    plt.title("t-SNE with Clusters and Centroids")
    plt.show()
