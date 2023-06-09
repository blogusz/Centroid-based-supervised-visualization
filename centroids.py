from matplotlib.lines import Line2D
from metrics import LocalMetric
import numpy as np
import umap
from typing_extensions import Literal
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
import os
from utils import create_results_folders

warnings.filterwarnings("ignore", category=FutureWarning)


# algorithm is a clustering algorithm we want to use
# method - global, local, hybrid
def compute_centroids(algorithm: Literal["kmeans", "agglomerative", "dbscan"], method: Literal["global", "local"], x_data: np.ndarray, y_data: np.ndarray, n_centroids_local: int = 10, n_centroids_global: int = 10, epsilon: float = 1.0, min_samples: int = 1):
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


# zapisywanie i ladowanie wynikow w jakis sposob psuje kmeans lokalny
# # n_centroids is a number of centroids we want to find in whole dataset
# def global_kmeans(x_data, n_centroids):
#     centroids_folder, labels_folder = create_results_folders(
#         "kmeans", "global")
#     centroids_file = os.path.join(
#         centroids_folder, f"centroids_{n_centroids}.npy")
#     labels_file = os.path.join(labels_folder, f"labels_{n_centroids}.npy")

#     if os.path.exists(centroids_file) and os.path.exists(labels_file) and os.path.getsize(centroids_file) > 0 and os.path.getsize(labels_file) > 0:
#         centroids = np.load(centroids_file)
#         labels = np.load(labels_file)
#     else:
#         kmeans = KMeans(n_clusters=n_centroids)
#         kmeans.fit(x_data)
#         centroids = kmeans.cluster_centers_
#         labels = kmeans.labels_

#         np.save(centroids_file, centroids)
#         np.save(labels_file, labels)

#     return centroids, labels


# # n_centroids here is a number of centroids we want to find in each class
# def local_kmeans(x_data, y_data, n_centroids):
#     centroids_folder, labels_folder = create_results_folders("kmeans", "local")

#     centroids = []
#     cluster_labels = []

#     for label in set(y_data):
#         centroid_file = os.path.join(
#             centroids_folder, f"kmeans_local_{label}_{n_centroids}_centroids.npy")
#         labels_file = os.path.join(
#             labels_folder, f"kmeans_local_{label}_{n_centroids}_labels.npy")

#         if os.path.exists(centroid_file) and os.path.exists(labels_file) and os.path.getsize(centroid_file) > 0 and os.path.getsize(labels_file) > 0:
#             centroid = np.load(centroid_file)
#             labels = np.load(labels_file)
#         else:
#             centroid, labels = global_kmeans(
#                 x_data[y_data == label], n_centroids)
#             np.save(centroid_file, centroid)
#             np.save(labels_file, labels)

#         centroids.append(centroid)
#         cluster_labels.append(labels + len(cluster_labels) * n_centroids)

#     centroids = np.concatenate(centroids, axis=0)
#     cluster_labels = np.concatenate(cluster_labels, axis=0)

#     return centroids, cluster_labels


# def global_agglomerative(x_data, n_centroids):
#     centroids_folder, labels_folder = create_results_folders(
#         "agglomerative", "global")
#     centroids_file = os.path.join(
#         centroids_folder, f"centroids_{n_centroids}.npy")
#     labels_file = os.path.join(labels_folder, f"labels_{n_centroids}.npy")

#     if os.path.exists(centroids_file) and os.path.exists(labels_file) and os.path.getsize(centroids_file) > 0 and os.path.getsize(labels_file) > 0:
#         centroids = np.load(centroids_file)
#         labels = np.load(labels_file)
#     else:
#         agglomerative = AgglomerativeClustering(n_clusters=n_centroids)
#         labels = agglomerative.fit_predict(x_data)
#         centroids = np.array([np.mean(x_data[labels == i], axis=0)
#                               for i in range(n_centroids)])

#         np.save(centroids_file, centroids)
#         np.save(labels_file, labels)

#     return centroids, labels


# def local_agglomerative(x_data, y_data, n_centroids):
#     centroids_folder, labels_folder = create_results_folders(
#         "agglomerative", "local")

#     centroids = []
#     cluster_labels = []

#     for label in set(y_data):
#         centroid_file = os.path.join(
#             centroids_folder, f"agglomerative_local_{label}_{n_centroids}_centroids.npy")
#         labels_file = os.path.join(
#             labels_folder, f"agglomerative_local_{label}_{n_centroids}_labels.npy")

#         if os.path.exists(centroid_file) and os.path.exists(labels_file) and os.path.getsize(centroid_file) > 0 and os.path.getsize(labels_file) > 0:
#             centroid = np.load(centroid_file)
#             labels = np.load(labels_file)
#         else:
#             x_labeled = x_data[y_data == label]
#             agglomerative = AgglomerativeClustering(n_clusters=n_centroids)
#             labels = agglomerative.fit_predict(x_labeled)
#             centroid = np.array([np.mean(x_labeled[labels == i], axis=0)
#                                  for i in range(n_centroids)])

#             np.save(centroid_file, centroid)
#             np.save(labels_file, labels)

#         centroids.append(centroid)
#         cluster_labels.append(labels + len(cluster_labels) * n_centroids)

#     centroids = np.concatenate(centroids, axis=0)
#     cluster_labels = np.concatenate(cluster_labels, axis=0)

#     return centroids, cluster_labels


# def global_dbscan(x_data, epsilon, min_samples):
#     centroids_folder, labels_folder = create_results_folders(
#         "dbscan", "global")
#     centroids_file = os.path.join(centroids_folder, "centroids.npy")
#     labels_file = os.path.join(labels_folder, "labels.npy")

#     if os.path.exists(centroids_file) and os.path.exists(labels_file) and os.path.getsize(centroids_file) > 0 and os.path.getsize(labels_file) > 0:
#         centroids = np.load(centroids_file)
#         labels = np.load(labels_file)
#     else:
#         dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
#         labels = dbscan.fit_predict(x_data)
#         unique_labels = [label for label in np.unique(labels) if label != -1]
#         centroids = [np.mean(x_data[labels == label], axis=0)
#                      for label in unique_labels]
#         np.save(centroids_file, centroids)
#         np.save(labels_file, labels)

#     return centroids, labels


# def local_dbscan(x_data, y_data, epsilon, min_samples):
#     centroids_folder, labels_folder = create_results_folders(
#         "dbscan", "local")

#     centroids_file = os.path.join(centroids_folder, "centroids.npy")
#     labels_file = os.path.join(labels_folder, "labels.npy")

#     if os.path.exists(centroids_file) and os.path.exists(labels_file) and os.path.getsize(centroids_file) > 0 and os.path.getsize(labels_file) > 0:
#         centroids = np.load(centroids_file)
#         cluster_labels = np.load(labels_file)
#     else:
#         centroids = []
#         cluster_labels = []

#         for idx, label in enumerate(set(y_data)):
#             x_labeled = x_data[y_data == label]
#             dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
#             labels_local = dbscan.fit_predict(x_labeled)
#             unique_labels = [label for label in np.unique(
#                 labels_local) if label != -1]
#             centroids.extend([np.mean(x_labeled[labels_local == label], axis=0)
#                               for label in unique_labels])
#             cluster_labels.extend(labels_local + idx * len(unique_labels))

#         centroids = np.array(centroids)
#         cluster_labels = np.array(cluster_labels)

#         np.save(centroids_file, centroids)
#         np.save(labels_file, cluster_labels)

#     return centroids, cluster_labels


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


# def visualize_tsne(x_data: np.ndarray, cluster_labels: np.ndarray, centroids: np.ndarray, algorithms_name: str, method: str, ax: int):
#     combined_data = np.concatenate((x_data, centroids), axis=0)

#     tsne = TSNE(n_components=2, random_state=42)
#     embedded_data = tsne.fit_transform(combined_data)

#     embedded_x_data = embedded_data[:len(x_data)]
#     embedded_centroids = embedded_data[len(x_data):]

#     scatter = ax.scatter(
#         embedded_x_data[:, 0], embedded_x_data[:, 1], c=cluster_labels, cmap='tab10')
#     ax.scatter(embedded_centroids[:, 0],
#                embedded_centroids[:, 1], c='red', marker='x')

#     legend_labels = [f"Cluster {label}" for label in np.unique(cluster_labels)]
#     legend_labels.append("Centroids")
#     ax.legend(handles=scatter.legend_elements()[
#               0], labels=legend_labels, loc='upper right')

#     if method == "global":
#         ax.set_title(
#             f"{algorithms_name} with {len(centroids)} global centroids")
#     else:
#         ax.set_title(
#             f"{algorithms_name} with {int(len(centroids)/10)} local centroids per cluster")


def tsne_algorithms(x_data: np.ndarray, cluster_labels: np.ndarray, algorithms_name: str, method: str, n_centroids: int, ax: int):
    tsne = TSNE(n_components=2, random_state=42)
    embedded_data = tsne.fit_transform(x_data)

    scatter = ax.scatter(
        embedded_data[:, 0], embedded_data[:, 1], c=cluster_labels, cmap='tab10')
    
    legend_labels = [f"Cluster {label}" for label in np.unique(cluster_labels)]
    ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc='upper right')

    if method == "global":
        ax.set_title(
            f"{algorithms_name} with {n_centroids} global centroids")
    else:
        ax.set_title(
            f"{algorithms_name} with {int(n_centroids)/10} local centroids per cluster")


def tsne_clean(x_data: np.ndarray, cluster_labels: np.ndarray, ax: int):
    tsne = TSNE(n_components=2, random_state=42)
    embedded_data = tsne.fit_transform(x_data)

    scatter = ax.scatter(embedded_data[:, 0], embedded_data[:, 1], c=cluster_labels, cmap='tab10')
    
    legend_labels = [f"Cluster {label}" for label in np.unique(cluster_labels)]
    ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc='upper right')

    ax.set_title(f"no centroids")


# def visualize_umap(x_data: np.ndarray, cluster_labels: np.ndarray, centroids: np.ndarray, algorithms_name: str, method: str, ax: int):
#     combined_data = np.concatenate((x_data, centroids), axis=0)

#     reducer = umap.UMAP(random_state=42)
#     embedded_data = reducer.fit_transform(combined_data)

#     embedded_x_data = embedded_data[:len(x_data)]
#     embedded_centroids = embedded_data[len(x_data):]

#     unique_labels = np.unique(cluster_labels)
#     scatter = ax.scatter(
#         embedded_x_data[:, 0], embedded_x_data[:, 1], c=cluster_labels, cmap='magma', vmin=min(cluster_labels), vmax=max(cluster_labels), s=20, alpha=0.5)

#     ax.scatter(embedded_centroids[:, 0],
#                embedded_centroids[:, 1], c='red', marker='x')

#     # legend_labels = [f"Cluster {label}" for label in unique_labels]
#     # legend_labels.append("Centroids")
#     # ax.legend(handles=scatter.legend_elements(num=len(unique_labels))
#     #           [0], labels=legend_labels, loc='upper right')
#     legend_labels = [f"Cluster {label}" for label in unique_labels]
#     legend_labels.append("Centroids")
#     ax.legend(handles=scatter.legend_elements()[
#               0], labels=legend_labels, loc='upper right')

#     if method == "global":
#         ax.set_title(
#             f"{algorithms_name} with {len(centroids)} global centroids")
#     else:
#         ax.set_title(
#             f"{algorithms_name} with {len(centroids)} local centroids per cluster")

def umap_algorithms(x_data: np.ndarray, cluster_labels: np.ndarray, algorithms_name: str, method: str, n_centroids: int, ax: int):
    reducer = umap.UMAP(random_state=42)
    embedded_data = reducer.fit_transform(x_data)

    scatter = ax.scatter(
        embedded_data[:, 0], embedded_data[:, 1], c=cluster_labels, cmap='tab10', vmin=min(cluster_labels), vmax=max(cluster_labels), s=20, alpha=0.5)

    legend_labels = [f"Cluster {label}" for label in np.unique(cluster_labels)]
    ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc='upper right')

    if method == "global":
        ax.set_title(f"{algorithms_name} with {n_centroids} global centroids")
    else:
        ax.set_title(f"{algorithms_name} with {int(n_centroids)} local centroids per cluster")
        

def umap_clean(x_data: np.ndarray, cluster_labels: np.ndarray, ax: int):
    reducer = umap.UMAP(random_state=42)
    embedded_data = reducer.fit_transform(x_data)

    scatter = ax.scatter(
        embedded_data[:, 0], embedded_data[:, 1], c=cluster_labels, cmap='tab10', vmin=min(cluster_labels), vmax=max(cluster_labels), s=20, alpha=0.5)

    legend_labels = [f"Cluster {label}" for label in np.unique(cluster_labels)]
    ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc='upper right')

    ax.set_title(f"no centroids")
