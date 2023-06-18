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
import matplotlib.transforms as transforms
import warnings
import os
import pickle
from utils import *
import jarvispatrick

warnings.filterwarnings("ignore", category=FutureWarning)


# algorithm is a clustering algorithm we want to use
# method - global, local, hybrid
def compute_centroids(
    algorithm: Literal["kmeans", "agglomerative", "dbscan"],
    method: Literal["global", "local"],
    x_data: np.ndarray,
    y_data: np.ndarray,
    n_centroids_local: int = 10,
    n_centroids_global: int = 10,
    epsilon: float = 1.0,
    min_samples: int = 1,
):
    if algorithm == "kmeans" and method == "global":
        centroids, cluster_labels = local_global_kmeans(
            x_data, n_centroids_global, True
        )
    elif algorithm == "kmeans" and method == "local":
        centroids, cluster_labels = local_kmeans(x_data, y_data, n_centroids_local)
    elif algorithm == "agglomerative" and method == "global":
        centroids, cluster_labels = global_agglomerative(x_data, n_centroids_global)
    elif algorithm == "agglomerative" and method == "local":
        centroids, cluster_labels = local_agglomerative(
            x_data, y_data, n_centroids_local
        )
    elif algorithm == "dbscan" and method == "global":
        centroids, cluster_labels = global_dbscan(x_data, epsilon, min_samples)
    elif algorithm == "dbscan" and method == "local":
        centroids, cluster_labels = local_dbscan(x_data, y_data, epsilon, min_samples)
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


def local_global_kmeans(
    x_data: np.ndarray, n_centroids, only_global: Literal[True, False]
):
    if (
        only_global
    ):  # if we want to compute global centroids we are going to save the results in a file
        centroids_folder, labels_folder = create_algorithm_directory("kmeans", "global")
        centroids_file = os.path.join(centroids_folder, f"centroids_{n_centroids}.npy")
        labels_file = os.path.join(labels_folder, f"labels_{n_centroids}.npy")

        if (
            os.path.exists(centroids_file)
            and os.path.exists(labels_file)
            and os.path.getsize(centroids_file) > 0
            and os.path.getsize(labels_file) > 0
        ):
            centroids = np.load(centroids_file)
            labels = np.load(labels_file)
        else:
            centroids, labels = global_kmeans(x_data, n_centroids)

            np.save(centroids_file, centroids)
            np.save(labels_file, labels)
    else:
        centroids, labels = global_kmeans(x_data, n_centroids)

    return centroids, labels


# n_centroids here is a number of centroids we want to find in each class
def local_kmeans(x_data: np.ndarray, y_data: np.ndarray, n_centroids):
    centroids_folder, labels_folder = create_algorithm_directory("kmeans", "local")

    centroids = []
    cluster_labels = []

    for label in set(y_data):
        centroid_file = os.path.join(
            centroids_folder, f"kmeans_local_{label}_{n_centroids}_centroids.npy"
        )
        labels_file = os.path.join(
            labels_folder, f"kmeans_local_{label}_{n_centroids}_labels.npy"
        )

        if (
            os.path.exists(centroid_file)
            and os.path.exists(labels_file)
            and os.path.getsize(centroid_file) > 0
            and os.path.getsize(labels_file) > 0
        ):
            centroid = np.load(centroid_file)
            labels = np.load(labels_file)
        else:
            centroid, labels = local_global_kmeans(
                x_data[y_data == label], n_centroids, False
            )
            np.save(centroid_file, centroid)
            np.save(labels_file, labels)

        centroids.append(centroid)
        cluster_labels.append(labels + len(cluster_labels) * n_centroids)

    centroids = np.concatenate(centroids, axis=0)
    cluster_labels = np.concatenate(cluster_labels, axis=0)

    return centroids, cluster_labels


def global_agglomerative(x_data, n_centroids):
    centroids_folder, labels_folder = create_algorithm_directory(
        "agglomerative", "global"
    )
    centroids_file = os.path.join(centroids_folder, f"centroids_{n_centroids}.npy")
    labels_file = os.path.join(labels_folder, f"labels_{n_centroids}.npy")

    if (
        os.path.exists(centroids_file)
        and os.path.exists(labels_file)
        and os.path.getsize(centroids_file) > 0
        and os.path.getsize(labels_file) > 0
    ):
        centroids = np.load(centroids_file)
        labels = np.load(labels_file)
    else:
        agglomerative = AgglomerativeClustering(n_clusters=n_centroids)
        labels = agglomerative.fit_predict(x_data)
        centroids = np.array(
            [np.mean(x_data[labels == i], axis=0) for i in range(n_centroids)]
        )

        np.save(centroids_file, centroids)
        np.save(labels_file, labels)

    return centroids, labels


def local_agglomerative(x_data, y_data, n_centroids):
    centroids_folder, labels_folder = create_algorithm_directory(
        "agglomerative", "local"
    )

    centroids = []
    cluster_labels = []

    for label in set(y_data):
        centroid_file = os.path.join(
            centroids_folder, f"agglomerative_local_{label}_{n_centroids}_centroids.npy"
        )
        labels_file = os.path.join(
            labels_folder, f"agglomerative_local_{label}_{n_centroids}_labels.npy"
        )

        if (
            os.path.exists(centroid_file)
            and os.path.exists(labels_file)
            and os.path.getsize(centroid_file) > 0
            and os.path.getsize(labels_file) > 0
        ):
            centroid = np.load(centroid_file)
            labels = np.load(labels_file)
        else:
            x_labeled = x_data[y_data == label]
            agglomerative = AgglomerativeClustering(n_clusters=n_centroids)
            labels = agglomerative.fit_predict(x_labeled)
            centroid = np.array(
                [np.mean(x_labeled[labels == i], axis=0) for i in range(n_centroids)]
            )

            np.save(centroid_file, centroid)
            np.save(labels_file, labels)

        centroids.append(centroid)
        cluster_labels.append(labels + len(cluster_labels) * n_centroids)

    centroids = np.concatenate(centroids, axis=0)
    cluster_labels = np.concatenate(cluster_labels, axis=0)

    return centroids, cluster_labels


def global_dbscan(x_data, epsilon, min_samples):
    centroids_folder, labels_folder = create_algorithm_directory("dbscan", "global")
    centroids_file = os.path.join(centroids_folder, "centroids.npy")
    labels_file = os.path.join(labels_folder, "labels.npy")

    if (
        os.path.exists(centroids_file)
        and os.path.exists(labels_file)
        and os.path.getsize(centroids_file) > 0
        and os.path.getsize(labels_file) > 0
    ):
        centroids = np.load(centroids_file)
        labels = np.load(labels_file)
    else:
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        labels = dbscan.fit_predict(x_data)
        unique_labels = [label for label in np.unique(labels) if label != -1]
        centroids = [
            np.mean(x_data[labels == label], axis=0) for label in unique_labels
        ]
        np.save(centroids_file, centroids)
        np.save(labels_file, labels)

    return centroids, labels


def local_dbscan(x_data, y_data, epsilon, min_samples):
    centroids_folder, labels_folder = create_algorithm_directory("dbscan", "local")

    centroids_file = os.path.join(centroids_folder, "centroids.npy")
    labels_file = os.path.join(labels_folder, "labels.npy")

    if (
        os.path.exists(centroids_file)
        and os.path.exists(labels_file)
        and os.path.getsize(centroids_file) > 0
        and os.path.getsize(labels_file) > 0
    ):
        centroids = np.load(centroids_file)
        cluster_labels = np.load(labels_file)
    else:
        centroids = []
        cluster_labels = []

        for idx, label in enumerate(set(y_data)):
            x_labeled = x_data[y_data == label]
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
            labels_local = dbscan.fit_predict(x_labeled)
            unique_labels = [label for label in np.unique(labels_local) if label != -1]
            centroids.extend(
                [
                    np.mean(x_labeled[labels_local == label], axis=0)
                    for label in unique_labels
                ]
            )
            cluster_labels.extend(labels_local + idx * len(unique_labels))

        centroids = np.array(centroids)
        cluster_labels = np.array(cluster_labels)

        np.save(centroids_file, centroids)
        np.save(labels_file, cluster_labels)

    return centroids, cluster_labels


def measure_distances(
    x_data: np.ndarray,
    centroids: np.ndarray,
    algorithm: str,
    method: str,
    n_centroids: int,
    metric: str = "euclidean",
):
    n_centroids = int(n_centroids)
    distances_file = os.path.join(
        create_directory(algorithm, method, "distances"),
        f"distances_{method}_{n_centroids}.pkl",
    )
    if os.path.exists(distances_file) and os.path.getsize(distances_file) > 0:
        distances = pickle.load(open(distances_file, "rb"))
    else:
        distances = np.empty((len(x_data), len(centroids)))

        for i, point in enumerate(x_data):
            for j, centroid in enumerate(centroids):
                if metric == "euclidean":
                    distance = np.linalg.norm(point - centroid)
                elif metric == "manhattan":
                    distance = np.sum(np.abs(point - centroid))
                elif metric == "chebyshev":
                    distance = np.max(np.abs(point - centroid))

                distances[i, j] = distance

        pickle.dump(distances, open(distances_file, "wb"))

    return distances


def tsne_algorithms(
    x_data: np.ndarray,
    cluster_labels: np.ndarray,
    algorithms_name: str,
    method: str,
    n_centroids: int,
    ax: int,
):
    n_centroids = int(n_centroids)

    tsne_file = os.path.join(
        create_directory(algorithms_name, method, "data", "tsne"),
        f"tsne_{method}_{n_centroids}.pkl",
    )
    tsne_image_file = os.path.join(
        create_directory(algorithms_name, method, "images", "tsne"),
        f"tsne_{method}_{n_centroids}.png",
    )

    if os.path.exists(tsne_file) and os.path.getsize(tsne_file) > 0:
        embedded_data = pickle.load(open(tsne_file, "rb"))
    else:
        tsne = TSNE(n_components=2, random_state=42)
        embedded_data = tsne.fit_transform(x_data)

        pickle.dump(embedded_data, open(tsne_file, "wb"))

    scatter = ax.scatter(
        embedded_data[:, 0],
        embedded_data[:, 1],
        c=cluster_labels,
        cmap="tab10",
        vmin=min(cluster_labels),
        vmax=max(cluster_labels),
        s=20,
        alpha=0.5,
    )

    legend_labels = [f"Cluster {label}" for label in np.unique(cluster_labels)]
    ax.legend(
        handles=scatter.legend_elements()[0], labels=legend_labels, loc="upper right"
    )

    if method == "global":
        ax.set_title(f"{algorithms_name} with {n_centroids} global centroids")
    else:
        ax.set_title(
            f"{algorithms_name} with {n_centroids} local centroids per cluster"
        )

    # zapisujemy wybrany obszar calego obrazka
    if method == "global":
        bbox = transforms.Bbox([[11.6, 1], [19, 9.1]])
    else:
        bbox = transforms.Bbox([[19.8, 1], [27.2, 9.1]])
    ax.figure.savefig(tsne_image_file, bbox_inches=bbox)


def tsne_clean(
    x_data: np.ndarray, cluster_labels: np.ndarray, algorithms_name: str, ax: int
):
    tsne_file = os.path.join(
        create_directory(algorithms_name, "global", "data", "tsne"), "tsne_clean.pkl"
    )
    tsne_image_file = os.path.join(
        create_directory(algorithms_name, "global", "images", "tsne"), "tsne_clean.png"
    )

    if os.path.exists(tsne_file) and os.path.getsize(tsne_file) > 0:
        embedded_data = pickle.load(open(tsne_file, "rb"))
    else:
        tsne = TSNE(n_components=2, random_state=42)
        embedded_data = tsne.fit_transform(x_data)
        pickle.dump(embedded_data, open(tsne_file, "wb"))

        # zapisujemy ponownie tylko w sciezce dla lokalnych centroid
        tsne_file = os.path.join(
            create_directory(algorithms_name, "local", "data", "tsne"), "tsne_clean.pkl"
        )
        pickle.dump(embedded_data, open(tsne_file, "wb"))

    scatter = ax.scatter(
        embedded_data[:, 0],
        embedded_data[:, 1],
        c=cluster_labels,
        cmap="tab10",
        vmin=min(cluster_labels),
        vmax=max(cluster_labels),
        s=20,
        alpha=0.5,
    )

    legend_labels = [f"Cluster {label}" for label in np.unique(cluster_labels)]
    ax.legend(
        handles=scatter.legend_elements()[0], labels=legend_labels, loc="upper right"
    )

    ax.set_title(f"no centroids")

    bbox = transforms.Bbox([[3.4, 1], [10.8, 9.1]])
    ax.figure.savefig(tsne_image_file, bbox_inches=bbox)

    # zapisujemy ponownie tylko w sciezce dla lokalnych centroid
    tsne_image_file = os.path.join(
        create_directory(algorithms_name, "local", "images", "tsne"), "tsne_clean.png"
    )
    ax.figure.savefig(tsne_image_file, bbox_inches=bbox)


def umap_algorithms(
    x_data: np.ndarray,
    cluster_labels: np.ndarray,
    algorithms_name: str,
    method: str,
    n_centroids: int,
    ax: int,
):
    n_centroids = int(n_centroids)

    umap_file = os.path.join(
        create_directory(algorithms_name, method, "data", "umap"),
        f"umap_{method}_{n_centroids}.pkl",
    )
    umap_image_file = os.path.join(
        create_directory(algorithms_name, method, "images", "umap"),
        f"umap_{method}_{n_centroids}.png",
    )

    if os.path.exists(umap_file) and os.path.getsize(umap_file) > 0:
        embedded_data = pickle.load(open(umap_file, "rb"))
    else:
        reducer = umap.UMAP(random_state=42)
        embedded_data = reducer.fit_transform(x_data)

        pickle.dump(embedded_data, open(umap_file, "wb"))

    scatter = ax.scatter(
        embedded_data[:, 0],
        embedded_data[:, 1],
        c=cluster_labels,
        cmap="tab10",
        vmin=min(cluster_labels),
        vmax=max(cluster_labels),
        s=20,
        alpha=0.5,
    )

    legend_labels = [f"Cluster {label}" for label in np.unique(cluster_labels)]
    ax.legend(
        handles=scatter.legend_elements()[0], labels=legend_labels, loc="upper right"
    )

    if method == "global":
        ax.set_title(f"{algorithms_name} with {n_centroids} global centroids")
    else:
        ax.set_title(
            f"{algorithms_name} with {n_centroids} local centroids per cluster"
        )

    # zapisujemy wybrany obszar calego obrazka
    if method == "global":
        bbox = transforms.Bbox([[11.6, 1], [19, 9.1]])
    else:
        bbox = transforms.Bbox([[19.8, 1], [27.2, 9.1]])
    ax.figure.savefig(umap_image_file, bbox_inches=bbox)


def umap_clean(
    x_data: np.ndarray, cluster_labels: np.ndarray, algorithms_name: str, ax: int
):
    umap_file = os.path.join(
        create_directory(algorithms_name, "global", "data", "umap"), "umap_clean.pkl"
    )
    umap_image_file = os.path.join(
        create_directory(algorithms_name, "global", "images", "umap"), "umap_clean.png"
    )

    if os.path.exists(umap_file) and os.path.getsize(umap_file) > 0:
        embedded_data = pickle.load(open(umap_file, "rb"))
    else:
        reducer = umap.UMAP(random_state=42)
        embedded_data = reducer.fit_transform(x_data)
        pickle.dump(embedded_data, open(umap_file, "wb"))

        # zapisujemy ponownie tylko w sciezce dla lokalnych centroid
        umap_file = os.path.join(
            create_directory(algorithms_name, "local", "data", "umap"),
            "umap_clean.pkl",
        )
        pickle.dump(embedded_data, open(umap_file, "wb"))

    scatter = ax.scatter(
        embedded_data[:, 0],
        embedded_data[:, 1],
        c=cluster_labels,
        cmap="tab10",
        vmin=min(cluster_labels),
        vmax=max(cluster_labels),
        s=20,
        alpha=0.5,
    )

    legend_labels = [f"Cluster {label}" for label in np.unique(cluster_labels)]
    ax.legend(
        handles=scatter.legend_elements()[0], labels=legend_labels, loc="upper right"
    )

    ax.set_title(f"no centroids")

    bbox = transforms.Bbox([[3.4, 1], [10.8, 9.1]])
    ax.figure.savefig(umap_image_file, bbox_inches=bbox)

    # zapisujemy ponownie tylko w sciezce dla lokalnych centroid
    umap_image_file = os.path.join(
        create_directory(algorithms_name, "local", "images", "umap"), "umap_clean.png"
    )
    ax.figure.savefig(umap_image_file, bbox_inches=bbox)


# # n_centroids here is a number of centroids we want to find in each class
# def local_kmeans(x_data, y_data, n_centroids):
#     centroids = []
#     cluster_labels = []

#     for label in set(y_data):
#         centroid, labels = global_kmeans(x_data[y_data == label], n_centroids)
#         centroids.append(centroid)
#         cluster_labels.append(labels + len(cluster_labels)*n_centroids)
#     centroids = np.concatenate(centroids, axis=0)
#     cluster_labels = np.concatenate(cluster_labels, axis=0)
#     return centroids, cluster_labels


# def global_agglomerative(x_data, n_centroids):
#     agglomerative = AgglomerativeClustering(n_clusters=n_centroids)
#     labels = agglomerative.fit_predict(x_data)
#     centroids = np.array([np.mean(x_data[labels == i], axis=0)
#                          for i in range(n_centroids)])
#     return centroids, labels


# def local_agglomerative(x_data, y_data, n_centroids):
#     centroids = []
#     cluster_labels = []

#     for label in set(y_data):
#         x_labeled = x_data[y_data == label]
#         agglomerative = AgglomerativeClustering(n_clusters=n_centroids)
#         labels = agglomerative.fit_predict(x_labeled)
#         centroids.extend([np.mean(x_labeled[labels == i], axis=0)
#                          for i in range(n_centroids)])
#         cluster_labels.extend(labels + len(cluster_labels)*n_centroids)

#     return centroids, cluster_labels


# def global_dbscan(x_data, epsilon, min_samples):
#     dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
#     labels = dbscan.fit_predict(x_data)
#     unique_labels = [label for label in np.unique(labels) if label != -1]
#     centroids = [np.mean(x_data[labels == label], axis=0)
#                  for label in unique_labels]
#     return centroids, labels


# def local_dbscan(x_data, y_data, epsilon, min_samples):
#     centroids = []
#     cluster_labels = []

#     for idx, label in enumerate(set(y_data)):
#         x_labeled = x_data[y_data == label]
#         dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
#         labels = dbscan.fit_predict(x_labeled)
#         unique_labels = [label for label in np.unique(labels) if label != -1]
#         centroids.extend([np.mean(x_labeled[labels == label], axis=0)
#                          for label in unique_labels])
#         cluster_labels.extend(labels + idx*len(unique_labels))

#     return centroids, cluster_labels
