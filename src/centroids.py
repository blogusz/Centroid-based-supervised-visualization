import numpy as np
import umap
from typing_extensions import Literal
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.transforms as transforms
import warnings
import os
import pickle
from utils import create_algorithm_directory, create_directory
from jarvis_patrick import JP

warnings.filterwarnings("ignore", category=FutureWarning)


# main function of the whole project. We choose what algorithm we want to use, in what form and how many centroids we want to find.
def compute_centroids(
    algorithm: Literal["kmeans", "agglomerative", "dbscan", "jp"],
    method: Literal["global", "local"],
    x_data: np.ndarray,
    y_data: np.ndarray,
    n_centroids_local: int = 10,
    n_centroids_global: int = 10,
    epsilon: float = 1.0,
    min_samples: int = 1,
    k: int = 100,
    kmin: int = 50,
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
    elif algorithm == "jp" and method == "global":
        centroids, cluster_labels = local_global_jp(x_data, k, kmin, True)
    elif algorithm == "jp" and method == "local":
        centroids, cluster_labels = local_jarvis_patrick(x_data, y_data, k, kmin)
    else:
        print("Not defined yet")
    return np.array(centroids), np.array(cluster_labels)


# n_centroids is a number of centroids we want to find in whole dataset
def global_kmeans(x_data: np.ndarray, n_centroids: int):
    kmeans = KMeans(n_clusters=n_centroids)
    kmeans.fit(x_data)
    centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    return centroids, cluster_labels


# it is a helper function to distinguish local and global k-means algorithm in order to avoid conflicts with read and write results
def local_global_kmeans(x_data: np.ndarray, n_centroids: int, only_global: bool):
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
    else:  # if we want to use the function to compute local centroids we dont save as its saved in local_kmeans function
        centroids, labels = global_kmeans(x_data, n_centroids)

    return centroids, labels


# n_centroids here is a number of centroids we want to find in each class
def local_kmeans(x_data: np.ndarray, y_data: np.ndarray, n_centroids: int):
    n_centroids = int(n_centroids)
    centroids_folder, labels_folder = create_algorithm_directory("kmeans", "local")
    centroids = []
    cluster_labels = []

    # compute centroids within each cluster from the input dataset.
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
            # global_kmeans can be used for each cluster separately
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


def global_agglomerative(x_data: np.ndarray, n_centroids: int):
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
        # calculate centroids as the centers of gravity of the cluster
        centroids = np.array(
            [np.mean(x_data[labels == i], axis=0) for i in range(n_centroids)]
        )

        np.save(centroids_file, centroids)
        np.save(labels_file, labels)

    return centroids, labels


def local_agglomerative(x_data: np.ndarray, y_data: np.ndarray, n_centroids: int):
    centroids_folder, labels_folder = create_algorithm_directory(
        "agglomerative", "local"
    )
    n_centroids = int(n_centroids)

    centroids = []
    cluster_labels = []

    # compute centroids within each cluster from the input dataset
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
            # calculate centroids as the centers of gravity of the cluster
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


def global_dbscan(x_data: np.ndarray, epsilon: float, min_samples: int):
    centroids_folder, labels_folder = create_algorithm_directory("dbscan", "global")
    centroids_file = os.path.join(
        centroids_folder, f"dbscan_global_{epsilon}_{min_samples}_centroids.npy"
    )
    labels_file = os.path.join(
        labels_folder, f"dbscan_global_{epsilon}_{min_samples}_labels.npy"
    )

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

        # calculate centroids as the centers of gravity of the cluster. Label '-1' means noise.
        unique_labels = [label for label in np.unique(labels) if label != -1]
        centroids = [
            np.mean(x_data[labels == label], axis=0) for label in unique_labels
        ]
        np.save(centroids_file, centroids)
        np.save(labels_file, labels)

    return centroids, labels


def local_dbscan(
    x_data: np.ndarray, y_data: np.ndarray, epsilon: float, min_samples: int
):
    centroids_folder, labels_folder = create_algorithm_directory("dbscan", "local")

    centroids_file = os.path.join(
        centroids_folder, f"dbscan_local_{epsilon}_{min_samples}_centroids.npy"
    )
    labels_file = os.path.join(
        labels_folder, f"dbscan_local_{epsilon}_{min_samples}_labels.npy"
    )

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

        # compute centroids within each cluster from the input dataset
        for idx, label in enumerate(set(y_data)):
            x_labeled = x_data[y_data == label]
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
            labels_local = dbscan.fit_predict(x_labeled)
            # calculate centroids as the centers of gravity of the cluster. Label '-1' means noise.
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


def global_jarvis_patrick(x_data: np.ndarray, k: int, kmin: int):
    jp = JP(k, kmin)
    clusters = jp.fit(x_data)
    cluster_labels = [None] * len(x_data)
    centroids = []

    for row_index, row in enumerate(clusters):
        c = np.mean(x_data[row], axis=0)
        centroids.append(list(c))
        for i in row:
            cluster_labels[i] = row_index
    return centroids, cluster_labels


# it is a helper function to distinguish local and global jarvis-patrick algorithm in order to avoid conflicts with read and write results
def local_global_jp(x_data: np.ndarray, k: int, kmin: int, only_global: bool):
    if only_global:
        centroids_folder, labels_folder = create_algorithm_directory("jp", "global")
        centroids_file = os.path.join(centroids_folder, f"centroids_{k}_{kmin}.npy")
        labels_file = os.path.join(labels_folder, f"labels_{k}_{kmin}.npy")

        if (
            os.path.exists(centroids_file)
            and os.path.exists(labels_file)
            and os.path.getsize(centroids_file) > 0
            and os.path.getsize(labels_file) > 0
        ):
            centroids = np.load(centroids_file)
            labels = np.load(labels_file, allow_pickle=True)
        else:
            centroids, labels = global_jarvis_patrick(x_data, k, kmin)

            np.save(centroids_file, centroids)
            np.save(labels_file, labels)
    else:  # if we want to use the function to compute local centroids we dont save as its saved in local_kmeans function
        centroids, labels = global_jarvis_patrick(x_data, k, kmin)

    return centroids, labels


def local_jarvis_patrick(x_data: np.ndarray, y_data: np.ndarray, k: int, kmin: int):
    centroids_folder, labels_folder = create_algorithm_directory("jp", "local")

    centroids = []
    cluster_labels = []

    for label in set(y_data):
        centroid_file = os.path.join(
            centroids_folder, f"jp_local_{label}_{k}_{kmin}_centroids.npy"
        )
        labels_file = os.path.join(
            labels_folder, f"jp_local_{label}_{k}_{kmin}_labels.npy"
        )

        if (
            os.path.exists(centroid_file)
            and os.path.exists(labels_file)
            and os.path.getsize(centroid_file) > 0
            and os.path.getsize(labels_file) > 0
        ):
            centroid = np.load(centroid_file)
            labels = np.load(labels_file, allow_pickle=True)
        else:
            # global_jarvis_patrick can be used for each cluster separately
            centroid, labels = local_global_jp(x_data[y_data == label], k, kmin, False)
            np.save(centroid_file, centroid)
            np.save(labels_file, labels)

        centroids.append(centroid)
        cluster_labels.append(labels)
        # cluster_labels.append(labels + len(cluster_labels) * n_centroids)

    centroids = np.concatenate(centroids, axis=0)
    cluster_labels = np.concatenate(cluster_labels, axis=0)

    return centroids, cluster_labels


# function used for calculating distances beetwen data points and computed centroids
def measure_distances(
    x_data: np.ndarray,
    centroids: np.ndarray,
    algorithm: Literal["kmeans", "agglomerative", "dbscan", "jp"],
    method: Literal["global", "local"],
    metric: Literal["euclidean", "manhattan", "chebyshev"] = "euclidean",
):
    n_centroids = len(centroids)
    # for local centroids, the number of centroids per cluster is equal to one tenth of the total number of centroids
    if algorithm not in ["dbscan", "jp"] and method == "local":
        n_centroids = int(n_centroids / 10)

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


# tsne used for data processed by one of our algorithms
def tsne_algorithms(
    x_data: np.ndarray,
    y_data: np.ndarray,
    algorithm: Literal["kmeans", "agglomerative", "dbscan", "jp"],
    method: Literal["global", "local"],
    ax: int,
):
    n_centroids = x_data.shape[1]
    # for local centroids, the number of centroids per cluster is equal to one tenth of the total number of centroids
    if algorithm not in ["dbscan", "jp"] and method == "local":
        n_centroids = int(n_centroids / 10)

    tsne_file = os.path.join(
        create_directory(algorithm, method, "data", "tsne"),
        f"tsne_{method}_{n_centroids}.pkl",
    )
    tsne_image_file = os.path.join(
        create_directory(algorithm, method, "images", "tsne"),
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
        c=y_data,
        cmap="tab10",
        vmin=min(y_data),
        vmax=max(y_data),
        s=20,
        alpha=0.5,
    )

    legend_labels = [f"Cluster {label}" for label in np.unique(y_data)]
    ax.legend(
        handles=scatter.legend_elements()[0], labels=legend_labels, loc="upper right"
    )

    if method == "global":
        ax.set_title(f"{algorithm} with {n_centroids} global centroids")
    elif algorithm == "dbscan":
        ax.set_title(f"{algorithm} with {n_centroids} local centroids")
    else:
        ax.set_title(f"{algorithm} with {n_centroids} local centroids per cluster")

    # saving selected part of the whole figure
    if method == "global":
        bbox = transforms.Bbox([[11.6, 1], [19, 9.1]])
    else:
        bbox = transforms.Bbox([[19.8, 1], [27.2, 9.1]])
    ax.figure.savefig(tsne_image_file, bbox_inches=bbox)


# tsne used for unaltered data
def tsne_clean(
    x_data: np.ndarray,
    y_data: np.ndarray,
    algorithm: Literal["kmeans", "agglomerative", "dbscan", "jp"],
    ax: int,
):
    tsne_file = os.path.join(
        create_directory(algorithm, "global", "data", "tsne"), "tsne_clean.pkl"
    )
    tsne_image_file = os.path.join(
        create_directory(algorithm, "global", "images", "tsne"), "tsne_clean.png"
    )

    if os.path.exists(tsne_file) and os.path.getsize(tsne_file) > 0:
        embedded_data = pickle.load(open(tsne_file, "rb"))
    else:
        tsne = TSNE(n_components=2, random_state=42)
        embedded_data = tsne.fit_transform(x_data)
        pickle.dump(embedded_data, open(tsne_file, "wb"))

        # saving figure in local centroids directory
        tsne_file = os.path.join(
            create_directory(algorithm, "local", "data", "tsne"), "tsne_clean.pkl"
        )
        pickle.dump(embedded_data, open(tsne_file, "wb"))

    scatter = ax.scatter(
        embedded_data[:, 0],
        embedded_data[:, 1],
        c=y_data,
        cmap="tab10",
        vmin=min(y_data),
        vmax=max(y_data),
        s=20,
        alpha=0.5,
    )

    legend_labels = [f"Cluster {label}" for label in np.unique(y_data)]
    ax.legend(
        handles=scatter.legend_elements()[0], labels=legend_labels, loc="upper right"
    )

    ax.set_title(f"no centroids")

    bbox = transforms.Bbox([[3.4, 1], [10.8, 9.1]])
    ax.figure.savefig(tsne_image_file, bbox_inches=bbox)

    # saving figure in global centroids directory
    tsne_image_file = os.path.join(
        create_directory(algorithm, "local", "images", "tsne"), "tsne_clean.png"
    )
    ax.figure.savefig(tsne_image_file, bbox_inches=bbox)


def umap_algorithms(
    x_data: np.ndarray,
    y_data: np.ndarray,
    algorithm: Literal["kmeans", "agglomerative", "dbscan", "jp"],
    method: Literal["global", "local"],
    ax: int,
):
    n_centroids = x_data.shape[1]
    # for local centroids, the number of centroids per cluster is equal to one tenth of the total number of centroids
    if algorithm not in ["dbscan", "jp"] and method == "local":
        n_centroids = int(n_centroids / 10)

    umap_file = os.path.join(
        create_directory(algorithm, method, "data", "umap"),
        f"umap_{method}_{n_centroids}.pkl",
    )
    umap_image_file = os.path.join(
        create_directory(algorithm, method, "images", "umap"),
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
        c=y_data,
        cmap="tab10",
        vmin=min(y_data),
        vmax=max(y_data),
        s=20,
        alpha=0.5,
    )

    legend_labels = [f"Cluster {label}" for label in np.unique(y_data)]
    ax.legend(
        handles=scatter.legend_elements()[0], labels=legend_labels, loc="upper right"
    )

    if method == "global":
        ax.set_title(f"{algorithm} with {n_centroids} global centroids")
    elif algorithm == "dbscan":
        ax.set_title(f"{algorithm} with {n_centroids} local centroids")
    else:
        ax.set_title(f"{algorithm} with {n_centroids} local centroids per cluster")

    # saving selected part of the whole figure
    if method == "global":
        bbox = transforms.Bbox([[11.6, 1], [19, 9.1]])
    else:
        bbox = transforms.Bbox([[19.8, 1], [27.2, 9.1]])
    ax.figure.savefig(umap_image_file, bbox_inches=bbox)


# umap used for unaltered data
def umap_clean(
    x_data: np.ndarray,
    y_data: np.ndarray,
    algorithm: Literal["kmeans", "agglomerative", "dbscan", "jp"],
    ax: int,
):
    umap_file = os.path.join(
        create_directory(algorithm, "global", "data", "umap"), "umap_clean.pkl"
    )
    umap_image_file = os.path.join(
        create_directory(algorithm, "global", "images", "umap"), "umap_clean.png"
    )

    if os.path.exists(umap_file) and os.path.getsize(umap_file) > 0:
        embedded_data = pickle.load(open(umap_file, "rb"))
    else:
        reducer = umap.UMAP(random_state=42)
        embedded_data = reducer.fit_transform(x_data)
        pickle.dump(embedded_data, open(umap_file, "wb"))

        # saving figure in local centroids directory
        umap_file = os.path.join(
            create_directory(algorithm, "local", "data", "umap"),
            "umap_clean.pkl",
        )
        pickle.dump(embedded_data, open(umap_file, "wb"))

    scatter = ax.scatter(
        embedded_data[:, 0],
        embedded_data[:, 1],
        c=y_data,
        cmap="tab10",
        vmin=min(y_data),
        vmax=max(y_data),
        s=20,
        alpha=0.5,
    )

    legend_labels = [f"Cluster {label}" for label in np.unique(y_data)]
    ax.legend(
        handles=scatter.legend_elements()[0], labels=legend_labels, loc="upper right"
    )

    ax.set_title(f"no centroids")

    bbox = transforms.Bbox([[3.4, 1], [10.8, 9.1]])
    ax.figure.savefig(umap_image_file, bbox_inches=bbox)

    # saving figure in global centroids directory
    umap_image_file = os.path.join(
        create_directory(algorithm, "local", "images", "umap"), "umap_clean.png"
    )
    ax.figure.savefig(umap_image_file, bbox_inches=bbox)
