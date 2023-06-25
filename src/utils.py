import os
from typing_extensions import Literal


# creates directiories for kmeans, agglomerative, dbscan and jp
def create_algorithm_directory(
    algorithm: Literal["kmeans", "agglomerative", "dbscan", "jp"],
    method: Literal["global", "local"],
) -> tuple:
    centroids_folder = os.path.join(
        f"../stored_results/{algorithm}/{method}", "centroids"
    )
    labels_folder = os.path.join(f"../stored_results/{algorithm}/{method}", "labels")

    os.makedirs(centroids_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    return centroids_folder, labels_folder


# creates directiories for distances, images and umap or tsne data
def create_directory(
    algorithm: Literal["kmeans", "agglomerative", "dbscan", "jp"],
    method: Literal["global", "local"],
    type: Literal["distances", "data", "images"],
    umap_tsne: Literal["umap", "tsne"] = "",
) -> str:
    folder = f"../stored_results/{algorithm}/{method}"
    if umap_tsne:
        folder = os.path.join(folder, umap_tsne)
    folder = os.path.join(folder, type)

    os.makedirs(folder, exist_ok=True)

    return folder
