import os


def create_algorithm_directory(algorithm: str, method: str) -> tuple:
    centroids_folder = os.path.join(f"stored_results/{algorithm}/{method}", "centroids")
    labels_folder = os.path.join(f"stored_results/{algorithm}/{method}", "labels")

    os.makedirs(centroids_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    return centroids_folder, labels_folder


def create_directory(
    algorithm: str, method: str, type: str, umap_tsne: str = ""
) -> str:
    folder = f"stored_results/{algorithm}/{method}"
    if umap_tsne:
        folder = os.path.join(folder, umap_tsne)
    folder = os.path.join(folder, type)

    os.makedirs(folder, exist_ok=True)

    return folder
