import os


def create_results_folders(algorithm: str, method: str) -> tuple:
    results_folder = f"stored_results/{algorithm}/{method}"
    centroids_folder = os.path.join(results_folder, "centroids")
    labels_folder = os.path.join(results_folder, "labels")

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(centroids_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    return centroids_folder, labels_folder
