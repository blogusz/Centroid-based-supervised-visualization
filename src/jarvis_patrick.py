import numpy as np
from sklearn.neighbors import NearestNeighbors


# this is our implementation of the original Jarvis-Patrick clustering algorithm
class JP:
    def __init__(self, k: int, kmin: int):
        self.k = k + 1
        self.kmin = kmin
        self.data = None
        self.nbrs = None

    def fit(self, data: np.ndarray):
        if isinstance(data, np.ndarray) and data.ndim == 2:
            self.data = data
        else:
            raise Exception("Wrong data type")
        neigh = NearestNeighbors(n_neighbors=self.k).fit(self.data)
        nbrs_array = neigh.kneighbors(self.data, return_distance=False)
        self.nbrs = {}
        for row in nbrs_array:
            key = row[0]
            values = row[1:].tolist()
            self.nbrs[key] = values
        self.similarity = {}
        for key in self.nbrs:
            for value in self.nbrs[key]:
                if key <= value:
                    if key in self.nbrs[value]:
                        set1 = set(self.nbrs[key])
                        set2 = set(self.nbrs[value])
                        key_similarity = key, value
                        self.similarity[key_similarity] = len(set1.intersection(set2))
        self.clusters = []
        for key in self.similarity:
            if self.similarity[key] >= self.kmin:
                if len(self.clusters) != 0:
                    added = False
                    for row_index, row in enumerate(self.clusters):
                        if key[0] in row:
                            if key[1] not in row:
                                self.clusters[row_index].append(key[1])
                            added = True
                            break
                        elif key[1] in row:
                            if key[0] not in row:
                                self.clusters[row_index].append(key[0])
                            added = True
                            break
                    if not added:
                        self.clusters.append([key[0], key[1]])
                else:
                    self.clusters.append([key[0], key[1]])
        return self.clusters
