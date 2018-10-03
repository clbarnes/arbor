import numpy as np


def euclidean_distance(coord1, coord2):
    return abs(np.linalg.norm(np.asarray(coord1) - coord2))
