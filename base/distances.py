import numpy as np

def euclidean_distance(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return np.sqrt(np.sum((vector1 - vector2) ** 2))

def manhattan_distance(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return np.sum(np.absolute(vector1 - vector2))

def absolute_distance(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return np.sum(np.absolute(vector1-vector2))
