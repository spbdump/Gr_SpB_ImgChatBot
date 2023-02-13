import numpy as np
from scipy.spatial import distance

def euclidian_distance(desc1: np.ndarray, desc2: np.ndarray):
    gap = abs(desc1.shape[0] - desc2.shape[0])

    descriptor_size = desc1.shape[1]

    if gap != 0:
        padding = np.zeros((gap, descriptor_size), dtype=np.float32)
        if desc1.shape[0] < desc2.shape[0]:
            desc1 = np.concatenate((desc1, padding))
        else:
            desc2 = np.concatenate((desc2, padding))

    # reshape to 1d
    resh_query_descriptor = desc1.reshape(-1)
    resh_db_descriptor = desc2.reshape(-1)

    return distance.euclidean(resh_query_descriptor, resh_db_descriptor)