import nmslib
import numpy as np
import random

def test_find_desc():

    data = np.random.randn(100, 128).astype(np.float32)

    index = nmslib.init(method='hnsw', space='l2')

    for id, desc in enumerate(data):
        index.addDataPoint(id, desc)

    # expected_id = random.randint(0, data.shape[0]-1)
    # ids, distances = index.knnQuery(data[expected_id], k=4)

    index.createIndex(print_progress=True)

    # # query for the nearest neighbours of the first datapoint
    ids, distances = index.knnQuery(data[0], k=10)

    print(ids)

