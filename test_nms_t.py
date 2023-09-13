import nmslib
import numpy as np
import random

import faulthandler

print("NMSLIB version:", nmslib.__version__)

def main():

    # create a random matrix to index
    data = np.random.randn(100, 128).astype(np.float32)

    # initialize a new index, using a HNSW index on Cosine Similarity
    pos = 0
    index = nmslib.init(method='hnsw', space='l2')
    for id, desc in enumerate(data):
        pos = index.addDataPoint(id, desc)
        print("pos :", pos)

    ids = list(range(pos + 1, 100))
    pos = index.addDataPointBatch(data, ids)
    #create index
    index.createIndex(print_progress=True)
    # query for the nearest neighbours of the first datapoint
    ids, distances = index.knnQuery(data[10], k=10)

    print(ids)

    

if __name__ == "__main__":
    main()