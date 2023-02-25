import nmslib
import numpy as np
import db_utils

import os

# Enable logging
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 1000
hnsw_indexies = []
hnsw_indexies_paths = []

def get_neighbors_descriptors(desc: np.ndarray, k:int =5):

    query_desc = np.array(desc[:500], dtype=np.float32) # take only part of desc to improve performance
    neighbors_map = {}

    logger.info("Search neighbors descriptors in Index")
    for idx, index in enumerate(hnsw_indexies):
        neighbors_data = index.knnQueryBatch(query_desc, k=k) # reurn tuple of indexies and distances
        indexies_set = set({})
        for indexies, distances in neighbors_data:
            indexies_set = indexies_set.union(indexies)

        neighbors_map[idx] = np.array(list(indexies_set), dtype=np.int32).tolist() # save indexies of descriptors by number of hnsw index batch

    logger.info("Reciving neighbor indexies from database")
    descriptors_data = db_utils.get_desc_by_batch_indexes(neighbors_map)
    logger.info("Recive %d indexies from database", len(descriptors_data))

    return descriptors_data
    
def load_hnsw_indexies():
    if len(hnsw_indexies) != 0:
        logger.info("Load indexies already loaded!")
        return

    for filename in os.listdir('./indexies'):
        if filename.endswith(".bin"):
            hnsw_indexies_paths.append('./indexies/' + filename)

    for path in hnsw_indexies_paths:
        index = nmslib.init(method='hnsw', space='l2')

        logger.info("Load index %s", path)
        index.loadIndex(path)
        hnsw_indexies.append(index)

    logger.info("Indexies was loaded: %d", len(hnsw_indexies))

# Build the HNSW index
def build_hnsw_index():
    global BATCH_SIZE

    batch_part_size = int(BATCH_SIZE/100)
    collection_size = db_utils.get_desc_collection_size()
    cnt_proccessed_desc = 0
    batch_idx = 0

    logger.info("Count documents: %d", collection_size)
    logger.info("Populate the index with descriptors")

    if collection_size < BATCH_SIZE:
        BATCH_SIZE = collection_size

    while cnt_proccessed_desc < collection_size:
        batch_offset = 0
        index = nmslib.init(method='hnsw', space='l2')
        file_name = "./indexies/index_optim_btach_{b_idx}_sz_{sz}.bin".format(b_idx=batch_idx, sz=BATCH_SIZE)

        while batch_offset < BATCH_SIZE:
            descriptors = db_utils.get_desc_batch(batch_offset, batch_part_size)
            index.addDataPointBatch(descriptors) # should be array(128*nfeatures, cnt_descriptors)
            
            batch_offset += batch_part_size

        logger.info("Index has %d descriptors", batch_offset)
        logger.info("Start building batch index")
        index.createIndex(print_progress=True)

        hnsw_indexies.append(index)
        hnsw_indexies_paths.append(file_name)

        # Save a meta index, but no data!
        logger.info("Save index to %s", file_name)
        index.saveIndex(file_name, save_data=False)

        batch_idx += 1
        cnt_proccessed_desc += batch_offset

        logger.info("Buildiing batch index Done")


    logger.info("Buildiing index - Done")
    logger.info("%d HNSW indexies was builted!", len(hnsw_indexies))

