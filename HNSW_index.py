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


def get_neighbors_descriptors(desc: np.ndarray, k:int =100):

    query_desc = np.array(desc[:300], dtype=np.float32) # take only part of desc to improve performance
    neighbors_map = {}

    logger.info("Search neighbors descriptors in Index")
    for idx, index in enumerate(hnsw_indexies):
        neighbors_data = index.knnQuery(query_desc.reshape(-1), k=k) # reurn tuple of indexies and distances
        indexies_set = set(neighbors_data[0])
        neighbors_map[idx] = np.array(list(indexies_set), dtype=np.int32).tolist() # save indexies of descriptors by number of hnsw index batch

    logger.info("Reciving neighbor indexies from database")
    print(neighbors_map)
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

        if os.path.isfile(file_name):
            batch_idx += 1
            cnt_proccessed_desc += BATCH_SIZE
            logger.info("Skip build Index. Index with name %s already exist", file_name)
            continue

        while batch_offset < BATCH_SIZE:
            descriptors = db_utils.get_desc_batch(cnt_proccessed_desc + batch_offset, batch_part_size)
            
            if len(descriptors) == 0:
                break
            
            ids = list( range( batch_offset, batch_offset + batch_part_size + 1 ) )
            index.addDataPointBatch(descriptors, ids) # should be array(128, nfeatures*cnt_descriptors)
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
        cnt_proccessed_desc += BATCH_SIZE

        logger.info("Buildiing batch index Done")


    logger.info("Buildiing index - Done")
    logger.info("%d HNSW indexies was builted!", len(hnsw_indexies))

