import nmslib
import numpy as np
import db_utils

# Enable logging
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 150
hnsw_indexies = []

def get_neighbors_descriptors(desc: np.ndarray, k:int =5):

    query_desc = desc[:1000] # take only part of desc to improve performance
    neighbors_indices = []
    neighbors = []
    for idx, index in enumerate(hnsw_indexies):
        neighbors += index.knnQueryBatch(query_desc, k=k)
        # neighbors_indices.append((idx, indices))

    logger.info("Reciving neighbor indexies from database")
    # descriptors_data = db_utils.get_desc_by_batch_indexes(neighbors_indices)

    logger.info("Recive %d indexies from database", len(descriptors_data))

    return neighbors
    

# Build the HNSW index
def build_hnsw_index():

    batch_part_size = int(BATCH_SIZE/10)
    collection_size = db_utils.get_desc_collection_size()
    cnt_proccessed_desc = 0

    logger.info("Count documents: %d", collection_size)
    logger.info("Populate the index with descriptors")

    while cnt_proccessed_desc < collection_size:
        batch_offset = 0
        index = nmslib.init(method='hnsw', space='l2')

        while batch_offset < BATCH_SIZE:
            descriptors = db_utils.get_desc_batch(batch_offset, batch_part_size)
            for desc in descriptors:
                index.addDataPointBatch(desc)

            hnsw_indexies.append(index)
            batch_offset += batch_part_size

        logger.info("Index has %d descriptors", batch_offset)
        logger.info("Start building batch index")

        index.createIndex(print_progress=True)
        hnsw_indexies.append(index)

        cnt_proccessed_desc += batch_offset
        
        logger.info("Buildiing batch index Done")


    logger.info("Buildiing index - Done")
    logger.info("%d HNSW indexies was builted!", len(hnsw_indexies))

