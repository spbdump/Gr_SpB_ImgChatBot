import nmslib
import numpy as np
import db_utils

import os
import json
import glob


# Enable logging
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
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



def load_hnsw_indexies(dir_prefix=""):
    if len(hnsw_indexies) != 0:
        logger.info("Load indexies already loaded!")
        return
    path = dir_prefix + './indexies'
    for filename in os.listdir(path):
        if filename.endswith(".bin"):
            hnsw_indexies_paths.append(path + filename)

    for path in hnsw_indexies_paths:
        index = nmslib.init(method='hnsw', space='l2')

        logger.info("Load index %s", path)
        index.loadIndex(path)
        hnsw_indexies.append(index)

    logger.info("Indexies was loaded: %d", len(hnsw_indexies))


def load_index(path):
    index = nmslib.init(method='hnsw', space='l2')

    logger.info("Load index %s", path)
    index.loadIndex(path)
    logger.info("Indexies was loaded: %d", len(hnsw_indexies))

    return index

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

def add_item_to_index(descriptors, ids, i_index):
    index = hnsw_indexies[i_index]
    # descriptors = db_utils.get_desc_batch(cnt_proccessed_desc + batch_offset, batch_part_size)
    # ids = list( range( batch_offset, batch_offset + batch_part_size + 1 ) )
    index.addDataPointBatch(descriptors, ids)


## new build code

def build_index_from_exist(path_to_index :str, 
                        #    index_size: int,
                           chunk_size: int, 
                           get_desc_chunk_func):
    # index_max_size = 10000
    # if index_size > index_max_size:
    #     logger.error("Input data too big")
    #     return

    logger.info("Populate the index with descriptors")

    offset = 0
    index_size = 0
    index = nmslib.init(method='hnsw', space='l2')

    for chunk in get_desc_chunk_func():

        chunk_rows = chunk.shape[0]
        if chunk_rows == 0:
            break
        else:
            ids = list( range( offset, offset + chunk_rows ) )

        # should be array(128, nfeatures*cnt_descriptors)
        index.addDataPointBatch(chunk, ids)
        offset = offset + chunk_size
        index_size = index_size + chunk_rows

    logger.info("Index has %d descriptors", offset)
    logger.info("Start building batch index")
    logger.info("Count documents: %d", index_size)

    index.createIndex(print_progress=True)

    # Save a meta index, but without data!
    logger.info("Save index to %s", path_to_index)
    index.saveIndex(path_to_index, save_data=False)

    logger.info("Buildiing index - Done")

    return index, index_size

def get_neighbors_desc_indexes(index, q_desc: np.ndarray, k=70):

    query_desc = np.array(q_desc, dtype=np.float32) # take only part of desc to improve performance

    logger.info("Search neighbors descriptors in Index")
    neighbors_data = index.knnQuery(query_desc.reshape(-1), k=k) # reurn tuple of indexies and distances
    
    # indexies_set = set(neighbors_data[0])
    # save indexies of descriptors by number of hnsw index batch
    # neighbors_map[idx] = np.array(list(indexies_set), dtype=np.int32).tolist()

    logger.info("%d neighbors was found", len(neighbors_data[0]))
    # print(neighbors_map)

    return neighbors_data[0]

def add_desc_to_index(path_to_index, desc: np.ndarray):
    index = load_index(path_to_index)

    index.addDataPoint(desc)
    index.saveIndex(path_to_index, save_data=False)
    logger.info("Index were updated")


def update_index_size(index_name, add_value, metadata_file):
    all_metadata = read_all_metadata(metadata_file)

    if index_name in all_metadata:
        curr_size = all_metadata[index_name]["index_size"]
        all_metadata[index_name]["index_size"] = curr_size + add_value

        save_all_metadata(all_metadata, metadata_file)
        return True
    else:
        return False


def append_metadata(index_name, metadata, metadata_file):
    all_metadata = read_all_metadata(metadata_file)

    # Add or update the metadata for the specified index
    all_metadata[index_name] = metadata

    # Save the updated metadata back to the file
    save_all_metadata(all_metadata, metadata_file)


def update_metadata(index_name, new_metadata, metadata_file):
    all_metadata = read_all_metadata(metadata_file)

    if index_name in all_metadata:
        all_metadata[index_name].update(new_metadata)
        save_all_metadata(all_metadata, metadata_file)
        return True
    else:
        return False

def read_all_metadata(metadata_file):
    try:
        with open(metadata_file, 'r') as file:
            all_metadata = json.load(file)
        return all_metadata
    except FileNotFoundError:
        return {}

def save_all_metadata(all_metadata, metadata_file):
    with open(metadata_file, 'w') as file:
        json.dump(all_metadata, file, indent=4)

# or use db function (preferable way)
def find_index_files(directory, pattern = 'index_id_*_sz_*_nfeat_*_desc_sz_*.bin'):
    files = glob.glob(f'{directory}/{pattern}')
    file_names = [os.path.basename(file) for file in files]
    return file_names

def extract_index_info(index_name):
    parts = index_name.split('_')
    if len(parts) >= 4 and parts[0] == 'index' and parts[3] == 'sz':
        try:
            index_size = int(parts[4])
            index_id = '_'.join(parts[1:3])
            return index_size, index_id
        except ValueError:
            pass
    return -1, -1


def create_empty_index( path_to_index:str ):
    index = nmslib.init(method='hnsw', space='l2')
    index.createIndex(print_progress=True)
    logger.info("Buildiing index - Done")

    index.saveIndex(path_to_index, save_data=False)
    logger.info("Save index to %s", path_to_index)
