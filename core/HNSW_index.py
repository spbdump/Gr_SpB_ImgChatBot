import numpy as np

import os
import json
import glob

# Enable logging
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import nmslib


MIN_FEATURES = 700
MAX_INDEX_SIZE = 1500
SIFT_DESC_SIZE = 128

def load_index(path):
    index = nmslib.init(method='hnsw', space='l2')

    logger.info(f"Load index {path}")
    index.loadIndex(path)
    logger.info(f"Index was loaded: {path}")

    return index

def create_index():
    index = nmslib.init(method='hnsw', space='l2')
    index.createIndex(print_progress=False)
    return index

def build_index_from_exist(path_to_index :str, 
                           chunk_size: int, 
                           get_desc_chunk_func):
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

    # take only part of desc to improve performance
    #query_desc = np.array(q_desc[:500], dtype=np.float32) 

    logger.info("Search neighbors descriptors in Index")
    neighbors_data = index.knnQuery(q_desc.reshape(-1), k=k) # reurn tuple of indexies and distances

    logger.info("%d neighbors was found", len(neighbors_data[0]))

    return neighbors_data[0]

def add_desc_to_index(path_to_index, desc: np.ndarray, id:int, index):

    logger.info("Add data point tp index: %s", path_to_index)
    index.addDataPoint(id, desc.reshape(-1))
    
    index.createIndex(print_progress=True)
    index.saveIndex(path_to_index, save_data=False)
    logger.info("Index were updated")
    
    return True

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


def add_data_batch(prefix_path:str, index_name:str, index_size, data:np.ndarray):
    path_to_index = prefix_path + index_name

    index =  nmslib.init(method='hnsw', space='l2')
    if os.path.exists(path_to_index):
        index.loadIndex(path_to_index)
    
    ids = list( range(index_size + data.shape[0]) )
    index.addDataPointBatch(data, ids)
    
    index.createIndex(print_progress=True)
    index.saveIndex(path_to_index, save_data=False)