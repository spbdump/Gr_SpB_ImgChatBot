import os
import re

from img_proccessing import compare_sift_descriprtors, get_image_data
from HNSW_index import load_index, get_neighbors_desc_indexes, add_desc_to_index, find_index_files 
from HNSW_index import extract_index_info
from file_descriptor_utils import read_specific_rows_from_file
from sqlight_storage import  get_context

from sqlite_db_utils import store_img_data, get_last_image_data, update_index_size, find_msg_id, get_index_desc_pairs

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

MIN_FEATURES = 700

def find_image_in_index(prefix_path, index_name, desc_name, q_desc, nfeatures:int =MIN_FEATURES):

    # path_to_index - should contain id, nfeatures, desc size
    descriptors_file = prefix_path + desc_name
    path_to_index = prefix_path + index_name
    index_size, index_id = extract_index_info(index_name)

    if index_size == -1 or index_id == -1:
        logger.error("Wrong index data")

    nfeatures_to_cmp = nfeatures - 100
    desc_size = 128

    if q_desc.shape[0] != nfeatures:
        logger.error("Wrong shape descriptor")

    if not os.path.exists(path_to_index):
        logger.error("No such index %s:", path_to_index)

    index = load_index(path_to_index)

    desc_idx_list = get_neighbors_desc_indexes(index, q_desc, k=100)

    del index

    if len(desc_idx_list) == 0:
        logger.info("Image not foud in index %s", path_to_index)
        return []

    mapped_indices, desc_list = read_specific_rows_from_file(descriptors_file,
                                             desc_idx_list,
                                             desc_size*nfeatures)
    if len(desc_list) == 0:
        logger.error("Can't read descriptors by indexes")

    res_img_id_list = []
    for idx, desc in zip(mapped_indices, desc_list):
        in_desc = desc.reshape(nfeatures, desc_size)[:nfeatures_to_cmp]
        q_desc = q_desc[:nfeatures_to_cmp]
        img_id = index_size*index_id + idx
        if compare_sift_descriprtors(q_desc, in_desc, 0.7) == True:
            res_img_id_list.append( img_id )

    return res_img_id_list


def find_image_in_indexes(path_to_img, chat_path, nfeatures:int =MIN_FEATURES):
    index_triplets = get_index_desc_pairs(chat_path) # find_index_files(chat_path)
    img_data = get_image_data(path_to_img, nfeatures)
    q_desc = img_data.descriptor

    res_ids = []
    for index_id, index_name, desc_name in index_triplets:
        img_id_list = find_image_in_index(chat_path, index_name, desc_name, q_desc, nfeatures)
        
        if len(img_id_list):
            res_ids.append(zip(index_id, img_id_list))

    return res_ids, q_desc


def update_index(prefix_path:str, desc):
    # append desc to index
    index_data = get_last_index_data( prefix_path )
    curr_size = index_data["index_size"]
    max_size = index_data["max_index_size"]

    index_name = index_data["index_name"]
    index_path = prefix_path + index_name

    if curr_size == max_size:
        index_name = generate_new_index_name(path_to_meta)
        create_index()

    add_desc_to_index(index_path, desc)
    update_index_size(index_name, 1, path_to_meta)


def save_img_data(prefix_path: str, img_name: str, message_id: int):
    index_data = get_last_index_data( prefix_path )
    index_size = index_data["index_size"]
    index_id = index_data["index_id"]
    img_id = index_size

    img_data = {
        "index_id": index_id,
        "t_msg_id": message_id,
        "img_id": img_id,
        "img_name": img_name,
    }

    store_img_data([img_data], prefix_path )


def get_message_id(prefix_path:str, img_id: int, index_id: int):
    msg_id = find_msg_id(img_id, index_id, prefix_path)
    return msg_id

def get_chat_ctx(chat_id):
    return get_context(chat_id)

def get_next_img_id(prefix_path:str):
    index_data = get_last_index_data( prefix_path )

    index_size = index_data["index_size"]
    return index_size

def get_last_index_id(prefix_path:str):
    index_data = get_last_index_data( prefix_path )

    index_id = index_data["index_id"]
    return index_id

# def last_index_is_fullfiled(prefix_path:str):
#     index_data = get_last_index_data( prefix_path )
#     curr_size = index_data["index_size"]
#     max_size = index_data["max_index_size"]
#     return curr_size == max_size

def generate_next_img_id(prefix_path:str):
    img_data = get_last_image_data(prefix_path)
    image_name = img_data["img_name"]

    match = re.search(r'photo_(\d+)', image_name)

    image_id = -1
    if match:
        image_id = int(match.group(1))
        logger.debug("Extracted Image ID:", image_id)
    else:
        logger.debug("Image ID not found in the image name")

    return image_id + 1