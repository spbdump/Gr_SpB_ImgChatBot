import os
import re

from img_proccessing import compare_sift_descriprtors, get_image_data
from HNSW_index import load_index, get_neighbors_desc_indexes, \
                       add_desc_to_index, create_empty_index, \
                       MIN_FEATURES, MAX_INDEX_SIZE, SIFT_DESC_SIZE

from file_descriptor_utils import append_array_with_same_width, \
                                  read_specific_rows_from_binfile

from sqlite_db_utils import store_img_data, get_last_image_data, \
                            update_index_size, find_msg_id, \
                            get_index_triplets, get_last_index_data, \
                            add_index_record, get_context_by_chat_id, \
                            create_index_table, create_iamge_table

from context import Context

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)



def find_image_in_index(prefix_path, index_name, desc_name, q_desc, nfeatures:int =MIN_FEATURES):
    descriptors_file = prefix_path + desc_name
    path_to_index = prefix_path + index_name

    nfeatures_to_cmp = nfeatures
    desc_size = 128

    if not os.path.exists(path_to_index):
        logger.error("No such index %s:", path_to_index)

    index = load_index(path_to_index)

    desc_idx_list = get_neighbors_desc_indexes(index, q_desc, k=5)

    del index

    if len(desc_idx_list) == 0:
        logger.info("Image not foud in index %s", path_to_index)
        return []

    desc_list = read_specific_rows_from_binfile(descriptors_file,
                                             desc_idx_list,
                                             desc_size*nfeatures)
    if len(desc_list) == 0:
        logger.error("Can't read descriptors by indexes")

    res_img_id_list = []
    for idx, desc in zip(desc_idx_list, desc_list):
        in_desc = desc.reshape(nfeatures, desc_size)[:nfeatures_to_cmp]
        q_desc = q_desc[:nfeatures_to_cmp]
        if compare_sift_descriprtors(q_desc, in_desc, 0.8) == True:
            res_img_id_list.append( idx )

    return res_img_id_list


def find_image_in_indexes(path_to_img, chat_path, nfeatures:int =MIN_FEATURES):
    img_data = get_image_data(path_to_img, nfeatures)
    q_desc = img_data.descriptor

    index_triplets = get_index_triplets(chat_path)
    if index_triplets == None:
        logger.error("Can't get indexes data")
        return [], q_desc

    if q_desc.shape[0] < nfeatures:
        logger.error("Wrong shape descriptor")
        return [], q_desc

    res_ids = []
    for index_id, index_name, desc_name in index_triplets:
        img_id_list = find_image_in_index(chat_path, index_name, desc_name, q_desc, nfeatures)
        
        if len(img_id_list):
            res_ids.append((index_id, img_id_list))

    return res_ids, q_desc

def generate_new_index_data(prefix_path: str, index_data):
    # index_data = get_last_index_data( prefix_path )

    nfeatures = index_data["nfeatures"]
    desc_size = index_data["desc_size"]
    index_size = 0
    max_size = index_data["max_size"]
    i = index_data["index_id"] + 1

    index_name = f'index_id_{i}_sz_{max_size}_nfeat_{nfeatures}_desc_sz_{desc_size}.bin'
    desc_name =  f'desc_id_{i}_sz_{max_size}_nfeat_{nfeatures}_desc_sz_{desc_size}.npy'

    index_rec = {
            "index_name": index_name,
            "desc_name" : desc_name,
            "index_id"  : i,
            "max_size"  : max_size,
            "nfeatures" : nfeatures,
            "desc_size" : desc_size,
            "index_size": index_size,
        }
    return index_rec

def update_index(desc, ctx: Context ):
    prefix_path = ctx.chat_path
    index_data = get_last_index_data( prefix_path )

    curr_size = 0
    max_size = MAX_INDEX_SIZE
    desc_size = SIFT_DESC_SIZE
    nfeatures = MIN_FEATURES

    if index_data == None:
        logger.info("Index list is empty")
        # create nessesary tables
        create_index_table(prefix_path)
        create_iamge_table(prefix_path)

        # retrive data from ctx
        max_size = ctx.max_size
        desc_size = ctx.desc_size
        nfeatures = ctx.nfeatures

        index_data = {}
        index_data["index_size"] = 0
        index_data["index_id"] = -1
        index_data["max_size"] = max_size
        index_data["desc_size"] = desc_size
        index_data["nfeatures"] = nfeatures

    else:
        curr_size = index_data["index_size"]
        max_size =  index_data["max_size"]
        desc_size = index_data["desc_size"]
        nfeatures = index_data["nfeatures"]

    if desc.shape[0] < nfeatures and desc.shape[1] == desc_size:
        logger.error("Descriptor shape isn't match. Current: ", 
                     desc.shape, ", Expected: ", [[1],[desc_size*nfeatures]])
        logger.error("Index wasn't updated")
        return False

    if curr_size == max_size or curr_size == 0:
        index_data = generate_new_index_data(prefix_path, index_data)
        index_name = index_data["index_name"]
        #create_empty_index(prefix_path + index_name)
        b_added = add_index_record(index_data, prefix_path)
        if not b_added:
            return False

    index_name = index_data["index_name"]
    desc_name = index_data["desc_name"]
    index_id = index_data["index_id"]
    index_size = index_data["index_size"]
    index_path = prefix_path + index_name

    b_added = add_desc_to_index(index_path, desc, index_size)
    if not b_added:
        return False
    
    append_array_with_same_width(prefix_path + desc_name, desc)

    b_updated = update_index_size(index_id, 1, prefix_path)
    return b_updated


def save_img_data(prefix_path: str, img_name: str, message_id: int):
    index_data = get_last_index_data( prefix_path )

    if index_data == None:
        logger.error("Can't retrive index data")
        return

    index_size = index_data["index_size"]
    index_id = index_data["index_id"]
    img_id = index_size - 1 # -1 if size updated before

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
    return get_context_by_chat_id(chat_id)

def get_next_img_id(prefix_path:str):
    index_data = get_last_index_data( prefix_path )

    if index_data == None:
        logger.error("Can't retrive index data")
        return

    index_size = index_data["index_size"]
    return index_size

def get_last_index_id(prefix_path:str):
    index_data = get_last_index_data( prefix_path )

    if index_data == None:
        logger.error("Can't retrive index data")
        return

    index_id = index_data["index_id"]
    return index_id

# def last_index_is_fullfiled(prefix_path:str):
#     index_data = get_last_index_data( prefix_path )
#     curr_size = index_data["index_size"]
#     max_size = index_data["max_size"]
#     return curr_size == max_size

def generate_next_img_id(prefix_path:str):
    img_data = get_last_image_data(prefix_path)

    if img_data == None:
        return 0

    image_name = img_data["img_name"]
    match = re.search(r'photo_(\d+)', image_name)

    image_id = -1
    if match:
        image_id = int(match.group(1))
        logger.debug("Extracted Image ID:", image_id)
    else:
        logger.debug("Image ID not found in the image name")

    return image_id + 1