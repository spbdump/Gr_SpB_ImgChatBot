import os

from img_proccessing import compare_sift_descriprtors, get_image_data
from HNSW_index import load_index, get_neighbors_desc_indexes, add_desc_to_index, find_index_files 
from HNSW_index import update_index_size, extract_index_info
from file_descriptor_utils import read_specific_rows_from_file
from sqlight_storage import store_img_data, get_context, get_last_img_record


import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def find_image_in_index(prefix_path, index_name, desc_name, q_desc, nfeatures=700):

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


def find_image_in_indexes(path_to_img, chat_path, nfeatures):
    indexes_list = find_index_files(chat_path)
    img_data = get_image_data(path_to_img, nfeatures)
    q_desc = img_data.descriptor

    res_ids = []
    for index_name, desc_name in indexes_list:
        ids = find_image_in_index(chat_path, index_name, desc_name, q_desc, nfeatures)
        res_ids.append(ids)
    
    return res_ids, q_desc


def update_index(path_to_meta, desc):
    # append desc to index
    index_path = get_last_index_path()
    index_name = ''
    add_desc_to_index(index_path, desc)
    update_index_size(index_name, 1, path_to_meta)
    # save message id, img name, img id, index id to database


def save_img_data(dest: str, img_data):
    store_img_data(dest, [img_data])


def get_message_id(img_id):

    return -1

def get_chat_ctx(chat_id):
    return get_context(chat_id)

#def get_next_img_id():
