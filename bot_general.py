import os

from img_proccessing import compare_sift_descriprtors, get_image_data
from HNSW_index import load_index, get_neighbors_desc_indexes, add_desc_to_index
from file_descriptor_utils import read_specific_rows_from_file
from sqlight_storage import store_img_data, get_context, get_last_img_record


import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def find_image_in_index(path_to_index, q_desc, nfeatures=700):

    # path_to_index - should contain id, nfeatures, desc size
    descriptors_file = "./descriptors/test_desc.npy"

    nfeatures_to_cmp = 600
    desc_size = 128

    if not os.path.exists(path_to_index):
        logger.error("No such index %s:", path_to_index)

    index = load_index(path_to_index)

    if q_desc.shape[0] != nfeatures:
        logger.error("Wrong shape descriptor")

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
        img_id = idx
        if compare_sift_descriprtors(q_desc, in_desc, 0.7) == True:
            res_img_id_list.append( img_id )

    return res_img_id_list


def find_image_in_indexes(path_to_img, path_to_indexes, nfeatures):
    indexes_list = get_list_indexes(path_to_indexes)
    img_data = get_image_data(path_to_img, nfeatures)
    q_desc = img_data.descriptor

    res_ids = []
    for _, path in indexes_list:
        ids = find_image_in_index(path, q_desc)
        res_ids.append(ids)
    
    return res_ids, q_desc


def update_index(desc):
    # append desc to index
    index_path = get_last_index_path()
    add_desc_to_index(index_path, desc)
    # save message id, img name, img id, index id to database


def save_img_data(dest: str, img_data):
    store_img_data(dest, [img_data])


def get_message_id(img_id):

    return -1

def get_chat_ctx(chat_id):
    return get_context(chat_id)

def get_next_img_id():
