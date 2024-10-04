import os
import re
import shutil

from core.img_proccessing import get_image_data_sift, \
                                 get_FLANN_matcher, get_descriptor_size
from core.HNSW_index import load_index, get_neighbors_desc_indexes, \
                            MIN_FEATURES

from core.image_utils import read_image
import core.runtime_index as rni
from core.index import Index

from sqlite_db_utils import store_img_data, get_last_image_data, \
                            find_msg_id, add_ctx_record, \
                            get_index_triplets, get_last_index_data, \
                            get_context_by_chat_id, delete_ctx_record, \
                            create_image_table, create_index_table, \
                            create_ctx_table, \
                            get_index_data_for_all_chats, \
                            save_runtime_img_data, add_index_record, \
                            update_index_record, \
                            update_PATH_TO_GENERAL_DB



from model.context import Context

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_general_db():
    create_ctx_table()
    logger.debug("General db has been created")

def init(runtime_index_size:int = rni.MAX_RUNTIME_INDEX_SIZE):
    create_general_db()
    data = get_index_data_for_all_chats()
    rni.init_runtime_chat_indexes(data, runtime_index_size)

def update_DBPATH(path:str):
    update_PATH_TO_GENERAL_DB(path)

def find_image_in_index(prefix_path, index_name, q_desc, matcher,
                        nfeatures:int, desc_size:int):
    path_to_index = os.path.join( prefix_path, index_name )

    if not os.path.exists(path_to_index):
        logger.error("No such index %s:", path_to_index)

    index = load_index(path_to_index, 1000, nfeatures, desc_size)
    desc_idx_list = get_neighbors_desc_indexes(index, q_desc, k=5)
    del index

    if len(desc_idx_list) == 0:
        logger.info("Image not foud in index %s", path_to_index)
        return []

    # following code is actiualy optional
    # first id from desc_idx_list should the closest image
    return desc_idx_list[0]

    # TODO: rewrite this block
    #       now we are able to get descriptors by lables from index

    # desc_list = read_specific_rows_from_binfile(descriptors_file,
    #                                          desc_idx_list,
    #                                          desc_size*nfeatures)
    # if len(desc_list) == 0:
    #     logger.error("Can't read descriptors by indexes")

    # res_img_id_list = []
    # for idx, desc in zip(desc_idx_list, desc_list):
    #     in_desc = desc.reshape(nfeatures, desc_size)[:nfeatures_to_cmp]
    #     q_desc = q_desc[:nfeatures_to_cmp]
    #     if compare_descriprtors(q_desc, in_desc, matcher, rni.MATCH_PERCENT) == True:
    #         res_img_id_list.append( idx )

    # return res_img_id_list

def find_image_in_indexes(path_to_img, chat_path, chat_id, nfeatures:int =MIN_FEATURES):
    image = read_image(path_to_img)
    q_desc, desc_type = get_image_data_sift(image, nfeatures)
    desc_size = get_descriptor_size(desc_type)

    if q_desc.shape[0] < nfeatures:
        logger.error("Wrong shape descriptor")
        return [], q_desc

    index_triplets = get_index_triplets(chat_path)
    if index_triplets == None:
        logger.error("Can't get indexes data")
        return [], q_desc

    matcher = get_FLANN_matcher()
    res_ids = []
    for index_id, index_name, _ in index_triplets:
        img_id_list = find_image_in_index(chat_path, index_name, q_desc,
                                          matcher, nfeatures, desc_size)

        if len(img_id_list):
            res_ids.append((index_id, img_id_list))

    runtime_index = rni.get_runtime_index(chat_id)
    rt_img_id_list = runtime_index.find_image(q_desc, matcher)
    if len(rt_img_id_list):
        res_ids.append((runtime_index.RUNTIME_INDEX_ID, rt_img_id_list))

    return res_ids, q_desc

# use separate chat_path variable to be able setup initial data path
# without updatating general bot context table
def update_index(ctx: Context, desc, img_name:str, t_msg_id:int):
    if desc.shape[0] > ctx.nfeatures:
        desc = desc[:ctx.nfeatures]
    elif desc.shape[0] < ctx.nfeatures:
        return False

    index = rni.get_runtime_index(ctx.chat_id)
    if index.add_data_point(desc.reshape(-1), img_name, t_msg_id) > 0:
        last_index_data = get_last_index_data(ctx.chat_path)
        if last_index_data == None:
            last_index_data = Index(0, 0, ctx.max_size, ctx.nfeatures, ctx.desc_size)

        if last_index_data.max_size != last_index_data.index_size:
            start_img_id = last_index_data.index_size
        else:
            start_img_id = 0

        if index.is_fullfilled():
            index.dump(last_index_data, ctx.chat_path)

            update_index_record(index, ctx.chat_path)
            save_runtime_img_data(index.index_id, start_img_id, index._dumped_img_data_, ctx.chat_path)
    else:
        return False

    return True

def get_message_id(prefix_path:str, chat_id:int,  img_id: int, index_id: int):
    index = rni.get_runtime_index(chat_id)
    if index_id == index.RUNTIME_INDEX_ID:
        msg_id = index.get_t_msg_id(img_id)
        return msg_id

    msg_id = find_msg_id(img_id, index_id, prefix_path)
    return msg_id

def get_chat_ctx(chat_id):
    return get_context_by_chat_id(chat_id)

def get_next_img_id(prefix_path:str):
    index_data = get_last_index_data( prefix_path )

    if index_data == None:
        logger.error("Can't retrive index data")
        return

    index_size = index_data.index_size
    return index_size

def get_last_index_id(prefix_path:str):
    index_data = get_last_index_data( prefix_path )

    if index_data == None:
        logger.error("Can't retrive index data")
        return

    index_id = index_data.index_id
    return index_id

def generate_next_img_id(prefix_path:str):
    img_data = get_last_image_data(prefix_path)

    if img_data == None:
        return 0

    image_name = img_data.img_name
    match = re.search(r'photo_(\d+)', image_name)

    image_id = -1
    if match:
        image_id = int(match.group(1))
        logger.debug("Extracted Image ID:", image_id)
    else:
        logger.debug("Image ID not found in the image name")

    return image_id + 1

def on_remove_bot(chat_id:int, prefix_path:str):
    # remove runtime index
    rni.delete_runtime_index(chat_id)
    # remove record from context table
    chat_folder = delete_ctx_record(chat_id)
    if chat_folder == None:
        logger.error("Can't find chat folder")
        return

    # remove chat folder
    path_to_folder = prefix_path + chat_folder
    shutil.rmtree(path_to_folder)
    logger.info("Erase chat data completed")

def on_add_bot(chat_id:int, prefix_path:str, chat_name:str,
               runtime_index_size:int, max_size:int, nfeatures:int, desc_size:int):
    # create runtime index
    index = Index(0, 0, max_size, nfeatures, desc_size )
    rni.add_runtime_index(chat_id, index, runtime_index_size)
    # add record to context table
    path_to_chat = '/'+chat_name+'/'
    add_ctx_record(chat_id, index.nfeatures, index.desc_size,
                   index.max_size, path_to_chat)

    # crate chat.db and tables
    path_to_chat_folder = os.path.join(prefix_path, path_to_chat)
    if not os.path.exists(path_to_chat_folder) :
        os.mkdir( path_to_chat_folder )
    
    path_to_tmp_folder = os.path.join(path_to_chat_folder, '/tmp/')
    if not os.path.exists(path_to_chat_folder) :
        os.mkdir(path_to_tmp_folder)

    create_image_table( path_to_chat_folder )
    create_index_table( path_to_chat_folder )

    logger.info("Creating chat data is complited")
