import numpy as np

from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

import os

# Enable logging
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

DB_NAME = os.environ.get("DB_NAME")
DB_ADDRES = os.environ.get("DB_ADDRES")


def check_connection() -> bool:
    client = MongoClient(DB_ADDRES, serverSelectionTimeoutMS=10, connectTimeoutMS=20000)

    try:
        info = client.server_info() # Forces a call.
        print(info)

        return True
        
    except ServerSelectionTimeoutError:
        print("server is down.")

    return False


def get_desc_collection_size() -> int:
    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return 0

    db = client[DB_NAME]
    collection = db["image_data_collection"]

    return collection.count_documents({})

def get_desc_batch(batch_offset:int, batch_size:int, raw_data:bool = False):

    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return []

    db = client[DB_NAME]
    collection = db["image_data_collection"]

    descriptors = list(collection.find({},{"descriptor": 1, "_id": 1, "img_name": 1}).limit(batch_size).skip(batch_offset))
    logger.info("Got %d descriptors from 'image_data_collection'", len(descriptors))

    if not raw_data:
        np_descriptors = []
        for desc in descriptors:
            first_nfeatures = 1000
            np_desc = np.array(desc["descriptor"], dtype=np.float32)[:first_nfeatures]

            if np_desc.shape[0] < first_nfeatures:
                np_desc = np.pad(np_desc, ((0,first_nfeatures - np_desc.shape[0]), (0, 0)), mode='constant', constant_values=0)

            np_descriptors.append(np_desc.flatten())

        logger.info("Got converted descriptors data from 'image_data_collection'")
        # np_descriptors = np.concatenate(np_descriptors, axis=1)
        print("res desc shape is ", np.array(np_descriptors).shape)
        return np_descriptors

    logger.info("Got raw image data from 'image_data_collection'")
    return descriptors

def get_desc_by_batch_indexes(indexies):
    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return []

    db = client[DB_NAME]
    collection = db["image_data_collection"]

    descriptors = []
    for idx in indexies:
        batch_idx = idx
        batch_indexies = indexies[idx]

        logger.info("batch_id: %d", batch_idx) 
        
        descriptors += list(collection.find(
            {"batch_id": batch_idx, "batch_id_in": { "$in" : batch_indexies}},
            {"descriptor": 1, "_id": 1, "img_name": 1}
            ))

    return descriptors


def drop_collection():
    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return

    db = client[DB_NAME]
    collection = db["image_data_collection"]
    collection.drop()
    logger.info("Collection 'image_data_collection' was deleted")

def save_img_data(img_data_arr):
    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return

    db = client[DB_NAME]
    collection = db["image_data_collection"]
    

    data_dicts = [i.__dict__() for i in img_data_arr]

    # print(len(bson.encode(data_dicts)))
    # size_of_array_of_dicts = sys.getsizeof(data_dicts)
    # size_of_array_of_dicts = sys.getsizeof(bson.BSON.encode(data_dicts))
    # print("curr bson data size is :", size_of_array_of_dicts)
    # max_av_size = 16793600
    # if size_of_array_of_dicts >= max_av_size:
    #     logger.info("Dictionary size is overflow: %d bytes", size_of_array_of_dicts)
    #     return

    logger.info("%d records should be saved", len(data_dicts))
    try:
        res = collection.insert_many(data_dicts)
        logger.info("Was saved %d descriptors", len(res.inserted_ids))
    except Exception as e:
        print("An error occurred:", e)


def get_addtional_data_about_image(id): #FIXIT
    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return

    db = client[DB_NAME]
    collection = db["image_data_collection"]
    data = list(collection.find({"_id": id}))

    return data
