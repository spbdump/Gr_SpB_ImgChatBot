import numpy as np

from pymongo import MongoClient
import os
import time
import sys

import math_utils
import annoy
import bson

# Enable logging
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

DB_NAME = os.environ.get("DB_NAME")
DB_ADDRES = os.environ.get("DB_ADDRES")

def drop_collection():
    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return

    db = client[DB_NAME]
    collection = db["image_data_collection"]
    collection.drop()
    logger.info("Collection 'image_data_collection' was deleted")

def find_desc(target_descriptor):
    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return

    db = client[DB_NAME]
    collection = db["image_data_collection"]

    # Create an index on the descriptor field
    collection.create_index([("descriptor", "2dsphere")])
    results = collection.find({"descriptor": {"$nearSphere": target_descriptor}})

    return results


def create_sphere_index():
    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return []

    db = client[DB_NAME]
    collection = db["image_data_collection"]

    # Create an index on the descriptor field
    collection.create_index([("descriptor", "2dsphere")])

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


def retrive_all_descriptors():
    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return []

    db = client[DB_NAME]
    collection = db["image_data_collection"]

    # cnt = collection.delete_many({"img_name":{"$regex": ".*_thumb.jpg"}})
    # logger.info("%d records ws delted", cnt)
    collection_size = collection.count_documents({})

    logger.info("collection size: %d", collection_size)
    logger.info("collection size: %d", collection_size)
    descriptors = list(collection.find({},{"descriptor": 1, "_id": 1, "img_name": 1}))
    return descriptors

def retrive_top_k_descriptors(query_descriptor):
    max_value_of_k = 30

    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return []

    db = client[DB_NAME]
    collection = db["image_data_collection"]

    max_documents = 150 # should be MUCH MORE around 500
    offset = max_documents # take smth around 16%
    curr_offset = 0
    collection_size = collection.count_documents({})

    logger.info("collection size: %d", collection_size)
    
    top_k_descriptors = []

    logger.info("collection size: %d", collection_size)

    while curr_offset < collection_size:

        start = time.time()
        descriptors = list(collection.find({},{"descriptor": 1, "_id": 1}).limit(max_documents).skip(offset))
        end = time.time()

        curr_offset += offset

        logger.info("currtent offset: %d", curr_offset)
        logger.info("time takes on request is : %d", end - start)

        print("Size of descs: ", len(descriptors))
        
        similarity_scores = []
        for d in descriptors:
            descriptor = np.array(d["descriptor"], dtype=np.float32)
            
            similarity_score = math_utils.euclidian_distance(query_descriptor, descriptor)
            similarity_scores.append((descriptor, similarity_score, d["_id"]))

        similarity_scores.sort(key=lambda x: x[1])
        
        top_k_descriptors = top_k_descriptors + similarity_scores[:max_value_of_k]


    return top_k_descriptors

def retrive_ann_index_descriptors(query_descriptor):
    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return []

    db = client[DB_NAME]
    collection = db["image_data_collection"]

    max_documents = 150 # should be MUCH MORE around 500
    offset = max_documents # take smth around 16%
    curr_offset = 0
    collection_size = collection.count_documents({})

    logger.info("collection size: %d", collection_size)
    
    res_descriptors = []

    logger.info("collection size: %d", collection_size)

    while curr_offset < collection_size:

        start = time.time()
        descriptors = list(collection.find({},{"descriptor": 1, "_id": 1}).limit(max_documents).skip(offset))
        end = time.time()
        curr_offset += offset

        logger.info("currtent offset: %d", curr_offset)
        logger.info("time takes on request is : %d", end - start)

        # Build the index
        descriptor_size = 128
        nfeatures = 1000
        f = descriptor_size*nfeatures
        index = annoy.AnnoyIndex(f, metric='angular')
        for i, desc in enumerate(descriptors):
            d = np.array(desc["descriptor"], dtype=np.float32)
            print(d.shape)
            gap = 1000 - d.shape[0]

            if gap > 0:
                padding = np.zeros((gap, descriptor_size), dtype=np.float32)
                d = np.concatenate((d, padding))
            elif gap < 0:
                d = d[:gap, :]

            print(d.shape)
            index.add_item(i, d.reshape(-1).tolist())

        index.build(50)


        gap = 1000 - query_descriptor.shape[0]

        if gap > 0:
            padding = np.zeros((gap, descriptor_size), dtype=np.float32)
            query_descriptor = np.concatenate((query_descriptor, padding))
        elif gap < 0:
            query_descriptor = query_descriptor[:gap, :]

        # Query the index to find the approximate nearest neighbors
        descriptors_idx = index.get_nns_by_vector(query_descriptor.reshape(-1).tolist(), 20, search_k=-1, include_distances=False)
        for idx in descriptors_idx:
            res_descriptors.append(np.array(descriptors[idx]["descriptor"], dtype=np.float32))

    return res_descriptors





def get_addtional_data_about_image(desc): #FIXIT
    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return

    db = client[DB_NAME]
    collection = db["image_data_collection"]
    data = list(collection.find({"value": desc}))

    return data
