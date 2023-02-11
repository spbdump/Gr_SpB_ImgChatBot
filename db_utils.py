import numpy as np
from scipy.spatial import distance
from pymongo import MongoClient
import os
import logging
import time

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)

DB_NAME = os.environ.get("DB_NAME")
DB_ADDRES = os.environ.get("DB_ADDRES")

def save_img_data(img_data_arr):
    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return

    db = client[DB_NAME]
    collection = db["image_data_collection"]

    data_dicts = [i.__dict__() for i in img_data_arr]

    # print(len(bson.encode(data_dicts)))
    try:
        collection.insert_many(data_dicts)
    except Exception as e:
        print("An error occurred:", e)


def retrive_top_k_descriptors(query_descriptor):
    max_value_of_k = 50

    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return []

    db = client[DB_NAME]
    collection = db["image_data_collection"]

    max_documents = 150 # should be MUCH MORE around 500
    offset = 150 # take smth around 16%
    curr_offset = 0
    descriptor_size = 128
    collection_size = collection.count_documents({})

    logger.info("collection size: %d", collection_size)
    
    similarity_scores = []
    top_k_descriptors = []

    while curr_offset < collection_size:

        start = time.time()
        descriptors = list(collection.find({},{"descriptor": 1, "_id": 1}).limit(max_documents).skip(offset))
        end = time.time()

        curr_offset += offset

        logger.info("currtent offset: %d", curr_offset)
        logger.info("time takes on request is : %d", end - start)

        print("Size of descs: ", len(descriptors))
        
        for d in descriptors:
            descriptor = np.array(d["descriptor"])
            gap = abs(descriptor.shape[0] - query_descriptor.shape[0])

            if gap != 0:
                padding = np.zeros((gap, descriptor_size), dtype=np.float32)
                if descriptor.shape[0] < query_descriptor.shape[0]:
                    descriptor = np.concatenate((descriptor, padding))
                else:
                    query_descriptor = np.concatenate((query_descriptor, padding))

            # print(descriptor.shape, query_descriptor.shape)
            # reshape to 1d
            resh_query_descriptor = query_descriptor.reshape(-1)
            resh_db_descriptor = np.array(descriptor).reshape(-1)

            similarity_score = distance.euclidean(resh_query_descriptor, resh_db_descriptor)

            similarity_scores.append((np.array(d["descriptor"], dtype=np.float32), similarity_score, d["_id"]))
            similarity_scores.sort(key=lambda x: x[1])
            
            top_k = max_value_of_k
            top_k_descriptors = similarity_scores[:top_k]

    return top_k_descriptors


def get_addtional_data_about_image(desc): #FIXIT
    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return

    db = client[DB_NAME]
    collection = db["image_data_collection"]
    data = list(collection.find({"value": desc}))

    return data
