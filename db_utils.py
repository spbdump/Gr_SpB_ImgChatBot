import os
import numpy as np
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError



NFEATURES=5000

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


def get_addtional_data_about_image(desc): #FIXIT
    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return

    db = client[DB_NAME]
    collection = db["image_data_collection"]
    data = list(collection.find({"value": desc}))

    return data
