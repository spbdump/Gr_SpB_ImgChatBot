import numpy as np
from scipy.spatial import distance
from pymongo import MongoClient
import os

DB_NAME = os.environ.get("DB_NAME")
DB_ADDRES = os.environ.get("DB_ADDRES")

def save_img_data(img_data_arr):
    client = MongoClient(DB_ADDRES)

    if not client.test :
        print("Error connection to DB")

    if DB_NAME == None:
        return

    db = client[DB_NAME]
    collection = db["image_data_collection"]

    data_dicts = [i.__dict__ for i in img_data_arr]
    collection.insert_many(data_dicts)


def retrive_top_k_descriptors(query_descriptor):
    max_value_of_k = 50

    client = MongoClient(DB_ADDRES)

    if DB_NAME == None:
        return

    db = client[DB_NAME]
    collection = db["image_data_collection"]
    descriptors = list(collection.find({"type": "descriptor"}))

    similarity_scores = []
    top_k_descriptors = []
    for d in descriptors:
        descriptor = d["descriptor"]
        similarity_score = distance.euclidean(query_descriptor, descriptor)
        similarity_scores.append((d["_id"], similarity_score))
        similarity_scores.sort(key=lambda x: x[1])
        
        top_k = max_value_of_k
        top_k_descriptors = similarity_scores[:top_k]
        # top_k_images = []
        # for d in top_k_descriptors:
        #     image = collection.find_one({"_id": d[0], "type": "image"})
        #     top_k_images.append(image)

    return top_k_descriptors
