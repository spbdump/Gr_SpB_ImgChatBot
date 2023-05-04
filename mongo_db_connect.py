import cv2
import pymongo
import numpy as np
import annoy

from  img_proccessing import NFEATURES

def store_descriptors(img, sift, index, i):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(gray, None)

    index.add_item(i, des)

def retrieve_descriptors(img, sift, collection):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(gray, None)

    document = collection.find_one({'descriptor': des.tolist()})
    if document is not None:
        return np.array(document['descriptor'])
    else:
        return None

def compare_images_sift_mongodb(img1):
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["image_db"]
    collection = db["descriptors"]

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=NFEATURES)

    index = annoy.AnnoyIndex(128, 'angular')

    documents = list(collection.find())
    for i, document in enumerate(documents):
        index.add_item(i, document['descriptor'])

    index.build(10)

    des1 = retrieve_descriptors(img1, sift, collection)
    if des1 is None:
        store_descriptors(img1, sift, index, len(documents))
        des1 = retrieve_descriptors(img1, sift, collection)
        documents = list(collection.find())

    id1 = index.get_nns_by_vector(des1, 1)[0]
    document = documents[id1]
    img2 = document['img_path']

    print("The incoming image matches with the image at", img2)
