import cv2
import os
import numpy as np

import image_d
import HNSW_index

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

NFEATURES = 2500

def compare_sift_descriprtors(desc1: np.ndarray, desc2: np.ndarray, match_percent=0.7) -> bool:
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append([m])

    v_match = len(good_matches)/len(matches)

    if v_match > match_percent:
        return True
    else:
        return False

def poces_similar_sift_descriprors(query_descriptor):
    logging.info("Start search similar inexies")
    desc_list = HNSW_index.get_neighbors_descriptors(query_descriptor)
    
    logging.info("Match all neighbors descriptors")
    res = []
    for desc in desc_list:
        db_desc = np.array(desc["descriptor"], dtype=np.float32)
        if compare_sift_descriprtors(query_descriptor, db_desc) == True:
            res.append(desc)

    logging.info("Done. Was found %d descriptors", len(desc_list))
    
    return res


def get_image_data(path_to_img) -> image_d.ImageData:

    img = cv2.imread(path_to_img)
    if img is None:
        logger.info("Can't open image: %s", path_to_img)
        return image_d.ImageData([[]], image_d.DescriptorType.SIF, img_name="")

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=NFEATURES)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp1, desc = sift.detectAndCompute(gray, None)

    return image_d.ImageData(desc, image_d.DescriptorType.SIF, img_name=os.path.basename(path_to_img))
