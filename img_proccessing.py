import cv2
import os
import numpy as np

import image_d
import HNSW_index

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

NFEATURES = 1000

def compare_images_sift(img1, img2):
    sift = cv2.SIFT_create(nfeatures=NFEATURES)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(gray1, None)

    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    # DEBUG
    # img_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
    # cv2.imshow('image', img_matches)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print("lenght od good matches: ", len(good_matches), "len matches: ", len(matches))

    if len(good_matches)/len(matches) > 0.9:
        return True
    else:
        return False

def compare_sift_descriprtors(desc1: np.ndarray, desc2: np.ndarray, match_percent=0.85) -> bool:
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < match_percent * n.distance:
            good_matches.append([m])

    v_match = len(good_matches)/len(matches)

    # print("Match persets: ", v_match,", Distanse: ", math_utils.euclidian_distance(desc1, desc2))

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

    return res

def poces_similar_sift_descriprors_brootforce(query_descriptor):
    desc_list = db_utils.retrive_all_descriptors()

    logger.info("Got %d descriptors", len(desc_list))
    res = []
    for desc in desc_list:
        if compare_sift_descriprtors(query_descriptor, np.array(desc["descriptor"], dtype=np.float32)) == True:
            logger.info("Got some match")
            res.append(desc)

    return res

def poces_similar_sift_descriprors_ann_index(query_descriptor):
    desc_list = db_utils.retrive_ann_index_descriptors_nms(query_descriptor)

    logger.info("Got %d descriptors", len(desc_list))
    # print(desc_list)
    res = []
    for desc in desc_list:
        if compare_sift_descriprtors(query_descriptor, desc[0]) == True:
            logger.info("Got some match")
            res.append(desc)

    return res

def get_image_data(path_to_img) -> image_d.ImageData:

    img = cv2.imread(path_to_img, cv2.COLOR_BGR2GRAY)
    if img is None:
        logger.info("Can't open image: %s", path_to_img)
        return image_d.ImageData([[]], image_d.DescriptorType.SIF, img_name="")

    sift = cv2.SIFT_create(nfeatures=NFEATURES, contrastThreshold=0.04, edgeThreshold=10)
    kp1, desc = sift.detectAndCompute(img, None)

    return image_d.ImageData(desc, image_d.DescriptorType.SIF, img_name=os.path.basename(path_to_img))
