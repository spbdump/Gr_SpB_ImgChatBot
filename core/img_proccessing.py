import cv2
import os
import numpy as np

import model.image_d as image_d
import core.HNSW_index as HNSW_index

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

def get_image_data(path_to_img, nfeatures:int =NFEATURES) -> image_d.ImageData:

    img = cv2.imread(path_to_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.info("Can't open image: %s", path_to_img)
        return image_d.ImageData([[]], image_d.DescriptorType.SIF, img_name="")

    width = int(img.shape[1] * 1.5)
    height = int(img.shape[0] * 1.5)
    scaled_img = cv2.resize(img, (width, height))

    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.04, edgeThreshold=10)
    kps, desc = sift.detectAndCompute(scaled_img, None)

    # if desc.shape[0] < nfeatures:
    #     desc = try_add_preprocess(img, sift, nfeatures)

    return image_d.ImageData(desc, image_d.DescriptorType.SIF, img_name=os.path.basename(path_to_img))


def try_add_preprocess(img, detector, nfeatures):
    logger.debug("Preprocess: histogram equalization")
    enhanced_image = cv2.equalizeHist(img)
    kps, desc = detector.detectAndCompute(enhanced_image, None)

    blurred_image = enhanced_image
    if desc.shape[0] < nfeatures:
        logger.debug("Preprocess: gaussian blur")
        kernel_size = 5
        blurred_image = cv2.GaussianBlur(enhanced_image, (kernel_size, kernel_size), 0)
        kps, desc = detector.detectAndCompute(blurred_image, None)

    return desc