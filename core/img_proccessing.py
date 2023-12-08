import cv2
import os
import numpy as np

from typing import Tuple
from enum import Enum

import model.img_data as img_d
import core.HNSW_index as HNSW_index

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

NFEATURES = 1000

# Define an enumeration for filter flags
class FilterFlags(Enum):
    GAUSSIAN = 1
    HIST_EQUALIZATION = 2
    SHARPERING = 4
    DENOISE = 8
    RES_ENHANCEMENT = 16

# Brightness and Contrast Adjustment
def brightness_contrast_adjust(image):
    alpha = 1.5  # Adjust as needed
    beta = 10    # Adjust as needed
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return enhanced_image

# Histogram Equalization
def histogram_equalization(gray_image):
    equalized_image = cv2.equalizeHist(gray_image)
    return equalized_image

# Noise Reduction
def noise_reduction_gaussian(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image

# Sharpening
def reduce_sharpening(image):
    kernel = np.array([[-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image

# Noise Reduction
def noise_reduction(image:np.ndarray):
    denoised_image = cv2.fastNlMeansDenoising(image)
    return denoised_image

# Resolution Enhancement
def resolution_enhancement(image, new_width, new_height):
    upscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return upscaled_image

filter_functions = {
    FilterFlags.GAUSSIAN: noise_reduction_gaussian,
    FilterFlags.SHARPERING: reduce_sharpening,
    FilterFlags.HIST_EQUALIZATION: histogram_equalization,
    FilterFlags.DENOISE: noise_reduction,
}

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
    
def compare_descriprtors(desc1: np.ndarray, desc2: np.ndarray, matcher, match_percent=0.85) -> bool:
    matches = matcher.knnMatch(desc1, desc2, k=2)
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

def get_image_data_sift(path_to_img, nfeatures:int =NFEATURES, filters=0) -> Tuple[np.ndarray, img_d.DescriptorType]:

    img = cv2.imread(path_to_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.info("Can't open image: %s", path_to_img)
        return np.empty([]), img_d.DescriptorType.SIFT

    width = int(img.shape[1] * 1.5)
    height = int(img.shape[0] * 1.5)
    img = resolution_enhancement(img, width, height)

    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.04, edgeThreshold=10)
    _, desc = sift.detectAndCompute(img, None)

    # Check filter flags and apply filters accordingly
    for flag, func in filter_functions.items():
        if desc.shape[0] < nfeatures and filters & flag.value:
            img = func(img)
            _, desc = sift.detectAndCompute(img, None)

    return desc, img_d.DescriptorType.SIFT

def get_FLANN_matcher():
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)