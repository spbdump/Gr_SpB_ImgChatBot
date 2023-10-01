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

def get_image_data(path_to_img, nfeatures:int =NFEATURES) -> Tuple[np.ndarray, img_d.DescriptorType]:

    img = cv2.imread(path_to_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.info("Can't open image: %s", path_to_img)
        return np.empty([]), img_d.DescriptorType.SIFT

    width = int(img.shape[1] * 1.5)
    height = int(img.shape[0] * 1.5)
    scaled_img = cv2.resize(img, (width, height))

    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.04, edgeThreshold=10)
    kps, desc = sift.detectAndCompute(scaled_img, None)

    # if desc.shape[0] < nfeatures:
    #     desc = try_add_preprocess(img, sift, nfeatures)

    return desc, img_d.DescriptorType.SIFT

def get_image_data_v2(path_to_img, nfeatures:int =NFEATURES, filters=0) -> Tuple[np.ndarray, img_d.DescriptorType]:

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
    if desc.shape[0] < nfeatures and filters & FilterFlags.GAUSSIAN.value:
        img = noise_reduction_gaussian(img)
        _, desc = sift.detectAndCompute(img, None)

    if desc.shape[0] < nfeatures and filters & FilterFlags.SHARPERING.value:
        img = reduce_sharpening(img)
        _, desc = sift.detectAndCompute(img, None)

    if desc.shape[0] < nfeatures and filters & FilterFlags.HIST_EQUALIZATION.value:
        img = histogram_equalization(img)
        _, desc = sift.detectAndCompute(img, None)

    if desc.shape[0] < nfeatures and filters & FilterFlags.DENOISE.value:
        img = noise_reduction(img)
        _, desc = sift.detectAndCompute(img, None)

    return desc, img_d.DescriptorType.SIFT


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
