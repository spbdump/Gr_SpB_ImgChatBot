import cv2
import numpy as np

def compare_images_sift(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    image1 = cv2.imread(img1)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(gray1, None)

    image2 = cv2.imread(img2)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        return True
    else:
        return False
