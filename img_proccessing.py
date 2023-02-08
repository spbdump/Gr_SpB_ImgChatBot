import cv2
import numpy as np

def compare_images_sift(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

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
