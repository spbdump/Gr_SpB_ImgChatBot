import db_utils
import img_proccessing
import build_index

import time
import logging

import cv2
import numpy as np
from scipy.spatial import distance

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)

def compare_evc_dist_TEST():
    start = time.time()
    path = './photos/'
    # photo_1@04-03-2022_01-26-02.jpg  # photo_12@04-03-2022_01-54-50_croped.jpg
    # photo_1@04-03-2022_01-26-02.jpg  # photo_12@04-03-2022_01-54-50.jpg

    pic1 = 'photo_1@04-03-2022_01-26-02.jpg'
    pic1_c = 'photo_1@04-03-2022_01-26-02_copy.jpg'
    pic2 = 'photo_12@04-03-2022_01-54-50_croped.jpg'
    pic2_c = 'photo_12@04-03-2022_01-54-50.jpg'
    img1 = cv2.imread(path + pic1)
    img2 = cv2.imread(path + pic1_c)

    img_high_res1 = cv2.resize(img1, (img1.shape[1]*2, img1.shape[0]*2), interpolation = cv2.INTER_CUBIC)
    img_high_res2 = cv2.resize(img2, (img2.shape[1]*2, img2.shape[0]*2), interpolation = cv2.INTER_CUBIC)


    sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)

    if img1 is None or img2 is None:
        print("can't read some image")
        return

    gray1 = cv2.cvtColor(img_high_res1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_high_res2, cv2.COLOR_BGR2GRAY)
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    # n_keypoints=1000
    # kp1 = cv2.KeyPoint(gray1, nkeypoints=n_keypoints)
    # kp2 = cv2.KeyPoint(gray2, nkeypoints=n_keypoints)

    # # Compute the SIFT descriptors
    # _, desc1 = sift.compute(gray1, kp1)
    # _, desc2 = sift.compute(gray2, kp2)


    print(desc1.shape, desc2.shape)

    # return
    
    gap = abs(desc1.shape[0] - desc2.shape[0])

    print("gap: ", gap)

    if gap != 0:
        padding = np.zeros((gap, 128), dtype=np.float32)
        if desc1.shape[0] < desc2.shape[0]:
            desc1 = np.concatenate((desc1, padding))
        else:
            desc2 = np.concatenate((desc2, padding))

    print(desc1.shape, desc2.shape)


    # # reshape to 1d
    # resh_query_descriptor = desc1.reshape(-1)
    # resh_db_descriptor = desc2.reshape(-1)

    print(np.sqrt(np.sum((desc1 - desc2)**2)))
    similarity_score =  distance.cdist(desc1, desc2, 'euclidean')# distance.euclidean(desc1, desc2)

    print("sym score: ", similarity_score)

    print("Is same imgs: ", img_proccessing.compare_sift_descriprtors(desc1, desc2, match_percent=0.6) )

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append([m])

    v_match = len(good_matches)/len(matches)
    print("Match value: ", v_match)

    img_out = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
    img_out_res = cv2.resize(img_out, (int(img_out.shape[1]/2), int(img_out.shape[0]/2)), interpolation = cv2.INTER_CUBIC)
    cv2.imshow("Matched Features", img_out_res)
    cv2.waitKey(0)

    end = time.time()
    logger.info("time takes to find match : %d", end - start)


    orb = cv2.ORB_create(nfeatures=6000, scaleFactor=3, edgeThreshold=5)

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Create brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches by distance
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw the matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=2)
    img_m_res = cv2.resize(img_matches, (int(img_matches.shape[1]/2), int(img_matches.shape[0]/2)), interpolation = cv2.INTER_CUBIC)
    # Show the results
    cv2.imshow("Matches", img_m_res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def retrive_top_k_descriptors_TEST():
    image_data = img_proccessing.get_image_data("./photos/photo_1000@01-08-2022_00-30-30.jpg")
    query_desc = image_data.descriptor
    res_desc_list = db_utils.retrive_top_k_descriptors(query_desc)
    print("size: ", len(res_desc_list), "list: ", res_desc_list)

def poces_similar_sift_descriprors_brootforce_TEST():
    start = time.time()

    image_data = img_proccessing.get_image_data("./photos/photo_1000@01-08-2022_00-30-30.jpg")
    query_desc = image_data.descriptor
    res_arr = img_proccessing.poces_similar_sift_descriprors_brootforce(query_desc)

    for res in res_arr:
        print(res["_id"], res["img_name"])

    end = time.time()
    logger.info("time takes to find match : %d", end - start)

def poces_similar_sift_descriprors_top_k_TEST():
    start = time.time()

    image_data = img_proccessing.get_image_data("./photos/photo_1000@01-08-2022_00-30-30.jpg")
    query_desc = image_data.descriptor
    res_arr = img_proccessing.poces_similar_sift_descriprors(query_desc)

    for res in res_arr:
        print(res["_id"], res["img_name"])

    end = time.time()
    logger.info("time takes to find match : %d", end - start)

def poces_similar_sift_descriprors_ann_index_TEST():
    start = time.time()

    image_data = img_proccessing.get_image_data("./photos/photo_1000@01-08-2022_00-30-30.jpg")
    query_desc = image_data.descriptor
    res_arr = img_proccessing.poces_similar_sift_descriprors_ann_index(query_desc)

    print("size res:", len(res_arr))
    for res in res_arr:
        print(res[1])

    end = time.time()
    logger.info("time takes to find match : %d", end - start)

def bild_index_TEST():
    start = time.time()

    print("TEST: Build Index DB")
    # db_utils.drop_collection()
    db_utils.drop_collection()
    build_index.build_index("./photos")

    end = time.time()
    logger.info("time takes to find match : %d", end - start)

def db_sphere_index_search_TEST():
    # db_utils.create_sphere_index()
    image_data = img_proccessing.get_image_data("./photos/photo_30@04-03-2022_02-15-28.jpg")
    query_desc = image_data.descriptor

    cursor = db_utils.find_desc(query_desc.tolist())
    if cursor is not None:
        for document in cursor:
            # Access the individual fields of each document
            image_name = document["img_name"]
            # print("img: ", image_name)


def main():

    # bild_index_TEST()
    # db_sphere_index_search_TEST()
    poces_similar_sift_descriprors_ann_index_TEST()

    # retrive_top_k_descriptors_TEST()
    # poces_similar_sift_descriprors_brootforce_TEST()
    # poces_similar_sift_descriprors_top_k_TEST()
    # compare_evc_dist_TEST()


if __name__ == "__main__":
    main()
