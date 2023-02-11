import db_utils
import img_proccessing
import time
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)
# print("Check base functionality")
# path = './test_images/'
# img2 = cv2.imread(path+'photo_2023-02-09_02-17-55.jpg')
# img1 = cv2.imread(path+'photo_2023-02-09_02-17-55_copy.jpg')
# img3 = cv2.imread(path+'photo_2023-02-08_19-24-07.jpg ')

# # Check if the image was loaded successfully
# if img2 is None:
#     print("Error: Could not load the image.")
#     return

# print("Is same imgs: ", img_proccessing.compare_images_sift(img1, img2) )
# print("Is same imgs: ", img_proccessing.compare_images_sift(img1, img3) )


def retrive_top_k_descriptors_TEST():
    image_data = img_proccessing.get_image_data("./photos/photo_1@04-03-2022_01-26-02_thumb.jpg")
    query_desc = image_data.descriptor
    res_desc_list = db_utils.retrive_top_k_descriptors(query_desc)
    print("size: ", len(res_desc_list), "list: ", res_desc_list)

def poces_similar_sift_descriprors_TEST():
    start = time.time()
    image_data = img_proccessing.get_image_data("./photos/photo_100@07-03-2022_23-30-29.jpg")
    query_desc = image_data.descriptor
    res = img_proccessing.poces_similar_sift_descriprors(query_desc)

    print(len(res))
    end = time.time()
    logger.info("time takes to find match : %d", end - start)

def main():
    # retrive_top_k_descriptors_TEST()
    poces_similar_sift_descriprors_TEST()

if __name__ == "__main__":
    main()
