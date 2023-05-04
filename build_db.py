import img_proccessing
import db_utils
import HNSW_index

import os

# Enable logging
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def get_image_files(directory: str):
    image_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_files.append(filename)
    return image_files

def fullfill(path_to_imgs_dir: str):

    db_utils.drop_collection()

    list_imgs = get_image_files(path_to_imgs_dir)
    max_iter_size = HNSW_index.BATCH_SIZE # should be same with size of index

    # if max_iter_size too big may be need divide external loop !!! because take a lot of RAM

    curr_idx = 0
    last_idx = 0
    logger.info("Proccessing %d images", len(list_imgs))

    curr_batch_idx = 0
    curr_batch_idx_in = 0

    while curr_idx < len(list_imgs):
        if curr_idx + max_iter_size < len(list_imgs):
           last_idx = curr_idx + max_iter_size 
        else:
            last_idx = len(list_imgs) - 1

        imgs_data = []
        for img_path in list_imgs[curr_idx:last_idx]:
            img_data = img_proccessing.get_image_data(path_to_imgs_dir + '/' + img_path)
            img_data.batch_id = curr_batch_idx
            img_data.batch_id_in = curr_batch_idx_in
            imgs_data.append(img_data)
            curr_batch_idx_in += 1

            if len(imgs_data) > 300:
                logger.info("%d images was proccessed", len(imgs_data))
                db_utils.save_img_data(imgs_data)
                logger.info("Images data was saved to db")
                imgs_data = []

        logger.info("%d images was proccessed", len(imgs_data))
        db_utils.save_img_data(imgs_data)
        logger.info("Images data was saved to db")


        curr_idx += max_iter_size

        if curr_batch_idx_in == HNSW_index.BATCH_SIZE:
            logger.info("Batch fullfilled. Current id: %d", curr_batch_idx)
            curr_batch_idx += 1
            curr_batch_idx_in = 0

