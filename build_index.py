import img_proccessing
import db_utils
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

def build_index(path_to_imgs_dir: str):

    list_imgs = get_image_files(path_to_imgs_dir)
    max_iter_size = 150

    curr_idx = 0
    last_idx = 0
    logger.info("Proccessing %d images", len(list_imgs))
    while curr_idx < 300:
        if curr_idx + max_iter_size < len(list_imgs):
           last_idx = curr_idx + max_iter_size 
        else:
            last_idx = len(list_imgs) - 1

        imgs_data = [img_proccessing.get_image_data(path_to_imgs_dir + '/' + img_path) for img_path in list_imgs[curr_idx:last_idx]]
        logger.info("%d images was proccessed", len(imgs_data))
        db_utils.save_img_data(imgs_data)
        logger.info("Images data was saved to db")

        curr_idx += max_iter_size


