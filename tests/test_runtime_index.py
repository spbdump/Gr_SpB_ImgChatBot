import core.runtime_index as runtime_index
import bot_general
import model.context as context
import utils
import core.sqlite_db_utils as sqlite_db_utils

PATH_TO_IMGS = './grbrt_spb/photos/'

import random

def main():

    nfeatures = 800
    max_size = 20
    chat_id = -1
    chat_path = './tests/'
    runtime_index.add_runtime_index(chat_id, 0, max_size, nfeatures)
    sqlite_db_utils.create_index_table(chat_path)
    sqlite_db_utils.create_image_table(chat_path)

    prefix_img_path = PATH_TO_IMGS
    list_imgs = utils.get_image_files( prefix_img_path )
    rand_imgs_list = utils.get_random_images(list_imgs, 60)

    saved_img_list = []

    for img_name in rand_imgs_list:
        path_to_img = prefix_img_path + img_name
        
        ctx = context.Context(nfeatures, 128, max_size, chat_path, chat_id)
        img_data = bot_general.get_image_data(path_to_img, nfeatures)
        img_desc = img_data.descriptor

        if img_desc.shape[0] < nfeatures:
            continue
        elif img_desc.shape[0] > nfeatures:
            img_desc = img_desc[:nfeatures]

        saved_img_list.append(path_to_img)
        bot_general.update_index( ctx, './', img_desc, img_name, -1 )

    path_to_img = saved_img_list[6]
    res, img_desc = bot_general.find_image_in_indexes(path_to_img, chat_path, chat_id, nfeatures)
    print(path_to_img, res)

    path_to_img = saved_img_list[24]
    res, img_desc = bot_general.find_image_in_indexes(path_to_img, chat_path, chat_id, nfeatures)
    print(path_to_img, res)

    path_to_img = saved_img_list[-3]
    res, img_desc = bot_general.find_image_in_indexes(path_to_img, chat_path, chat_id, nfeatures)
    print(path_to_img, res)

if __name__ == "__main__":
    main()