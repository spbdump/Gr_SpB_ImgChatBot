import os
import random

import bot_impl as bi
import core.img_proccessing as imp
from model.context import Context
from core.image_utils import read_image, get_image_filenames

import shutil

def test_run_app():

    chat_id = 123
    runtime_index_size = 4
    chat_path = "./data/test_chat"
    path = os.path.dirname(__file__)
    path_to_chat_folder = os.path.join(path, chat_path)

    ctx = Context(1000, 128, 10, path_to_chat_folder, chat_id)

    if not os.path.exists(path_to_chat_folder):
        os.mkdir( path_to_chat_folder )
    else:
        shutil.rmtree(path_to_chat_folder)

    bi.update_DBPATH(os.path.join(path, './data'))

    # create general db
    bi.create_general_db()
    # create chat db and add test_chat path
    bi.on_add_bot( ctx.chat_id, path, ctx.chat_path, runtime_index_size,
                  ctx.max_size, ctx.nfeatures, ctx.desc_size)


    # actual app running

    bi.init(runtime_index_size)

    path_to_images = os.path.join(path, './images')
    list_names = get_image_filenames( path_to_images )

    # add images to index
    count_for_rn_index = runtime_index_size - 1 # SHOULD BE LESS THAT runtime_index_size
    for image_name in list_names[:count_for_rn_index]:
        # random_index = random.randint(0, len(list_names) - 1)
        # random_image_name = list_names[random_index]
        path_to_image =  os.path.join( path_to_images, image_name )

        image = read_image(path_to_image)
        desc, _ = imp.get_image_data_sift( image )

        is_updated = bi.update_index( ctx, desc, image_name, -1 )
        assert is_updated == True, f"New image wasn't saved to index {image_name}"

    random_index = random.randint(0, count_for_rn_index - 1)
    random_image_name = list_names[random_index]
    path_to_random_image =  os.path.join( path_to_images, random_image_name )

    # find image in runtime index
    res, _ = bi.find_image_in_indexes(path_to_random_image, ctx.chat_path, ctx.chat_id, ctx.nfeatures)
    print( f'image id {res}')
    assert len(res) != 0, "Can't find input image in indexes"
    assert res[0][0] != random_index, "Wrong image index"


    # add images to index
    for image_name in list_names[count_for_rn_index:]:
        path_to_image =  os.path.join( path_to_images, image_name )

        image = read_image(path_to_image)
        desc, _ = imp.get_image_data_sift( image )

        is_updated = bi.update_index( ctx, desc, image_name, -1 )
        assert is_updated == True, f"New image wasn't saved to index {image_name}"

    # try find same image wich is should be in saved index already
    res, _ = bi.find_image_in_indexes(path_to_random_image, ctx.chat_path, ctx.chat_id, ctx.nfeatures)
    print( f'image id {res}')
    assert len(res) != 0, "Can't find input image in indexes"
    assert res[0][0] != random_index, "Wrong image index"



    # path_to_img_not_in_index = ""
    # res, img_desc = bi.find_image_in_indexes(path_to_img_not_in_index, ctx.chat_path, ctx.chat_id, ctx.nfeatures)
    # assert len(res) != 0, "Image was found in indexes"

    # # search inmemory indexes
    # res, img_desc = bi.find_image_in_indexes(path_to_img_in_index, ctx.chat_path, ctx.chat_id, ctx.nfeatures)
    # assert len(res) == 0
