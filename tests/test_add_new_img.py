import pytest
import bot_general
import context

import utils

PATH_TO_IMGS = './grbrt_spb/photos/'
N_RAND_IMGS = 25


@pytest.fixture
def prefix_img_path():
    return PATH_TO_IMGS


list_imgs = utils.get_image_files( PATH_TO_IMGS )
rand_imgs_list = utils.get_random_images(list_imgs, 24)


@pytest.mark.parametrize("img_name", rand_imgs_list)
def test_add_new_img(img_name, prefix_img_path):
    nfeatures = 800
    chat_path = './tests/'
    path_to_img = prefix_img_path + img_name
    ctx = context.Context(nfeatures, 128, 10, chat_path, -1)
    img_data = bot_general.get_image_data(path_to_img, nfeatures)
    img_desc = img_data.descriptor

    if img_desc.shape[0] < nfeatures:
        assert False

    # move this check to get_imag_data
    if img_desc.shape[0] > nfeatures:
        img_desc = img_desc[:nfeatures]

    # should be before saving img data to db
    # because can create new index
    #b_updated = False
    #with pytest.raises(Exception):
    b_updated = bot_general.update_index( img_desc, ctx )

    assert b_updated == True

    message_id = -1
    assert bot_general.save_img_data(chat_path, img_name, message_id) == True

    return
