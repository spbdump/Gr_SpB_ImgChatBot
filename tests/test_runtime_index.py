import pytest
import core.runtime_index as ri
import core.img_proccessing as imp
import model.context as context
import utils
import cv2

PATH_TO_IMGS = 'tests/images/'
NFEATURES = 800

# setup test enviroment
max_size = 20
chat_id = -1
img_list = utils.get_image_files( PATH_TO_IMGS )
img_data = []

@pytest.fixture
def runtime_index():
    ri.add_runtime_index(chat_id, 0, max_size, NFEATURES)
    rt_index = ri.get_runtime_index(chat_id)

    for img_name in img_list:
        desc, _ = imp.get_image_data_sift(image_path, NFEATURES)
        id = rt_index.add_data_point( desc, img_name, -1 )
        if :
            img_data.append( (image_path, id, True) )
        else:
            img_data.append( (image_path, -1, False) )

    return rt_index

@pytest.mark.parametrize("image_path, image_id, expected_res", img_data )
def test_image_search_BF(image_path, image_id, expected_res, runtime_index):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) # or pass empty dictionary
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    desc, _ = imp.get_image_data_sift(image_path, NFEATURES)
    id_list = runtime_index.find_image(desc, matcher)

    assert image_id in id_list == expected_res

@pytest.mark.parametrize("image_path, image_id, expected_res", img_data )
def test_image_search_FLANN(image_path, image_id, expected_res, runtime_index):
    matcher = cv2.BFMatcher()
    desc, _ = imp.get_image_data_sift(image_path, NFEATURES)
    id_list = runtime_index.find_image(desc, matcher)

    assert image_id in id_list == expected_res
