import HNSW_index
import img_proccessing
import file_descriptor_utils

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)


def test_find_image():
    path_to_img = "./src_imgs/indexed_imgs/photo_720@26-05-2022_18-10-53.jpg"
    expected_found = True
    index = HNSW_index.load_index("./indexies/test_index.bin")

    img_data = img_proccessing.get_image_data(path_to_img)
    q_desc = img_data.descriptor

    desc_idx_list = HNSW_index.get_neighbors_desc_indexies(index, q_desc)

    if len(desc_idx_list) == 0:
        assert False

    desc_list = file_descriptor_utils.read_specific_rows_from_file("./descriptors/test_desc.npy", desc_idx_list)

    if len(desc_list) == 0:
        assert False

    res = []
    for desc in desc_list:
        if img_proccessing.compare_sift_descriprtors(q_desc, desc_list) == True:
            res.append(desc)

    is_found = bool(len(res))

    assert expected_found == is_found


def test_banch_of_find_imgs():

    find_pres = 80
    assert find_pres >= 80