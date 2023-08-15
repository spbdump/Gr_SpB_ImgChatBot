
import os

import HNSW_index
import file_descriptor_utils


def test_build_index_from_exist_data():
    index_path = "./indexies/test_index.bin"

    # fill file descriptors 
    file_descriptor_utils.fullfill_desc_file("./src_images/indexed_imgs/", "./descriptors/")
    index = HNSW_index.build_index_from_exist(index_path, 10000)

    assert os.path.exists(index_path)


