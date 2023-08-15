import os
import file_descriptor_utils
import HNSW_index


def test_build_index_from_exist_data():
    index_path = "./indexies/test_index.bin"

    chunk_size = 200
    # fill file descriptors 
    file_descriptor_utils.fullfill_desc_file("./src_imgs/indexed_imgs", "./descriptors/test_desc.npy", chunk_size)
    index = HNSW_index.build_index_from_exist(index_path, 10000)

    assert os.path.exists(index_path)


