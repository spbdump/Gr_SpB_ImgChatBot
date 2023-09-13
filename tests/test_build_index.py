import os
import file_descriptor_utils
import HNSW_index

import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

def test_build_index_from_exist_data():
    index_path = "./indexies/test_index.bin"
    # file_name = "{prefix_path}/indexies/index_optim_{b_idx}_maxsz_{sz}.bin"
    #   .format(prefix_path=path_to_indexies, b_idx=batch_idx, sz=index_max_size)

    chunk_size = 200
    imgs_max_cnt = 500
    # fill file descriptors 
    file_descriptor_utils.fullfill_desc_file("./src_imgs/indexed_imgs",
                                             "./descriptors/test_desc.npy",
                                             imgs_max_cnt, chunk_size)

    index = HNSW_index.build_index_from_exist(index_path, 10000)

    assert os.path.exists(index_path)


