import HNSW_index
import img_proccessing
import file_descriptor_utils

import os

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

import functools

def main():
    path_to_indexed_img = "./src_imgs/indexed_imgs/"
    path_to_img = path_to_indexed_img + "photo_586@30-04-2022_21-24-19.jpg"

    index_path = "./indexies/test_index.bin"
    descriptors_file = "./descriptors/test_desc.npy"

    expected_found = True

    chunk_size = 200
    imgs_max_cnt = 1000
    nfeatures = 1000
    desc_size = 128

    # fill file descriptors 
    if not os.path.exists(descriptors_file):
        file_descriptor_utils.fullfill_desc_file(path_to_indexed_img,
                                                descriptors_file,
                                                imgs_max_cnt, chunk_size)

    get_desc_chunk_func = functools.partial(file_descriptor_utils.read_array_from_file_by_chunks,
                                            filename=descriptors_file, 
                                            chunk_size=chunk_size, nfeatures=nfeatures)
    if os.path.exists(index_path):
        index = HNSW_index.load_index(index_path)
    else:
        index = HNSW_index.build_index_from_exist(index_path, 10000, chunk_size, get_desc_chunk_func)

    img_data = img_proccessing.get_image_data(path_to_img)
    q_desc = img_data.descriptor

    if q_desc.shape[0] != nfeatures:
        assert False

    desc_idx_list = HNSW_index.get_neighbors_desc_indexies(index, q_desc, k=100)

    if len(desc_idx_list) == 0:
        assert False

    desc_list = file_descriptor_utils.read_specific_rows_from_file(descriptors_file,
                                                                   desc_idx_list,
                                                                   desc_size*nfeatures)

    if len(desc_list) == 0:
        assert False

    res = []
    for desc in desc_list:
        in_desc = desc.reshape(nfeatures, desc_size)[:700]
        q_desc = q_desc[:700]
        if img_proccessing.compare_sift_descriprtors(q_desc, in_desc, 0.7) == True:
            res.append(desc)

    is_found = bool(len(res))

    if is_found:
        print("IMAGE WAS FOUNDED!")

    assert expected_found == is_found


def test_banch_of_find_imgs():

    find_pres = 80
    assert find_pres >= 80

if __name__ == "__main__":
    main()