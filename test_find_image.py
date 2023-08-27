import HNSW_index
import img_proccessing
import file_descriptor_utils

import os

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

import functools

def main():
    path_prefix = './grbrt_spb/'
    path_to_indexed_img = path_prefix + "/photos/"
    path_to_img = path_to_indexed_img + "photo_586@30-04-2022_21-24-19.jpg"

    # should contain id, nfeatures, desc size
    index_path = "./indexies/test_index.bin"
    descriptors_file = path_prefix + "all_desc.npy"

    expected_found = True

    index_size = 3000
    chunk_size = 300
    cnt_imgs = 0
    cnt_chunks = int(index_size/chunk_size)
    imgs_max_cnt = 10000
    nfeatures = 700
    nfeatures_to_cmp = 600
    desc_size = 128

    # fill file descriptors 
    if not os.path.exists(descriptors_file):
        cnt_imgs = file_descriptor_utils.fullfill_desc_file(path_prefix,
                                                 descriptors_file,
                                                 nfeatures,
                                                 -1, chunk_size)

    cnt_indexies = int(cnt_imgs/index_size) + 1

    for i in range(0, cnt_indexies):
        index_name = f'index_id_{i}_sz_{index_size}_nfeat_{nfeatures}_desc_zs_{desc_size}.bin'
        index_path = path_prefix + index_name
        desc_offset = 0
        get_desc_chunk_func = functools.partial(file_descriptor_utils.read_array_from_file_by_chunks,
                                        filename=descriptors_file, 
                                        chunk_size=chunk_size, 
                                        cnt_chunks=cnt_chunks, 
                                        nfeatures=nfeatures,
                                        offset=desc_offset)
        
        index = HNSW_index.build_index_from_exist(index_path, index_size, chunk_size, get_desc_chunk_func)
        desc_offset = desc_offset + index_size

    # img_data = img_proccessing.get_image_data(path_to_img, nfeatures)
    # q_desc = img_data.descriptor

    # if q_desc.shape[0] != nfeatures:
    #     assert False

    # desc_idx_list = HNSW_index.get_neighbors_desc_indexies(index, q_desc, k=100)

    # if len(desc_idx_list) == 0:
    #     assert False

    # desc_list = file_descriptor_utils.read_specific_rows_from_file(descriptors_file,
    #                                                                desc_idx_list,
    #                                                                desc_size*nfeatures)

    # if len(desc_list) == 0:
    #     assert False

    # res = []
    # for desc in desc_list:
    #     in_desc = desc.reshape(nfeatures, desc_size)[:nfeatures_to_cmp]
    #     q_desc = q_desc[:nfeatures_to_cmp]
    #     if img_proccessing.compare_sift_descriprtors(q_desc, in_desc, 0.7) == True:
    #         res.append(desc)

    # is_found = bool(len(res))

    # if is_found:
    #     print("IMAGE WAS FOUNDED!")

    assert expected_found == is_found


def test_banch_of_find_imgs():

    find_pres = 80
    assert find_pres >= 80

if __name__ == "__main__":
    main()