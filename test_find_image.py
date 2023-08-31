import HNSW_index
import img_proccessing
import file_descriptor_utils
import sqlite_db_utils

import os
from datetime import datetime


import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

import functools

def main():
    path_prefix = './grbrt_spb/'
    path_to_indexed_img = path_prefix + "/photos/"
    path_to_img = path_to_indexed_img + "photo_586@30-04-2022_21-24-19.jpg"


    expected_found = True

    cnt_imgs = 3800

    index_size = 1000
    chunk_size = 200
    cnt_chunks = int(index_size/chunk_size)
    imgs_max_cnt = 10000
    nfeatures_to_cmp = 600
    nfeatures = 700
    desc_size = 128
    metadata_file = path_prefix + 'metadata.json'
    cnt_indexies = int(cnt_imgs/index_size) + 1

    path_to_imgs_dir = path_prefix + 'photos/'
    list_imgs = file_descriptor_utils.get_image_files(path_to_imgs_dir)
    if imgs_max_cnt > 0 and imgs_max_cnt < len(list_imgs) :
        list_imgs = list_imgs[:imgs_max_cnt]

    sqlite_db_utils.create_iamge_table(path_prefix)
    sqlite_db_utils.create_index_table(path_prefix)

    for i in range(0, cnt_indexies):
        index_name = f'index_id_{i}_sz_{index_size}_nfeat_{nfeatures}_desc_sz_{desc_size}.bin'
        desc_name =  f'desc_id_{i}_sz_{index_size}_nfeat_{nfeatures}_desc_sz_{desc_size}.bin'

        descriptors_file = path_prefix + desc_name
        # fill file descriptors 
        if not os.path.exists(descriptors_file):
            imgs_offset = i*index_size
            file_descriptor_utils.fullfill_desc_file(path_prefix,
                                                    list_imgs[imgs_offset:imgs_offset+index_size],
                                                    descriptors_file,
                                                    i,
                                                    nfeatures,
                                                    chunk_size)
        else:
            cnt_imgs = file_descriptor_utils.get_array_rows_count(descriptors_file, desc_size*nfeatures)
        print("count descriptors in desc file: ", cnt_imgs)

        index_path = path_prefix + index_name
        get_desc_chunk_func = functools.partial(file_descriptor_utils.read_array_from_file_by_chunks,
                                        filename=descriptors_file, 
                                        chunk_size=chunk_size, 
                                        cnt_chunks=cnt_chunks, 
                                        nfeatures=nfeatures)

        index, builded_index_size = HNSW_index.build_index_from_exist(index_path, chunk_size, get_desc_chunk_func)

        index_rec = {
            "index_name": index_name,
            "desc_name" : desc_name,
            "index_id"  : i,
            "max_size"  : index_size,
            "nfeatures" : nfeatures,
            "desc_size" : desc_size,
            "index_size": builded_index_size,
            # "updated_at": str(datetime.now()),
        }
        sqlite_db_utils.add_index_record(index_rec)

        # HNSW_index.update_metadata("curr_index", index_name, metadata_file)
        # HNSW_index.append_metadata(index_name, metadata, metadata_file)

    img_data = img_proccessing.get_image_data(path_to_img, nfeatures)
    q_desc = img_data.descriptor

    if q_desc.shape[0] != nfeatures:
        assert False

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