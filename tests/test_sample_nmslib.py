
import bot_impl
import utils

import nmslib
import numpy as np

import core.HNSW_index as HNSW_index

PATH_TO_IMGS = './grbrt_spb/photos/'

tmp_index = nmslib.init(method='hnsw', space='l2')

def add_desc_to_index(path_to_index, desc: np.ndarray, id:int, index):
    
    index.addDataPoint(id, desc.reshape(-1))
    
    index.createIndex(print_progress=True)
    index.saveIndex(path_to_index, save_data=False)
    
    return True

def main():
    global tmp_index
    prefix_img_path = PATH_TO_IMGS
    list_imgs = utils.get_image_files( prefix_img_path )
    rand_imgs_list = utils.get_random_images(list_imgs, 20)
    index = nmslib.init(method='hnsw', space='l2')

    data = np.empty((0, 800*128), dtype=np.float32)
    id = 0
    for img_name in rand_imgs_list:
        path_to_img = prefix_img_path + img_name
        img_data = bot_impl.get_image_data(path_to_img, 800)
        img_desc = img_data.descriptor
        
        if img_desc.shape[0] < 800:
            continue

        img_desc = img_desc[:800]

        add_desc_to_index('./debug_index.bin', img_desc, id, tmp_index)

        desc = img_desc.reshape(-1)
        # pos = index.addDataPoint(id, desc)
        # print(f'pos: {pos}, id: {id}')
        id = id + 1
        data = np.vstack((data, desc))

    # index.createIndex(print_progress=True)
    tmp_index.saveIndex('./debug_index.bin', save_data=False)
    # del index
    index = nmslib.init(method='hnsw', space='l2')
    index.loadIndex('./debug_index.bin')

    # for img_name in rand_imgs_list[10:]:
    #     path_to_img = prefix_img_path + img_name
    #     img_data = bot_general.get_image_data(path_to_img, 800)
    #     img_desc = img_data.descriptor
        
    #     if img_desc.shape[0] < 800:
    #         continue

    #     img_desc = img_desc[:800]

    #     desc = img_desc.reshape(-1)
    #     pos = index.addDataPoint(id, desc)
    #     print(f'pos: {pos}, id: {id}')
    #     id = id + 1
    #     data = np.vstack((data, desc))
    # index.createIndex(print_progress=True)
    
    query_id = 5
    # query for the nearest neighbours of the first datapoint
    ids, distances = index.knnQuery(data[query_id], k=10)

    print(ids)


if __name__ == "__main__":
    main()