import HNSW_index
import build_db
import img_proccessing
import db_utils

import time

def main():

    db_utils.check_connection()

    image_data = img_proccessing.get_image_data("./photos/photo_1476@23-10-2022_17-14-18.jpg")
    query_desc = image_data.descriptor

    print( query_desc.shape )
    # build_db.fullfill("./photos")
    # HNSW_index.build_hnsw_index()
    
    # load index on each bot start
    HNSW_index.load_hnsw_indexies()

    start = time.time()
    
    res_arr = img_proccessing.poces_similar_sift_descriprors(query_desc)

    print("size res:", len(res_arr))
    for res in res_arr:
        print(res["_id"], res["img_name"])

    end = time.time()
    print("time to find match: sec", end - start)



if __name__ == "__main__":
    main()