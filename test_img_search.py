import HNSW_index
import build_db
import img_proccessing
import db_utils

import time

def main():

    if not db_utils.check_connection() :
        return

    image_data = img_proccessing.get_image_data("./photos/photo_1628@05-11-2022_00-23-36.jpg")
    query_desc = image_data.descriptor

    # build_db.fullfill("./photos")
    
    # build index on each bot start
    HNSW_index.build_hnsw_index()

    start = time.time()
    
    res_arr = img_proccessing.poces_similar_sift_descriprors(query_desc)

    print("size res:", len(res_arr))
    for res in res_arr:
        print(res["_id"], res["img_name"])

    end = time.time()
    print("time to find match : %d sec", end - start)



if __name__ == "__main__":
    main()