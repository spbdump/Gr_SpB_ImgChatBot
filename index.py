import numpy as np
import file_descriptor_utils as fd_u
import sqlite_db_utils as sq3_u
import img_proccessing as imp
from HNSW_index import create_index, add_data_batch

MAX_RUNTIME_INDEX_SIZE = 10

class Index:
    def __init__(self, index_id=0, index_size=0, 
                 max_size=0, nfeatures=0, 
                 desc_size=128, index_name="", desc_name=""):
        self.index_id = index_id
        self.index_size = index_size
        self.max_size = max_size
        self.nfeatures = nfeatures
        self.desc_size = desc_size
        self.index_name = index_name if index_name != "" else self.generate_index_name()
        self.desc_name = desc_name if desc_name != "" else self.generate_desc_name()

    def generate_desc_name(self):
        return (
            f'desc_id_{self.index_id}'
            f'_sz_{self.index_size}'
            f'_nfeat_{self.nfeatures}'
            f'_desc_sz_{self.desc_size}.npy'
        )
            
    def generate_index_name(self):
        return (
            f"index_id_{self.index_id}"
            f"_sz_{self.index_size}"
            f"_nfeat_{self.nfeatures}"
            f"_desc_sz_{self.desc_size}.bin"
        )



class ImageData:
    def __init__(self, index_id:int, img_id:int, t_msg_id:int, img_name:str):
        self.index_id = index_id
        self.img_id = img_id
        self.t_msg_id = t_msg_id
        self.img_name = img_name

    def __str__(self):
        return f"ImageData(\
                 index_id={self.index_id}, \
                 img_id={self.img_id}, \
                 t_msg_id={self.t_msg_id}, \
                 img_name='{self.img_name}')"

class RuntimeIndex(Index):
    RUNTIME_INDEX_ID = -1
    # def __init__(self, index_id=0, index_name="", index_size=0,
    #              max_size=0, nfeatures=0, desc_size=128,
    #              desc_name="", chat_path=""):
    #     super().__init__(index_id, index_name, index_size, max_size, nfeatures, desc_size, desc_name)
    def __init__(self, index:Index = Index(), k:int = 20):
        super().__init__(index.index_id, index.index_size, 
                         index.max_size, index.nfeatures,
                         index.desc_size, index.index_name,
                         index.desc_name)
        self.index_data = np.empty((0, self.nfeatures*self.desc_size), dtype=np.float32 )
        self.index = create_index()
        # must be integer divisible with self.max_size
        # make it as 1% of self.max_size ??
        self.max_runtime_size = MAX_RUNTIME_INDEX_SIZE
        self.img_data = []
        self.k = k
    
    def get_t_msg_id(self, img_id):
        msg_id = self.img_data[img_id].t_msg_id

        return msg_id

    def knn_query(self, query_data):
        # reurn tuple of indexies and distances
        neighbors_data = self.index.knnQuery(query_data.reshape(-1), k=self.k)
        return neighbors_data[0]
    
    def find_image(self, q_desc):
        res_img_id_list = []
        desc_idx_list = self.knn_query(q_desc)
        for idx in desc_idx_list:
            desc = self.index_data[idx]
            in_desc = desc.reshape(self.nfeatures, self.desc_size)[:self.nfeatures]
            if imp.compare_sift_descriprtors(q_desc, in_desc, 0.8) == True:
                res_img_id_list.append( idx )
        return res_img_id_list

    def add_data_point(self, data: np.ndarray, img_name:str, t_msg_id:int):
        id = self.index_data.shape[0]
        self.index_data = np.vstack( (self.index_data, data) )
        self.index.addDataPoint(id, data)
        self.index.createIndex(print_progress=False)
        
        # pos in img_data shoud be equal id
        self.img_data.append( ImageData(self.RUNTIME_INDEX_ID, id, t_msg_id, img_name) )

        
    def is_fullfilled(self):
        return self.index_data.shape[0] == self.max_runtime_size

    def dump(self, prefix_path):

        data_size = self.index_data.shape[0]
            
        # should be before updating index_size because it used in calc ids
        add_data_batch(prefix_path, self.index_name, self.index_size, self.index_data)
        sq3_u.save_runtime_img_data(self.index_id, self.index_size, self.img_data, prefix_path)
        #

        fd_u.append_array_with_same_width(prefix_path + self.desc_name, self.index_data)
        
        # to save correct index_size value to db
        # if there no such record
        self.index_size += data_size
        is_exist = sq3_u.does_index_exist(self.index_id, prefix_path)
        if not is_exist:
            sq3_u.add_index_record(self, prefix_path)
        else:
            sq3_u.update_index_size(self.index_id, data_size, prefix_path)
            

        # clear runtime index 
        self.index_data = np.empty((0, self.nfeatures*self.desc_size), dtype=np.float32 )
        self.index = create_index()
        self.img_data = []

        if self.index_size == self.max_size:
            self.index_size = 0
            self.index_id += 1
            self.desc_name = self.generate_desc_name()
            self.index_name = self.generate_index_name()
            # add arrecord of new empty index
            sq3_u.add_index_record(self, prefix_path)

        # try:
        #     with open(path_to_file, 'wb') as file:
        #         pickle.dump(self.index_data, file)
        # except Exception as e:
        #     print(f"An error occurred while saving the index: {e}")

    def clear_index(self):
        self.index_data = np.empty((0, self.nfeatures*self.desc_size), dtype=np.float32 )


CHAT_ID_INDEX_MAP = {}

def init_runtime_chat_indexes(prefix_path: str = './'):
    data = sq3_u.get_last_index_data_for_all(prefix_path)
    for chat_id, index in data:
        CHAT_ID_INDEX_MAP[chat_id] = RuntimeIndex(index)

def add_runtime_index(chat_id, index_id, max_size, nfeatures):
    index = Index(index_id=index_id, max_size=max_size, nfeatures=nfeatures)
    CHAT_ID_INDEX_MAP[chat_id] = RuntimeIndex(index)

def get_runtime_index(chat_id) -> RuntimeIndex : 
    return CHAT_ID_INDEX_MAP.get(chat_id, None)


