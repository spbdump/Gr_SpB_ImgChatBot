import numpy as np
from HNSW_index import create_index, add_data_batch
from file_descriptor_utils import append_array_with_same_width
from sqlite_db_utils import update_index_size, \
                            save_runtime_img_data, \
                            add_index_record, \
                            get_last_index_data_for_all, \
                            does_index_exist

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
        index_id = self.index_id if self.index_id == 0 else self.index_id + 1
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index_data = np.empty((0, self.nfeatures*self.desc_size), dtype=np.float32 )
        self.index = create_index()
        self.max_runtime_size = 10 # must be integer divisible with self.max_size
        self.img_data = []
        self.k = 20
    
    def get_t_msg_id(self, img_id):
        msg_id = self.img_data[img_id].t_msg_id

        return msg_id

    def knn_query(self, query_data):
        # reurn tuple of indexies and distances
        neighbors_data = self.index.knnQuery(query_data.reshape(-1), k=self.k)
        return neighbors_data[0]
    
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
        save_runtime_img_data(self.index_id, self.index_size, self.img_data, prefix_path)
        #

        append_array_with_same_width(prefix_path + self.desc_name, self.index_data)
        
        # to save correct index_size value to db
        # if there no such record
        self.index_size += data_size
        is_exist = does_index_exist(self.index_id, prefix_path)
        if not is_exist:
            add_index_record(self, prefix_path)
        else:
            update_index_size(self.index_id, data_size, prefix_path)
            

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
            add_index_record(self, prefix_path)

        # try:
        #     with open(path_to_file, 'wb') as file:
        #         pickle.dump(self.index_data, file)
        # except Exception as e:
        #     print(f"An error occurred while saving the index: {e}")

    def clear_index(self):
        self.index_data = np.empty((0, self.nfeatures*self.desc_size), dtype=np.float32 )


CHAT_ID_INDEX_MAP = {}

def init_runtime_chat_indexes(prefix_path: str = './'):
    data = get_last_index_data_for_all(prefix_path)
    for chat_id, index_data in data:
        CHAT_ID_INDEX_MAP[chat_id] = RuntimeIndex(index_data)

def add_runtime_index(chat_id, index_id, max_size, nfeatures):
    #index = Index(index_id=index_id, max_size=max_size, nfeatures=nfeatures)
    CHAT_ID_INDEX_MAP[chat_id] = RuntimeIndex(index_id=index_id, max_size=max_size, nfeatures=nfeatures)

def get_runtime_index(chat_id) -> RuntimeIndex : 
    return CHAT_ID_INDEX_MAP.get(chat_id, None)


