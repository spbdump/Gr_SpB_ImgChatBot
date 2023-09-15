
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