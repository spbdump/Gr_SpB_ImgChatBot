from enum import Enum

class DescriptorType(Enum):
    SIFT = "SIFT"
    SURF = "SURF"
    ORB = "ORB"
    OTHER = "OTHER"

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