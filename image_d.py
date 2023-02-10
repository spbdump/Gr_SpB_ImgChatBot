import datetime
from enum import Enum

class DescriptorType(Enum):
    SIF = "SIF"
    SURF = "SURF"
    ORB = "ORB"
    OTHER = "OTHER"

class ImageData:
    def __init__(self, descriptor, desc_type, datestamp=datetime.datetime.now(), post_id=None):

        self.descriptor = descriptor

        if not isinstance(desc_type, DescriptorType):
            raise TypeError("The descriptor must be an instance of the Descriptor enumerated type.")

        self.desc_type = desc_type
        self.datestamp = datestamp
        self.post_id = post_id

    def __repr__(self):
        return f"ImageData(descriptor='{self.descriptor}', datestamp='{self.datestamp}', post_id='{self.post_id}')"

