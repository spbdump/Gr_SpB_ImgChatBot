import datetime
from enum import Enum

class DescriptorType(Enum):
    SIF = "SIF"
    SURF = "SURF"
    ORB = "ORB"
    OTHER = "OTHER"

class ImageData:
    def __init__(self, descriptor, desc_type, datestamp=datetime.datetime.now(), message_id=-1):

        self.descriptor = descriptor

        if not isinstance(desc_type, DescriptorType):
            raise TypeError("The descriptor must be an instance of the Descriptor enumerated type.")

        self.desc_type = desc_type
        self.datestamp = datestamp
        self.message_id = message_id

    def __repr__(self):
        return f"ImageData(descriptor='{self.descriptor}', datestamp='{self.datestamp}', post_id='{self.message_id}')"

    def __dict__(self):
        return {'descriptor': self.descriptor.tolist(), 'desc_type': str(self.desc_type), 'datestamp': self.datestamp, 'message_id': self.message_id }

