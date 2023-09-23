

class Context:
    def __init__(self, nfeatures, desc_size, max_size, chat_path, chat_id):
        self.nfeatures = nfeatures
        self.desc_size = desc_size
        self.max_size = max_size
        self.chat_path = chat_path
        self.chat_id = chat_id

    def __str__(self):
        return f"Context(nfeatures={self.nfeatures}, desc_size={self.desc_size}, max_size={self.max_size}, " \
               f"chat_path='{self.chat_path}', chat_id={self.chat_id})"

