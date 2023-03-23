class Function():
    def __init__(self):
        pass
        self.chk_data = []
        self.chk_id = []

    def save_checkpoint(self, data):
        self.chk_data.append(data)
    
    def get_checkpoint_id(self, n_write):
        self.chk_id = n_write
