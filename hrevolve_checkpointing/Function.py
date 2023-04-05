class Backend:
    """Backend function.

    """
    def __init__(self):
        self.chk_data = []
        self.chk_id = []

    def store_checkpoint(self, data) -> None:
        """Append the checkpoint data.

        """
        self.chk_data.append(data)
    
    def get_checkpoint_id(self, n_write: int) -> None:
        """Collect the checkpoint identity.
    
        """
        self.chk_id = n_write

    def pop_checkpoint(self) -> None:
        """Employ the pop method to remove the latest checkpoint data.

        """
        l = len(self.chk_data)
        self.chk_data.pop(l-1)

    def get_checkpoint(self):
        """Return the latest checkpoint data stored in the list.

        """
        l = len(self.chk_data)
        return self.chk_data[l-1]
