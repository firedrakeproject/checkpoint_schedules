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
