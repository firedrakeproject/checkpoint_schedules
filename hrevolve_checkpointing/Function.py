class Function:
    """Backend function.

    """
    def __init__(self):
        self.checkpoint = []
        self.chk_id = []
        
    def StoreCheckpoint(self, data) -> None:
        """Append the checkpoint data.

        """
        self.checkpoint.append(data)

    def GetCheckpoint(self):
        """Return the checkpoint data.

        """
        len_chk = len(self.checkpoint)

        return self.checkpoint[len_chk-1]
    
    def DeleteCheckpoint(self) -> None:
        """Delete checkpoint data.

        """
        len_chk = len(self.checkpoint)
        self.checkpoint.pop(len_chk-1)