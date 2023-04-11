class Backend:
    """Backend function.

    """
    def __init__(self, num_chk):
        self.checkpoint = []
        self.chk_id = []
        self.num_chk = num_chk

    def StoreCheckpoint(self, data, n_write: int) -> None:
        """Append the checkpoint data.

        """
        self.checkpoint.append((data, n_write))

    def GetCheckpoint(self):
        """Return the checkpoint data.

        """
        return self.checkpoint[self.num_chk-1]
    
    def DeleteCheckpoint(self):
        self.checkpoint.pop(self.num_chk-1)
    
    # def SetInitialCheckpoint(self, data) -> None:
    #     """Set the initial data to be used as initial condition.

    #     data
    #         Initial condition

    #     """
    #     self.initial_data = data
    
    # def GetInitialCheckpoint(self):
    #     """Return the initial data.

    #     data
    #         Initial condition set in Backend.SetInitialCheckpoint.
        
    #     See Also
    #     --------
    #     Backend.SetInitialCheckpoint
        
    #     """
    #     return self.initial_data



