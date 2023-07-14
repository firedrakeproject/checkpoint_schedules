from checkpoint_schedules import Forward, EndForward, Reverse, Copy, EndReverse
from checkpoint_schedules import Revolve
import functools

 
class CheckpointingManager():
    """Manage the forward and adjoint solvers.

    Attributes
    ----------
    save_ram : int
        Number of checkpoint that will be stored in RAM.
    save_disk : int
        Number of checkpoint that will be stored on disk.
    list_actions : list
        Store the list of actions.
    max_n : int
        Total steps used to execute the solvers.
    """
    def __init__(self, max_n, save_ram, save_disk):
        self.max_n = max_n
        self.save_ram = save_ram
        self.save_disk = save_disk
        self.list_actions = []
        
    def execute(self, cp_schedule):
        """Execute forward and adjoint with a checkpointing strategy.

        Parameters
        ----------
        cp_schedule : object
            Checkpointing schedule.
        """
        @functools.singledispatch
        def action(cp_action):
            raise TypeError("Unexpected action")

        @action.register(Forward)
        def action_forward(cp_action):
            nonlocal model_n
            if cp_action.write_ics:
                print(("*").rjust(cp_action.n0*4))

            print(("|" + "--->"*(cp_action.n1-cp_action.n0)).rjust(cp_action.n1*4) +
                   "   "*(self.max_n - cp_action.n1 + 4) + 
                   self.list_actions[len(self.list_actions) - 1])

            n1 = min(cp_action.n1, self.max_n)
            model_n = n1
            if cp_action.n1 == self.max_n:
                cp_schedule.finalize(n1)

        @action.register(Reverse)
        def action_reverse(cp_action):
            nonlocal model_r
            print(("<---"*(cp_action.n1-cp_action.n0) + "|").rjust(cp_action.n1*4) 
                  + "   "*(self.max_n - cp_action.n1 + 4) + 
                    self.list_actions[len(self.list_actions) - 1])

            model_r += cp_action.n1 - cp_action.n0
            
        @action.register(Copy)
        def action_copy(cp_action):
            print(("+").rjust(cp_action.n*4) 
                  + "   "*(self.max_n + 4) + 
                    self.list_actions[len(self.list_actions) - 1])

    
        @action.register(EndForward)
        def action_end_forward(cp_action):
            assert model_n == self.max_n
            print("End Forward")
            
        @action.register(EndReverse)
        def action_end_reverse(cp_action):
            nonlocal model_r
            assert model_r == self.max_n
            print("End Reverse")

        model_n = 0
        model_r = 0

        count = 0
        while True:
            cp_action = next(cp_schedule)
            self.list_actions.append(str(cp_action))
            action(cp_action)
            count += 1
            if isinstance(cp_action, EndReverse):  
                break


max_n = 4  # Total number of time steps.
save_ram = 1  # Number of steps to save i RAM.
save_disk = 2  # Number of steps to save in disk.

chk_manager = CheckpointingManager(max_n, save_ram, save_disk)  # manager object

revolver = Revolve(max_n, save_ram)
revolver.sequence()
chk_manager.execute(revolver)