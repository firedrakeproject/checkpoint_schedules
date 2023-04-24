from checkpoint_schedules import HRevolveCheckpointSchedule
import time as tm
class Forward():
    """Define the a forward solver.

    """
    def __init__(self):
        self.exp = None
        self.chk_id = None
        self.steps = steps
        self.chk = None

    def advance(self, n_0: int, n_1: int) -> None:
        """Advance the foward equation.

        Parameters
        ----------
        n0
            Initial time step.
        n1
            Final time step.

        """
        print((">"*(n_1-n_0)).rjust(n_1))
        i_n = n_0
        while i_n < n_1:
            i_np1 = i_n + 1
            i_n = i_np1
        self.chk = i_n
           
    def getsteps(self) -> int:
        """Return the total time steps.

        """
        return self.steps
   

class Backward():
    """This object define the a forward solver.

    """
    def __init__(self):
        self.exp = None
        self.sol = None

    def advance(self, n_1: int, n_0: int) -> None:
        """Execute the backward equation.

        Parameters
        ----------
        n1
            Initial time step in reverse state.
        n0
            Final time step in reverse state.

        """
        print("<".rjust(n_1))
        i_n = n_1
        while i_n > n_0:
            i_np1 = i_n - 1
            i_n = i_np1

start = tm.time()
steps = 10
schk = 3
hrev_schedule = HRevolveCheckpointSchedule(steps, schk, 0)
fwd_schedule = hrev_schedule.get_forward_schedule()
bwd_schedule = hrev_schedule.get_reverse_schedule()

print("end")
# bwd_schedule = 
# cp_action = next(hrev_schedule)
# self.action_list.append(cp_action)
# action(cp_action)
# fwd = Forward()
# bwd = Backward()


# manage = Manage(fwd, bwd, schk, steps)
# manage.actions()
# end = tm.time()
# print(end-start)